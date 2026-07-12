import json
import time
from pathlib import Path
from dataclasses import asdict

import torch
from torch.amp import autocast

from models.training_model import Transformer, ModelArgs
from utils.tokenizer import Tokenizer

def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def sample_with_t_topp(logits, temperature=0.8, top_p=0.9):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, sampled_idx)

    return next_token.item()

def apply_repetition_penalty(logits, input_ids, penalty=1.2, window_size=64):
    logits = logits.clone()

    recent_tokens = input_ids[0, -window_size:]

    for token_id in set(input_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty

    return logits

def sample_token_from_logit(logits, input_ids, keepdim=False):
    process_logits = apply_repetition_penalty(logits, input_ids)
    token = sample_with_t_topp(process_logits)
    # token = torch.argmax(logits, dim=-1, keepdim=keepdim)

    return token

@torch.no_grad()
def forward_all_heads(model, input_ids):
    autocast_enabled = (input_ids.device.type == "cuda")
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        h_trunk, freqs_cis, mask = model.forward_trunk(input_ids)

    head_logits_list = []
    for head_idx in range(model.n_future_tokens):
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits, h = model.forward_head_from_trunk(
                h_trunk=h_trunk,
                head_index=head_idx,
                freqs_cis=freqs_cis,
                mask=mask,
            )
        head_logits_list.append(logits)

    return h_trunk, freqs_cis, mask, head_logits_list

@torch.no_grad()
def sampler_draft_from_head_logits(
    head_logits_list,
    input_ids,
    pos: int,
    max_tokens: int | None = None,
):
    n_heads = len(head_logits_list)
    if max_tokens is not None:
        n_heads = min(n_heads, max_tokens)

    block_tokens = []
    for head_idx in range(n_heads):
        logits = head_logits_list[head_idx][:, pos, :]
        token = sample_token_from_logit(logits, input_ids)
        block_tokens.append(token)

    return block_tokens

@torch.no_grad()
def verify_block_with_main_logits(
    main_logits,
    input_ids,
    prefix_len_before_block: int,
    block_tokens: list[int],
):
    if len(block_tokens) == 0:
        return 0, False, None

    accepted = 1
    mismatch_at = None

    for draft_idx in range(1, len(block_tokens)):
        pos = prefix_len_before_block + draft_idx - 1
        main_pred = sample_token_from_logit(main_logits[:, pos, :], input_ids)

        drafted_token = block_tokens[draft_idx]

        if main_pred != drafted_token:
            mismatch_at = draft_idx
            break

        accepted += 1

    verified = mismatch_at is None
    return accepted, verified, mismatch_at

def make_initial_draft(input_ids, model, remain_tokens, num_heads_used):
        _, _, _, head_logits_list = forward_all_heads(model, input_ids)
        pos = input_ids.shape[1] - 1

        max_tokens = min(num_heads_used, remain_tokens)
        draft = sampler_draft_from_head_logits(
            head_logits_list=head_logits_list[:num_heads_used],
            input_ids=input_ids,
            pos=pos,
            max_tokens=max_tokens,
        )
        return draft

@torch.no_grad()
def inference_warmup(warmup_iters, model, device, gen_len, num_heads_used, max_seq_len, warm_input_ids):
    for _ in range(warmup_iters):
        remain = min(gen_len, num_heads_used)
        if warm_input_ids.shape[1] >= max_seq_len:
            warm_input_ids = warm_input_ids[:, -(max_seq_len - 1):]

        draft = make_initial_draft(warm_input_ids, model, remain, num_heads_used)
        if len(draft) == 0:
            break

        if warm_input_ids.shape[1] + len(draft) > max_seq_len:
            keep_len = max(1, max_seq_len - len(draft))
            warm_input_ids = warm_input_ids[:, -keep_len:]
            draft = make_initial_draft(warm_input_ids, model, remain, num_heads_used)

        block_tensor = torch.tensor([draft], dtype=torch.long, device=device)
        seq_with_block = torch.cat([warm_input_ids, block_tensor], dim=1)

        _, _, _, head_logits_list = forward_all_heads(model, seq_with_block)
        main_logits = head_logits_list[0]

        accepted, _, _ = verify_block_with_main_logits(
            main_logits=main_logits,
            input_ids=warm_input_ids,
            prefix_len_before_block=warm_input_ids.shape[1],
            block_tokens=draft,
        )

        warm_input_ids = seq_with_block[:, :warm_input_ids.shape[1] + accepted]

        if warm_input_ids.shape[1] > max_seq_len:
            warm_input_ids = warm_input_ids[:, -max_seq_len:]

@torch.no_grad()
def run_mtp_verify_speed_test(
    model,
    tokenizer,
    prompt_text="The meaning of life is",
    prompt_ids=None,
    gen_len=1024,
    warmup_iters=10,
    device="cuda",
    eos_id=None,
    num_heads_used=None,
    allow_sliding_window=True,
):
    device = torch.device(device)
    model.eval()

    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt_text, True, False)

    if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be a non-empty list[int].")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    original_prompt_len = input_ids.shape[1]

    max_seq_len = model.params.max_seq_len
    total_heads = model.n_future_tokens

    if num_heads_used is None:
        num_heads_used = total_heads
    num_heads_used = min(num_heads_used, total_heads)

    if num_heads_used < 1:
        raise ValueError("num_heads_used must be >= 1.")

    # 如果 prompt 本身太長，先裁切到可 forward 的長度
    if input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, -max_seq_len:]

    # warmup：只暖機，不統計
    warm_input_ids = input_ids.clone()
    inference_warmup(warmup_iters, model, device, gen_len, num_heads_used, max_seq_len, warm_input_ids)

    # reset
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, -max_seq_len:]

    total_generated = 0
    total_model_forwards = 0
    verification_forwards = 0
    recompute_forwards = 0

    accepted_token_hist = []
    verify_pass_count = 0
    verify_fail_count = 0
    mismatch_hist = []
    per_step_latency = []

    sync_if_cuda(device)
    start_time = time.perf_counter()

    remain = gen_len - total_generated
    draft = make_initial_draft(input_ids, model, remain, num_heads_used)
    total_model_forwards += 1

    while total_generated < gen_len:
        remain = gen_len - total_generated
        draft = draft[:remain]

        if len(draft) == 0:
            break

        if input_ids.shape[1] + len(draft) > max_seq_len:
            if not allow_sliding_window:
                raise RuntimeError(
                    "input_ids + draft exceeds max_seq_len. "
                    "Set larger max_seq_len or enable allow_sliding_window=True."
                )

            keep_len = max(1, max_seq_len - len(draft))
            input_ids = input_ids[:, -keep_len:]

            remain = gen_len - total_generated
            draft = make_initial_draft(input_ids, model, remain, num_heads_used)
            total_model_forwards += 1
            recompute_forwards += 1
            continue

        sync_if_cuda(device)
        t0 = time.perf_counter()

        prefix_len = input_ids.shape[1]

        block_tensor = torch.tensor([draft], dtype=torch.long, device=device)
        seq_with_block = torch.cat([input_ids, block_tensor], dim=1)

        _, _, _, head_logits_list = forward_all_heads(model, seq_with_block)
        total_model_forwards += 1
        verification_forwards += 1

        main_logits = head_logits_list[0]

        accepted, verified, mismatch_at = verify_block_with_main_logits(
            main_logits=main_logits,
            input_ids=input_ids,
            prefix_len_before_block=prefix_len,
            block_tokens=draft,
        )

        if accepted <= 0:
            raise RuntimeError("accepted should never be <= 0 in greedy self-speculative decoding.")

        new_len = prefix_len + accepted
        input_ids = seq_with_block[:, :new_len]

        total_generated += accepted
        accepted_token_hist.append(accepted)
        mismatch_hist.append(mismatch_at)

        if verified:
            verify_pass_count += 1
        else:
            verify_fail_count += 1

        if eos_id is not None:
            generated_part = input_ids[0, prefix_len:new_len].tolist()
            if eos_id in generated_part:
                eos_pos = generated_part.index(eos_id)
                keep_len = prefix_len + eos_pos + 1
                input_ids = input_ids[:, :keep_len]

                total_generated = total_generated - accepted + eos_pos + 1
                break

        if total_generated >= gen_len:
            break

        next_pos = input_ids.shape[1] - 1
        remain = gen_len - total_generated

        draft = sampler_draft_from_head_logits(
            head_logits_list=head_logits_list[:num_heads_used],
            input_ids=input_ids,
            pos=next_pos,
            max_tokens=min(num_heads_used, remain),
        )

        sync_if_cuda(device)
        t1 = time.perf_counter()
        per_step_latency.append(t1 - t0)

    sync_if_cuda(device)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    steps = max(len(accepted_token_hist), 1)
    total_accepted_tokens = sum(accepted_token_hist)

    result = {
        "mode": "paper_greedy_self_speculative",
        "prompt_len": original_prompt_len,
        "generated_tokens": total_generated,

        # 所有真的呼叫 forward_all_heads 的次數
        "total_model_forwards": total_model_forwards,

        # steady-state verification forward 次數
        # 論文的 tokens / forward 比較接近看這個概念
        "verification_forwards": verification_forwards,

        "recompute_forwards_due_to_sliding_window": recompute_forwards,

        # 保守版：把 initial draft forward 和 sliding recompute 都算進去
        "tokens_per_total_model_forward": (
            total_generated / max(total_model_forwards, 1)
        ),

        # 論文式 steady-state 版本：
        # 每個 verification forward 同時 verify previous draft + predict next draft
        "tokens_per_verification_forward": (
            total_generated / max(verification_forwards, 1)
        ),

        "tokens_per_sec": total_generated / max(total_time, 1e-12),
        "total_time_sec": total_time,
        "avg_step_latency_sec": sum(per_step_latency) / max(len(per_step_latency), 1),

        "avg_accepted_tokens_per_step": total_accepted_tokens / steps,
        "acceptance_rate_vs_block": (
            total_accepted_tokens / max(steps * num_heads_used, 1)
        ),
        "mean_extra_accept_len_excluding_main": (
            (total_accepted_tokens - steps) / steps
        ),

        "verify_pass_count": verify_pass_count,
        "verify_fail_count": verify_fail_count,
        "accepted_token_hist": accepted_token_hist,
        "mismatch_hist": mismatch_hist,

        "num_heads_used": num_heads_used,
        "model_n_future_tokens": model.n_future_tokens,
        "max_seq_len": max_seq_len,
        "allow_sliding_window": allow_sliding_window,
    }

    return input_ids, result

@torch.no_grad()
def run_vanilla_greedy_head1(
    model,
    tokenizer,
    prompt_text="The meaning of life is",
    prompt_ids=None,
    gen_len=128,
    device="cuda",
    eos_id=None,
):
    device = torch.device(device)
    model.eval()

    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt_text, True, False)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    total_forwards = 0

    for _ in range(gen_len):
        if input_ids.shape[1] > model.params.max_seq_len:
            input_ids = input_ids[:, -model.params.max_seq_len:]

        _, _, _, head_logits_list = forward_all_heads(model, input_ids)
        total_forwards += 1

        logits = head_logits_list[0][:, -1, :]
        next_token = sample_token_from_logit(logits, input_ids, True)
        next_token = torch.tensor([[next_token]], dtype=torch.long, device=input_ids.device)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    return input_ids, {
        "mode": "vanilla_greedy_head1",
        "generated_tokens": input_ids.shape[1] - len(prompt_ids),
        "total_forwards": total_forwards,
        "tokens_per_forward": (input_ids.shape[1] - len(prompt_ids)) / max(total_forwards, 1),
    }
