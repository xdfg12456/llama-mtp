from models.training_model import Transformer, ModelArgs
from utils.tokenizer import Tokenizer
import time
import json
import random
import numpy as np
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from collections import Counter
from utils.print_to_file import print_to_file, create_output_file_path

from training_datasets.pretrain.openweb_text_dataset import preprocess_datasets, collate_fn

from avg_token_acceptance_eval import avg_token_acceptance
from token_acceptence_with_length_eval import eval_accept_rate_for_length
from ppl_eval import evaluate_total_loss, calc_ppl
from utils.generate import run_vanilla_greedy_head1
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def init(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'[INFO] device: {device}')

    return device

def load_model(
    ckpt_dir,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
):
    start_time = time.time()

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[-1]
    print(f"[INFO]: model choosed {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    model.load_state_dict(checkpoint, strict=False)

    print(f"[INFO]: Loaded in {time.time() - start_time:.2f} seconds")

    return model, tokenizer

def avg_token_acceptance_test(model, dataloader, device, max_batches, model_name, output_file_path):
    avg_accept = avg_token_acceptance(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=max_batches,
        )

    title = f"\n===== {model_name} avg token acceptance Summary ====="
    result = f"\n{avg_accept:.4f}"
    message = title+result

    print_to_file(message, output_file_path)

def ppl_test(model, dataloader, device, model_name, output_file_path):
    n_future_token = model.params.n_future_tokens
    
    outputs = evaluate_total_loss(model, dataloader, device)
    ppls = calc_ppl(outputs, n_future_token)

    title = f"\n===== {model_name} PPL Summary ====="
    message = title
    for i in range(n_future_token):
        result = f"\n{i+1}: {ppls[i]}"
        message = message + result

    print_to_file(message, output_file_path)

def token_accept_with_length_test(model, dataloader, device, batche_size, lengths, max_seq_length, random_slice, model_name, output_file_path):
    lengths = [int(x) for x in lengths.split(",") if x.strip()]

    results = {}

    for L in lengths:
        if L + 2 > max_seq_length:
            continue

        avg_accept = eval_accept_rate_for_length(
            model=model,
            dataloader=dataloader,
            prefix_len=L,
            device=device,
            max_batches=batche_size,
            random_slice = random_slice
        )
        results[L] = avg_accept

    title = f"\n===== {model_name} prefix vs token accept Summary ====="
    message = title
    for L in lengths:
        result = f"\n{L}: {results[L]:.4f}"
        message = message + result

    print_to_file(message, output_file_path)

def generative_ppl_test(model, eval_loader, gen_len, tokenizer, eos_id, device, max_batches, max_length, model_name, output_file_path):
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
    gpt2_model.eval()

    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    total_nll = 0.0
    total_tokens = 0
    generated_input_ids = []

    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)   # [B, T]

        B = input_ids.size(0)

        for b in range(B):
            prompt_ids = input_ids[b, :128].tolist()

            prompt_ids = [int(x) for x in prompt_ids if int(x) >= 0]

            if len(prompt_ids) == 0:
                continue

            generated, _ = run_vanilla_greedy_head1(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                gen_len=gen_len,
                device=str(device),
                eos_id=eos_id,
            )

            if torch.is_tensor(generated):
                generated = generated.squeeze(0).detach().cpu().tolist()

            generated_input_ids.append({
                "prompt_ids": list(map(int, prompt_ids)),
                "generated_ids": list(map(int, generated)),
            })

    all_tokens = []
    total_ngrams = 0.0
    repeated_ngrams = 0.0
    for sample in generated_input_ids:
        prompt_ids = sample["prompt_ids"]
        generated_ids = sample["generated_ids"]

        continuation_ids = generated_ids[len(prompt_ids):]

        ngrams = [
                tuple(continuation_ids[i:i + 4])
                for i in range(len(continuation_ids) - 4 + 1)
            ]

        counts = Counter(ngrams)

        total_ngrams += len(ngrams)
        repeated_ngrams += sum(c - 1 for c in counts.values() if c > 1)

        for token_id in continuation_ids:
            token_id = int(token_id)
            if token_id >= 0:
                all_tokens.append(token_id)

        prompt_text = tokenizer.decode(prompt_ids)
        continuation_text = tokenizer.decode(continuation_ids)

        if len(continuation_text.strip()) == 0:
            continue

        full_text = prompt_text + continuation_text

        full_enc = gpt2_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        prompt_enc = gpt2_tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        input_ids = full_enc["input_ids"].to(device)
        labels = input_ids.clone()

        prompt_len = prompt_enc["input_ids"].size(1)
        labels[:, :prompt_len] = -100
        if (labels != -100).sum().item() == 0:
            continue

        outputs = gpt2_model(input_ids=input_ids, labels=labels)

        valid_tokens = (labels[:, 1:] != -100).sum().item()

        total_nll += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)

    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    counts = torch.bincount(token_tensor)
    probs = counts[counts > 0].float()
    probs = probs / probs.sum()
    entropy_nats = float(-(probs * probs.log()).sum().item())

    title = f"\n===== {model_name} Generate Summary ====="
    result = f"\ngenerative ppl: {ppl:.4f}\ngenerative entropy: {entropy_nats:.4f}\n4-gram repetition: {(repeated_ngrams/total_ngrams):.4f}"
    message = title+result

    print_to_file(message, output_file_path)

def main(
    seed: int = 1,
    ckpt_dir: str = '30M_2_sdc',
    tokenizer_path: str = "./tokenizer.model",
    max_seq_len: int = 1024,
    per_device_train_batch_size: int = 8,
    prefix_lengths: str = "128,256,512,768",
    output_dir: str = "./log",
):
    print(f"seed: {seed}")
    device = init(seed=seed)

    test_ds, valid_ds = preprocess_datasets('test')
    test_loader = DataLoader(
        valid_ds,
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    output_file_path = create_output_file_path(output_dir, 'all_eval')
    
    max_batches = 500

    model, tokenizer = load_model(ckpt_dir, tokenizer_path, max_seq_len, per_device_train_batch_size)
    eos_id = tokenizer.eos_id
    model_name = str(ckpt_dir)

    model.to(device)
    model.eval()
    
    ppl_test(model, test_loader, device, model_name, output_file_path)
    avg_token_acceptance_test(model, test_loader, device, max_batches, model_name, output_file_path)
    token_accept_with_length_test(model, test_loader, device, max_batches, prefix_lengths, max_seq_len, True, model_name, output_file_path)
    generative_ppl_test(model, test_loader, max_seq_len, tokenizer, eos_id, device, 25, max_seq_len, model_name, output_file_path)
    
if __name__ == "__main__":
    import fire

    fire.Fire(main)