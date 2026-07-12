import json
import time
import math
from pathlib import Path
import wandb
import random
import numpy as np
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from models.training_model import Transformer, ModelArgs
from utils.tokenizer import Tokenizer

from training_datasets.pretrain.openweb_text_dataset import preprocess_datasets, collate_fn

def init(seed: int = 1, model_args = None, ckpt_dir = ""):
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
    print(f"\n===== model params =====")
    print(json.dumps(model_args, indent=2, ensure_ascii=False))
    
    run = wandb.init(
        entity="gqop0919-ai13-national-yang-ming-chiao-tung-university",
        project="llama-mtp",
        name=ckpt_dir,
        config=model_args,
    )

    return device, run

def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    from_scrach: bool,
):
    start_time = time.time()

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    
    if from_scrach == False:
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        ckpt_path = checkpoints[-1]
        print(f"[INFO]: model choosed {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

    print(f"[INFO]: Loaded in {time.time() - start_time:.2f} seconds")

    return model, tokenizer, asdict(model_args)

def token_confidence_from_logits(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    entropy_mtp = -(probs * log_probs).sum(dim=-1)
    V = logits.size(-1)
    H_max = torch.log(torch.tensor(float(V), device=logits.device))

    confidence = 1.0 - entropy_mtp / H_max

    return confidence

def calc_sdc_loss(student_logit, teacher_logit, min_confidence, soft=False):
    assert student_logit.shape == teacher_logit.shape

    with torch.no_grad():
        teacher_conf = token_confidence_from_logits(teacher_logit)
        mean_confidence = teacher_conf.mean()

        if soft:
            weight_or_mask = (teacher_conf - min_confidence) / (1.0 - min_confidence)
            weight_or_mask = torch.clamp(weight_or_mask, min=0.0, max=1.0)
        else:
            weight_or_mask = (teacher_conf > min_confidence).float()

    logp_student = F.log_softmax(student_logit, dim=-1)
    prob_teacher = F.softmax(teacher_logit, dim=-1)

    kl_all = F.kl_div(logp_student, prob_teacher, reduction="none").sum(dim=-1)

    weighted_kl = kl_all * weight_or_mask

    denom = weight_or_mask.sum().clamp(min=1.0)
    loss = weighted_kl.sum() / denom

    return loss, mean_confidence

def main(
    seed: int = 1,
    ckpt_dir: str = "./30M_2",
    tokenizer_path: str = "./tokenizer.model",
    max_seq_len: int = 1024,
    per_device_train_batch_size: int = 8,
    num_train_epochs: int = 1,
    warmup_steps: int = 5000,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    sdc_loss_term: bool = False,
    sdc_weight: float = 0.5,
    lcm_loss_term: bool = False,
    lcm_weight: float = 1.0,
    min_confidence: float = 0.7,
    confidence_soft: bool = False,
    mtp_weight: float = 0.5,
    output_dir: str = "/home/at0842/ycl466704.ai13/lung_mtp/sandbox/30M_2",
    from_scrach: bool = True,
    start_from_epoch: int = 0,
):
    model, tokenizer, model_args = load_model(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=per_device_train_batch_size,
        from_scrach=from_scrach,
    )
    model_args['epochs'] = num_train_epochs
    model_args['mtp_weight'] = mtp_weight
    model_args['sdc_weight'] = sdc_weight
    model_args['min_confidence'] = min_confidence
    wandb_name = output_dir.split('/')[-1]
    device, run = init(seed, model_args, wandb_name)
    
    K = model.n_future_tokens
    print(f'[INFO]: Model future tokens: {K}')
    

    model.to(device)
    model.train()

    train_ds = preprocess_datasets('train')
    test_ds, valid_ds = preprocess_datasets('test')
    print('[INFO]: Loaded datasets')

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        valid_ds,
        batch_size=per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    num_update_steps_per_epoch = len(train_loader)
    total_training_steps = num_train_epochs * num_update_steps_per_epoch

    if warmup_steps < 1:
        warmup_steps = warmup_steps * total_training_steps
        
    if from_scrach == False:
        warmup_steps = 0
        global_step = start_from_epoch * num_update_steps_per_epoch
    else:
        global_step = 0
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    print('[INFO]: Start traning')
    
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(num_train_epochs):
        for batch in train_loader:
            global_step += 1
            global_warmup_rate = min(1.0, float(global_step) / float(max(1, warmup_steps))) 

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            _, S = labels.shape
            
            loss_kl = 0.0
            loss_lcm = 0.0
            total_loss = 0.0
            
            autocast_enabled = (device.type == "cuda")
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                h_trunk, freqs_cis, mask = model.forward_trunk(input_ids)

            d = h_trunk.detach()
            d.requires_grad_(True)

            teacher_logits = None
            teacher_embedding = None
            any_head = False
            
            for i in range(K):
                T = S - (i + 1)
                if T <= 0:
                    break
                
                any_head = True
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    logits_ce, embedding = model.forward_head_from_trunk(
                        h_trunk=d,
                        head_index=i,
                        freqs_cis=freqs_cis,
                        mask=mask,
                    )
                
                li = logits_ce[:, :T, :]
                yi = labels[:, i + 1 :]
                
                ce_val = ce(li.float().reshape(-1, li.size(-1)), yi.reshape(-1))
                loss_i = ce_val.item()
                
                if i != 0:
                    ce_val = ce_val * ((mtp_weight**i) * global_warmup_rate)
                
                total_loss += ce_val.item()
                
                if sdc_loss_term and teacher_logits is not None:
                    teacher_slice = teacher_logits[:, i:, :].detach().float()
                    student_slice = logits_ce[:, :-i, :].float()

                    kl, mean_confidence = calc_sdc_loss(student_slice, teacher_slice, min_confidence, confidence_soft)

                    teacher_mean_confidence = mean_confidence.item()
                    loss_kl += kl.item()

                    if lcm_loss_term:
                        teacher = teacher_embedding[:, i:, :].detach().float()
                        student = embedding[:, :-i, :].float()

                        lcm = (student- teacher).pow(2).mean(dim=-1)
                        lcm = lcm.mean()

                        loss_lcm += lcm.item()

                    if i == 1:
                        run.log({
                            f"teacher mean_confidence": teacher_mean_confidence,
                        }, step = global_step)

                    head_loss = ce_val + kl * sdc_weight * global_warmup_rate + lcm * lcm_weight * global_warmup_rate
                    head_loss.backward()
                else:
                    ce_val.backward()
                
                if i == 0:
                    teacher_logits = logits_ce.detach()
                    teacher_embedding = embedding.detach()
                
                run.log({
                    f"head {i + 1} loss": loss_i,
                }, step = global_step)          
            
            total_loss = total_loss + loss_kl + loss_lcm
            
            if any_head:
                grad_to_trunk = d.grad
                if grad_to_trunk is None:
                    raise RuntimeError("d.grad is None; check that heads depend on d and backward() was called.")
                h_trunk.backward(gradient=grad_to_trunk.to(dtype=h_trunk.dtype))
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            clip_scale = min(1.0, max_grad_norm / (grad_norm + 1e-6))

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            run.log({
                "step": global_step,
                "total loss": total_loss,
                "sdc loss": loss_kl,
                "lcm loss": loss_lcm,
                "grad": grad_norm,
                "clip": clip_scale,
                "lr": current_lr,
            }, step = global_step)

            if global_step > 1200000:
                torch.save(model.state_dict(), output_dir + f"/model_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.pth")
                print('[INFO]: Model saved')
                break
                
    print('[INFO]: Complete traning')
    run.finish()

if __name__ == "__main__":
    import fire

    fire.Fire(main)
    