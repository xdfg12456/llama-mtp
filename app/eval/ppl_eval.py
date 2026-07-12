from dataclasses import dataclass
from typing import List

import torch

@dataclass
class PerHeadOutput:
    total_nll: int = 0
    total_tokens: int = 0

@torch.no_grad()
def evaluate_total_loss(model, dataloader, device, max_batches=500):
    model.eval()
    
    result = [PerHeadOutput() for _ in range(model.params.n_future_tokens)]
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        _, S = labels.shape
        
        for i in range(model.params.n_future_tokens):
            T = S - (i + 1)
            if T <= 0:
                break
            
            logits = model.forward(input_ids, head_index=i)

            shift_logits = logits[:, :T, :].contiguous()
            shift_labels = input_ids[:, i + 1 :].contiguous()

            ce = torch.nn.CrossEntropyLoss(reduction="none")

            vocab_size = shift_logits.size(-1)
            token_loss = ce(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
            )

            token_loss = token_loss.view(shift_labels.size())

            result[i].total_nll += token_loss.sum().item()
            result[i].total_tokens += shift_labels.numel()
        
    return result
            
def calc_ppl(per_head_output: List[PerHeadOutput], n_future_token: int):
    result = []
    
    for i in range(n_future_token):
        avg_nll = per_head_output[i].total_nll / per_head_output[i].total_tokens
        ppl = torch.exp(torch.tensor(avg_nll))
        
        result.append(ppl)
    
    return result