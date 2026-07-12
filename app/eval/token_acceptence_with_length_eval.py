import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def eval_accept_rate_for_length(
    model,
    dataloader: DataLoader,
    prefix_len: int,
    device: torch.device,
    max_batches: int = 500,
    random_slice: bool = True,
):
    model.eval()

    n_future_token = model.params.n_future_tokens
    
    total_drafts = 0  # 提出的 draft tokens 數量
    total_accepts = 0  # 被接受的 draft tokens 數量

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        input_ids: torch.Tensor = batch["input_ids"].to(device)  # (B, S)
        B, S = input_ids.shape

        # 要留至少 L + 2 個 token（prefix + token1 + token2）
        if S < prefix_len + n_future_token:
            continue

        max_start = S - (prefix_len + n_future_token)

        if random_slice:
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = 0

        end = start + prefix_len

        prefix_batch = input_ids[:, start:end]

        # === 1) trunk on prefix ===
        h_trunk, freqs_cis, mask = model.forward_trunk(prefix_batch)  # (B, L, dim)
        
        predict_tokens = []
        candidate = prefix_batch
        
        for i in range(n_future_token):
            logits_main, h = model.forward_head_from_trunk(
                h_trunk=h_trunk,
                head_index=i,
                freqs_cis=freqs_cis,
                mask=mask,
            ) 
            
            token = torch.argmax(logits_main[:, -1, :], dim=-1)
            predict_tokens.append(token)
            
            candidate = torch.cat(
                [candidate, token.unsqueeze(-1)], dim=1
            )

        logits_verify = model(candidate, head_index=0)
        verify_tokens = torch.argmax(logits_verify[:, -n_future_token:-1, :], dim=-1)
        accepts = 0
        for i in range(B):
            for j in range(n_future_token - 1):          
                if verify_tokens[i, j] != predict_tokens[j + 1][i]:
                    break
                else:
                    accepts = accepts + 1

        # === 5) 統計 accept rate ===
        total_drafts += B
        total_accepts += accepts

    if total_drafts == 0:
        return 0.0

    return total_accepts / total_drafts
