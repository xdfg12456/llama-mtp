import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def avg_token_acceptance(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 1,
):
    model.eval()

    n_future_token = model.params.n_future_tokens
    max_draft_tokens = n_future_token - 1

    total_accepted = 0
    total_trials = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        B, S = input_ids.shape

        h_trunk, freqs_cis, mask = model.forward_trunk(input_ids)

        heads_predict_tokens = []

        for head_idx in range(n_future_token):
            logits, _ = model.forward_head_from_trunk(
                h_trunk=h_trunk,
                head_index=head_idx,
                freqs_cis=freqs_cis,
                mask=mask,
            )

            tokens = torch.argmax(logits, dim=-1)
            heads_predict_tokens.append(tokens)

        for b in range(B):
            main_head_predict = heads_predict_tokens[0][b]

            for j in range(S - max_draft_tokens):
                for k in range(max_draft_tokens):
                    target_pos = j + k + 1

                    head_predict = heads_predict_tokens[k + 1][b]

                    if head_predict[j] == main_head_predict[target_pos]:
                        total_accepted += 1
                    else:
                        break
                total_trials += 1

    if total_trials == 0:
        return 0.0

    return total_accepted / total_trials