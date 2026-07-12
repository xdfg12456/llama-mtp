from ppl_eval import evaluate_total_loss, calc_ppl
from models.training_model import Transformer, ModelArgs
from utils.tokenizer import Tokenizer
import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.print_to_file import print_to_file, create_output_file_path
from training_datasets.pretrain import openweb_text_dataset, wikitext_dataset, ptb_dataset, lambada_dataset, arxiv_dataset, pubmed_dataset

def init(seed: int = 1):
    torch.manual_seed(seed)
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'[INFO] device: {device}')

    return device

def load_model(
    ckpt_dir: str,
    ckpt_dir_suffix:str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
):
    start_time = time.time()
    
    checkpoints = sorted(Path(ckpt_dir + ckpt_dir_suffix).glob("*.pth"))
    ckpt_path = checkpoints[-1]
    print(f"[INFO]: model choosed {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
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
    model.load_state_dict(checkpoint, strict=False)

    print(f"[INFO]: Loaded in {time.time() - start_time:.2f} seconds")

    return model, tokenizer

def ppl_test(model, dataloader, device, key_name, output_file_path):
    n_future_token = model.params.n_future_tokens
    
    outputs = evaluate_total_loss(model, dataloader, device)
    ppls = calc_ppl(outputs, n_future_token)

    title = f"\n===== {key_name} Zero-Shot PPL Summary ====="
    message = title
    for i in range(n_future_token):
        result = f"\n{i+1}: {ppls[i]}"
        message = message + result

    print_to_file(message, output_file_path)

def generlize_ppl_test(model, device, model_name):
    test_ds, valid_ds = openweb_text_dataset.preprocess_datasets('test')
    test_loader = DataLoader(
        valid_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=openweb_text_dataset.collate_fn,
        num_workers=0,
    )
    ppl_test(model, test_loader, device, f'{model_name} owt')

    valid_ds = wikitext_dataset.preprocess_datasets('train', 1024)
    test_loader = DataLoader(
        valid_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=wikitext_dataset.collate_fn,
        num_workers=0,
    )
    ppl_test(model, test_loader, device, f'{model_name} wikitext')

    valid_ds = ptb_dataset.preprocess_datasets('train', 1024)
    test_loader = DataLoader(
        valid_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=ptb_dataset.collate_fn,
        num_workers=0,
    )
    ppl_test(model, test_loader, device, f'{model_name} ptb')

    valid_ds = lambada_dataset.preprocess_datasets('train', 1024)
    test_loader = DataLoader(
        valid_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=lambada_dataset.collate_fn,
        num_workers=0,
    )
    ppl_test(model, test_loader, device, f'{model_name} lambada')

    valid_ds = arxiv_dataset.preprocess_datasets('validation', 1024)
    test_loader = DataLoader(
        valid_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=arxiv_dataset.collate_fn,
        num_workers=0,
    )
    ppl_test(model, test_loader, device, f'{model_name} arxiv')

    valid_ds = pubmed_dataset.preprocess_datasets('validation', 1024)
    test_loader = DataLoader(
        valid_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=pubmed_dataset.collate_fn,
        num_workers=0,
    )
    ppl_test(model, test_loader, device, f'{model_name} pubmed')

def main(
    seed: int = 1,
    ckpt_dir: str = '30M_2_sdc',
    ckpt_dir_suffix: str = '',
    tokenizer_path: str = "./tokenizer.model",
    max_seq_len: int = 1024,
    per_device_train_batch_size: int = 8,
    output_dir: str = "./log",
):
    device = init(seed=seed)
    model, _ = load_model(ckpt_dir, ckpt_dir_suffix, tokenizer_path, max_seq_len, per_device_train_batch_size)
    model_name = ckpt_dir_suffix
    output_file_path = create_output_file_path(output_dir, 'zero_shot_ppl')
    
    model.to(device)
    model.eval()
    
    generlize_ppl_test(model, device, model_name, output_file_path)
    
if __name__ == "__main__":
    import fire

    fire.Fire(main)