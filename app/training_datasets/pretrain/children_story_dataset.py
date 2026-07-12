from datasets import load_dataset
from utils.tokenizer import Tokenizer
import numpy as np
import torch

DATASETNAME = "ajibawa-2023/Children-Stories-Collection"
TEXT_COLUMN = 'text'
CACHE_DIR = "/workspace/hf_cache"
NUM_PROC = 8

EOS = '</s>'

tokenizer = Tokenizer(
    model_path="/workspace/app/tokenizer.model"
)

def tokenize_function(batch):
    texts = [t if isinstance(t, str) else "" for t in batch[TEXT_COLUMN]]
    tokenized = [tokenizer.encode(t, False, True) for t in texts]
    
    return {'input_ids': tokenized}

def group_texts(row, max_seq_length):
    if len(row["input_ids"]) == 0:
        return {"input_ids": [], "labels": []}
    
    concatenated_ids = list(np.concatenate(row["input_ids"]))

    total_length = len(concatenated_ids)

    total_length = (total_length // max_seq_length) * max_seq_length
    concatenated_ids = concatenated_ids[:total_length]

    input_ids = [
        concatenated_ids[i : i + max_seq_length]
        for i in range(0, total_length, max_seq_length)
    ]

    result = {
        "input_ids": input_ids,
        "labels": input_ids.copy(),
    }

    return result

def get_train_test_raw(test_size: int = 0.01, seed: int = 42):
    raw = load_dataset(DATASETNAME, cache_dir=CACHE_DIR, split="train")
    split_ds = raw.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split_ds["train"], split_ds["test"]

def preprocess_datasets(ds, max_seq_length = 1024):
    tokenize_ds = ds.map(
        tokenize_function,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )
    
    lm_dataset = tokenize_ds.map(
        lambda row: group_texts(row, max_seq_length),
        batched=True,
        num_proc=NUM_PROC,
    )
    
    return lm_dataset
    
def collate_fn(batch):
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}
