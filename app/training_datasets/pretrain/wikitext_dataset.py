from datasets import load_dataset
from utils.tokenizer import Tokenizer
import numpy as np
import torch

DATASETNAME = "wikitext"
CACHE_DIR = "/workspace/hf_cache"
TEXT_COLUMN = 'text'
NUM_PROC = 8

EOS = '</s>'

tokenizer = Tokenizer(
    model_path="/workspace/app/tokenizer.model"
)

def tokenize_function(batch):
    tokenized = [tokenizer.encode(t, False, True) for t in batch[TEXT_COLUMN]]
    
    return {'input_ids': tokenized}

def group_texts(row, max_seq_length):
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

def preprocess_datasets(split='train', max_seq_length = 4096):
    ds = load_dataset(DATASETNAME, name='wikitext-103-raw-v1', cache_dir=CACHE_DIR)
    ds = ds[split]
        
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
