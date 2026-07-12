from datasets import load_dataset
from utils.tokenizer import Tokenizer
import numpy as np
import torch

DATASETNAME = "armanc/scientific_papers"
CONFIG_NAME = "pubmed"
CACHE_DIR = "/workspace/hf_cache"
TEXT_COLUMN = "text"
NUM_PROC = 8
TRUST_REMOTE_CODE = True

EOS = "</s>"

tokenizer = Tokenizer(
    model_path="/workspace/app/tokenizer.model"
)

def combine(example):
    return {
        "text":
            f"{example['input']}\n\n{example['output']}"
    }

def tokenize_function(batch):
    texts = [t if t is not None else "" for t in batch[TEXT_COLUMN]]
    tokenized = [tokenizer.encode(t, False, True) for t in texts]
    return {"input_ids": tokenized}


def group_texts(row, max_seq_length):
    ids_list = [ids for ids in row["input_ids"] if len(ids) > 0]
    if len(ids_list) == 0:
        return {"input_ids": [], "labels": []}

    concatenated_ids = list(np.concatenate(ids_list))
    total_length = len(concatenated_ids)
    total_length = (total_length // max_seq_length) * max_seq_length
    concatenated_ids = concatenated_ids[:total_length]

    input_ids = [
        concatenated_ids[i: i + max_seq_length]
        for i in range(0, total_length, max_seq_length)
    ]

    return {
        "input_ids": input_ids,
        "labels": input_ids.copy(),
    }


def preprocess_datasets(split="train", max_seq_length=4096):
    data_files = {
        "train": "hf://datasets/scillm/scientific_papers-archive/scientific_papers_pubmed_train.jsonl",
        "validation": "hf://datasets/scillm/scientific_papers-archive/scientific_papers_pubmed_val.jsonl",
        "test": "hf://datasets/scillm/scientific_papers-archive/scientific_papers_pubmed_test.jsonl",
    }

    ds = load_dataset("json", data_files=data_files)
    ds = ds[split]
    ds = ds.map(combine)

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
