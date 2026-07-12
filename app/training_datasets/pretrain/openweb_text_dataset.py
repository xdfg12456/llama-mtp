import os
from datasets import load_from_disk, concatenate_datasets
import torch

test_ds_path = "/workspace/hf_cache/segyges___open_web_text2/default/0.0.0/test_shards"
train_ds_path = "/workspace/hf_cache/segyges___open_web_text2/default/0.0.0/train_shards"

def load_sharded_dataset(shard_root):
    shard_dirs = sorted(
        os.path.join(shard_root, d)
        for d in os.listdir(shard_root)
        if d.startswith("shard_")
    )
    datasets_list = [load_from_disk(p) for p in shard_dirs]
    return concatenate_datasets(datasets_list)

def preprocess_datasets(split = 'train'):
    if split == 'train':
        return load_sharded_dataset(train_ds_path)
    elif split == 'test':
        full_ds = load_sharded_dataset(test_ds_path)
        n = len(full_ds)
        valid_size = min(10000, n)
        valid_ds = full_ds.select(range(valid_size))
        test_ds = full_ds.select(range(valid_size, n))
        return test_ds, valid_ds

def collate_fn(batch):
    input_ids = torch.tensor(
        [example["input_ids"] for example in batch],
        dtype=torch.long
    )
    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
