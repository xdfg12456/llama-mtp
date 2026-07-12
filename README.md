# 9Lung MTP

This repository trains and evaluates a Llama-style transformer with multi-token prediction (MTP) and auxiliary loss terms for confidence-aware training. The code is organized around the app directory and provides both training and evaluation entry points.

## Overview

The project includes:

- a Llama-style transformer implementation in app/models/training_model.py
- a training loop in app/trainer/pre_trainer.py
- example training and evaluation shell scripts in app/train.sh and app/eval.sh
- dataset loaders under app/training_datasets/pretrain/
- evaluation utilities under app/eval/

The training code is designed for GPU environments and uses torchrun for distributed execution.

## Project Structure

- app/models/training_model.py: model architecture, transformer blocks, rotary embeddings, and MTP heads
- app/trainer/pre_trainer.py: training loop, optimizer, scheduler, loss computation, WandB logging, and checkpoint saving
- app/train.sh: example command for launching training
- app/eval.sh: example command for launching evaluation
- app/eval/: scripts for perplexity, token acceptance, and generation-based evaluation
- app/training_datasets/pretrain/: dataset preprocessing for pretraining corpora
- app/tokenizer.model: tokenizer file used by the model

## Requirements

This project is designed for Python 3.11 and PyTorch 2.5 with CUDA 11.8.

### Conda environment

```bash
conda env create -f environment.yml
conda activate lung-llama-mtp
pip install -r requirements.txt
```

## Data and Tokenizer

Before training, make sure:

- the tokenizer file exists at app/tokenizer.model
- the required pretraining datasets are available and correctly referenced by the dataset loaders

The dataset code under app/training_datasets/pretrain/ supports multiple corpora such as OpenWebText, Wikipedia-style text, arXiv, PubMed, and others.

### Download and preprocess `segyges/OpenWebText2`

The [`segyges/OpenWebText2`](https://huggingface.co/datasets/segyges/OpenWebText2) repository provides the corpus as a large `openwebtext2.jsonl.zst.tar` archive. The `zstd2arrow.py` script reads the extracted `.jsonl.zst` files, tokenizes the `text` field, packs the tokens into fixed-length sequences, and saves sharded Hugging Face Arrow datasets.

> Make sure there is enough free disk space for the downloaded archive, the extracted `.jsonl.zst` files, and the generated Arrow shards.

#### 1. Install the download and preprocessing dependencies

```bash
conda activate lung-llama-mtp
pip install -U hf_xet zstandard
```

#### 2. Download the raw OpenWebText2 archive

Use a fixed local directory instead of relying on a Hugging Face cache hash, because the generated cache path may differ between machines.

```bash
mkdir -p /workspace/openwebtext2_raw

hf download segyges/OpenWebText2 openwebtext2.jsonl.zst.tar \
  --repo-type dataset \
  --local-dir /workspace/openwebtext2_raw
```

The downloaded archive should be located at:

```text
/workspace/openwebtext2_raw/openwebtext2.jsonl.zst.tar
```

#### 3. Extract the `.jsonl.zst` files

```bash
mkdir -p /workspace/openwebtext2_raw/extracted

tar -xf /workspace/openwebtext2_raw/openwebtext2.jsonl.zst.tar \
  -C /workspace/openwebtext2_raw/extracted
```

Check where the extracted files are located:

```bash
find /workspace/openwebtext2_raw/extracted \
  -type f -name "*.jsonl.zst" | head
```

All `.jsonl.zst` files must be directly inside the directory assigned to `zstd_download_path`, because `zstd2arrow.py` currently searches with `*.jsonl.zst` rather than recursively searching subdirectories.

#### 4. Configure `zstd2arrow.py`

Update the following paths to match your environment:

```python
# Directory that directly contains the extracted .jsonl.zst files
zstd_download_path = "/workspace/openwebtext2_raw/extracted"

tokenizer = Tokenizer(
    model_path="/workspace/app/tokenizer.model"
)
```

Configure the conversion parameters in the `__main__` block:

```python
if __name__ == "__main__":
    save_sharded_arrow_datasets(
        max_seq_length=1024,
        num_test_files=5,
        shard_file_count=5,
        output_dir="/workspace/hf_cache/segyges___open_web_text2/default/0.0.0",
        shuffle=True,
        seed=42,
    )
```

Parameter behavior:

- `max_seq_length=1024`: concatenates tokenized documents and creates fixed-length 1,024-token samples
- `num_test_files=5`: reserves five raw `.jsonl.zst` files for the test split
- `shard_file_count=5`: processes five raw files into each saved Arrow shard
- `shuffle=True` and `seed=42`: reproducibly shuffle the raw file list before splitting
- incomplete token buffers shorter than `max_seq_length` at the end of each shard are discarded

#### 5. Run the conversion

Run the script from an environment where `/workspace/app` is available on `PYTHONPATH`, because it imports `utils.tokenizer`.

```bash
cd /workspace/app
PYTHONPATH=/workspace/app python /path/to/zstd2arrow.py
```

For example, if the script is stored under `app/training_datasets/pretrain/`:

```bash
cd /workspace/app
PYTHONPATH=/workspace/app \
  python training_datasets/pretrain/zstd2arrow.py
```

#### 6. Generated dataset structure

With the default output path, the generated dataset is organized as follows:

```text
/workspace/hf_cache/segyges___open_web_text2/default/0.0.0/
├── train_shards/
│   ├── shard_00000/
│   ├── shard_00001/
│   └── ...
└── test_shards/
    ├── shard_00000/
    └── ...
```

Each saved sample contains one field:

```text
input_ids: Sequence[int32]
```

Existing shard directories are skipped automatically, so an interrupted conversion can be restarted without rebuilding completed shards.

## Training

A sample training command is provided in app/train.sh.

```bash
bash train.sh
```
Please executing above command on ../../app path

You can customize the run by editing the script or by passing arguments directly to the trainer module. Important options include:

- --ckpt_dir: directory containing model config and checkpoints
- --output_dir: directory for saved checkpoints and logs
- --mtp_weight: weight of the multi-token prediction loss
- --sdc_loss_term: enable self-distillation confidence loss
- --lcm_loss_term: enable latent consistency loss
- --min_confidence: confidence threshold for the auxiliary loss
- --sdc_weight: weight for the SDC loss term
- --lcm_weight: weight for the LCM loss term

The trainer also logs metrics to Weights & Biases by default.

## Evaluation

A sample evaluation command is provided in app/eval.sh.

```bash
bash eval.sh
```
Please executing above command on ../../app path

The evaluation pipeline includes:

- perplexity evaluation
- average token acceptance evaluation
- average token acceptance by prefix length
- generation-based analysis and repetition metrics

Results are written into the log directory.

## Checkpoints and Logs

- checkpoints/: default location for saved model weigh. click [here](https://huggingface.co/xdfg12456/llama-mtp/tree/main/checkpoints) to download checkpoints
- log/: evaluation and analysis output files
- hf_cache/: default location for saved datasets

## Notes

- The repository assumes a CUDA-capable GPU and uses NVIDIA NCCL for distributed training.
- The default scripts reference a conda environment named lung-llama-mtp.
- If you are adapting this project to a new machine, update the checkpoint paths, output directories, and training arguments in the shell scripts first.
