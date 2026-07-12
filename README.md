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
```

### Python dependencies

If you prefer to install packages manually:

```bash
pip install -r requirements.txt
```

## Data and Tokenizer

Before training, make sure:

- the tokenizer file exists at app/tokenizer.model
- the required pretraining datasets are available and correctly referenced by the dataset loaders

The dataset code under app/training_datasets/pretrain/ supports multiple corpora such as OpenWebText, Wikipedia-style text, arXiv, PubMed, and others.

## Training

A sample training command is provided in app/train.sh.

```bash
bash app/train.sh
```

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
bash app/eval.sh
```

The evaluation pipeline includes:

- perplexity evaluation
- average token acceptance evaluation
- average token acceptance by prefix length
- generation-based analysis and repetition metrics

Results are written into the log directory.

## Checkpoints and Logs

- checkpoints/: default location for saved model weights
- log/: evaluation and analysis output files

## Notes

- The repository assumes a CUDA-capable GPU and uses NVIDIA NCCL for distributed training.
- The default scripts reference a conda environment named lung-llama-mtp.
- If you are adapting this project to a new machine, update the checkpoint paths, output directories, and training arguments in the shell scripts first.
