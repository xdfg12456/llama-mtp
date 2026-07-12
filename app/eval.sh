#!/bin/bash

#==========================
# Anaconda Environment
#==========================

eval "$(conda shell.bash hook)"

conda deactivate
conda activate lung-llama-mtp

#==========================
# Execute My Program
#==========================

nvidia-smi

torchrun \
    --master_port 29501 \
    --nproc_per_node 1 \
    -m eval.run_all_eval \
    --ckpt_dir /workspace/checkpoints/1.3B_4 \
    --output_dir /workspace/log \
    --seed 1
