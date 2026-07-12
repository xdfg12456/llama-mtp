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
    --nproc_per_node 1 \
    -m trainer.pre_trainer \
    --ckpt_dir /workspace/checkpoints/1.3B_4_sdc \
    --output_dir /workspace/checkpoints/1.3B_4_sdc \
    --mtp_weight 1 \
    --from_scrach True \
    --sdc_loss_term True \
    --lcm_loss_term True \
    --min_confidence 0.3 \
    --sdc_weight 1.0 \
    --lcm_weight 0.5 \
    --warmup_steps 15000
