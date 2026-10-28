#!/bin/bash

##NECESSARY JOB SPECIFICATIONS  
#SBATCH commands


source ~/.bashrc
conda activate env

cd /path/to/your/project

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'
pretrained_path='openai'
project_name_wandb=''

srun python -u src/training/main.py \
    --save-frequency 1 \
    --train-data '/dfn/../path/to/dfn.tar' \
    --pretrained ${pretrained_path} \
    --train-num-samples 12025856 --data_size 19280000 \
    --warmup 500 \
    --batch-size 256 \
    --epochs 5 \
    --workers 6 \
    --model ViT-B-16 \
    --name TuneCLIP_OSR_openai_vitb16 \
    --seed 2025 \
    --wd 0.02 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_constant \
    --lr 0.0 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 \
    --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 4 \
    --report-to wandb \
    --wandb-project-name ${project_name_wandb}


    ## INSTRUCTIONS TO RUN THE SCRIPT

    ## This is the script to run the first part of the TuneCLIP - OSR.

    ## Sample shown below for ViT-B-16 model:

    ## set pretrained to 'openai' for ViT-B/X OPENAI CLIPs (Refer to /TuneCLIP/src/fast_clip/pretrained.py for details)
    ## set lr = 0.0 as we freeze the model
    ## set paths of data and hyperparameters as per need
    ## Run for 5 epochs to get the model weights to be used in second phase
  
