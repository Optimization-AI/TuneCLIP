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
pretrained_path='' # Not needed as we resume from the checkpoint after OSR
project_name_wandb=''
#normal lr to use is 1e-5
srun python -u src/training/main.py \
    --save-frequency 1 \
    --hinge_sogclr \
    --margin 0.1 --margin_schedule constant --margin_smoothing 2 \
    --train-data '/path/to/data.tar' \
    --resume /path_to/checkpoints/epoch_5.pt \
    --train-num-samples 12025856 --data_size 19280000 \
    --warmup 500 \
    --batch-size 256 \
    --epochs 10 \
    --workers 6 \
    --model ViT-B-16 \
    --name TuneCLIP_final_vitb16_openai \
    --seed 2025 \
    --wd 0.02 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_constant \
    --lr 1e-5 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 \
    --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 4 \
    --report-to wandb \
    --wandb-project-name ${project_name_wandb}

    ## INSTRUCTIONS TO RUN THE SCRIPT

    ## This is the script to run the second part of the TuneCLIP - OSR+HGCL -> Combine.
    ## set resume to the checkpoint from the first part of the TuneCLIP - OSR only. It resumes from the weights and the optimizer state.
    ## set lr to 1e-5 and hyperparameters as specified in the paper.
    ## use the flag --hinge_sogclr to use the HGCL loss.
    ## use --margin 0.1 to set the margin for the HGCL loss.
    ## use --margin_smoothing 2 to set the margin smoothing for the HGCL loss.
    ## use epochs 10 to set the number of epochs as the checkpoint starts from epoch number 5, so 5+5 = 10 total.