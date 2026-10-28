#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=datacomp-eval
#SBATCH --time=3:00:00                     
#SBATCH --ntasks=1                         
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G                             
#SBATCH --output=/path/%x_%j.log  
#SBATCH --gres=gpu:1
#SBATCH --partition=xxx
#SBATCH --account=xxx            

cd /path/to/your/project

python evaluate.py \
 --train_output_dir /path/to/your/project/train_output_dir \
 --data_dir /path/to/your/project/datasets/datacomp \
 --model ViT-B-32 \
 --task datacomp \
 --use_model "ViT-B-32 /checkpoints/sample.pt" \