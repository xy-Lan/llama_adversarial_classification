#!/bin/bash
#SBATCH --job-name=llama_task
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.8.0

python3 classify_adversarial.py