#!/bin/bash

#SBATCH --job-name=mamba_baseline
#SBATCH --output=saida_mamba_baseline.log
#SBATCH --error=erro_mamba_baseline.log

#SBATCH --time=01:00:00
#SBATCH --partition=h100n3

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

python3 evaluate_ragas.py
