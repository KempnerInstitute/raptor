#!/bin/bash
#SBATCH --job-name=raptor_extract_imagenet
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=3-00:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-0
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate slot_attention6

python 000_precompute_dinov2_act.py