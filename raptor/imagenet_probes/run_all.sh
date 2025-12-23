#!/bin/bash
#SBATCH --job-name=raptor_imagenet_probe
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=3-00:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-10
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate slot_attention6


PARAMS=(
  "--variant dino_s --seed 4000"
  "--variant dino_b --seed 4001"
  "--variant raptor2 --model_seed 1001 --seed 4002"
  "--variant raptor2 --model_seed 1002 --seed 4003"
  "--variant raptor2 --model_seed 1003 --seed 4004"
  "--variant raptor3 --model_seed 1101 --seed 4005"
  "--variant raptor3 --model_seed 1102 --seed 4006"
  "--variant raptor3 --model_seed 1103 --seed 4007"
  "--variant raptor4 --model_seed 1201 --seed 4008"
  "--variant raptor4 --model_seed 1202 --seed 4009"
  "--variant raptor4 --model_seed 1203 --seed 4010"
)

ARGS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

python train_probe.py $ARGS
