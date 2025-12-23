#!/bin/bash
#SBATCH --job-name=raptor_nyud_probe
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
  "--variant dino_s --seed 6000"
  "--variant dino_b --seed 6001"
  "--variant raptor2 --model_seed 1001 --seed 6002"
  "--variant raptor2 --model_seed 1002 --seed 6003"
  "--variant raptor2 --model_seed 1003 --seed 6004"
  "--variant raptor3 --model_seed 1101 --seed 6005"
  "--variant raptor3 --model_seed 1102 --seed 6006"
  "--variant raptor3 --model_seed 1103 --seed 6007"
  "--variant raptor4 --model_seed 1201 --seed 6008"
  "--variant raptor4 --model_seed 1202 --seed 6009"
  "--variant raptor4 --model_seed 1203 --seed 6010"
)

ARGS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

python train_probe.py $ARGS
