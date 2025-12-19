#!/bin/bash
#SBATCH --job-name=raptor_array_distill
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=3-00:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-17
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate slot_attention6


PARAMS=(
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 7 --seed 100"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 10  --seed 101"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 4 --seed 102"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 4 --end_layer 7 --seed 103"
  "--teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 10 --end_layer 12 --seed 104"
  "--teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 12 --seed 105"

  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 7 --seed 106"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 10  --seed 107"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 4 --seed 108"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 4 --end_layer 7 --seed 109"
  "--teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 10 --end_layer 12 --seed 110"
  "--teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 12 --seed 111"

  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 7 --seed 112"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 10  --seed 113"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 4 --seed 114"
  "--teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 4 --end_layer 7 --seed 115"
  "--teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 10 --end_layer 12 --seed 116"
  "--teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 12 --seed 117"
)

ARGS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

python ../trainer.py $ARGS