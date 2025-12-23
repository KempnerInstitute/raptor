#!/bin/bash
#SBATCH --job-name=raptor_array_distill
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --time=3-00:00:00
#SBATCH --mem=375G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-2
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt

module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate slot_attention6

BP1_1="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_0_end_4_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_102_step_312500.pt"
BP1_2="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_0_end_4_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_108_step_312500.pt"
BP1_3="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_0_end_4_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_114_step_312500.pt"

BP2_1="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_4_end_7_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_103_step_312500.pt"
BP2_2="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_4_end_7_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_109_step_312500.pt"
BP2_3="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_4_end_7_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_115_step_312500.pt"

BP3_1="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_7_end_10_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_101_step_312500.pt"
BP3_2="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_7_end_10_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_107_step_312500.pt"
BP3_3="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_7_end_10_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_113_step_312500.pt"

BP4_1="final_weighted_True_autoregressive_False_distillation_False_teacher_True_mse_False_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_10_end_12_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_104_step_312500.pt"
BP4_2="final_weighted_True_autoregressive_False_distillation_False_teacher_True_mse_False_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_10_end_12_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_110_step_312500.pt"
BP4_3="final_weighted_True_autoregressive_False_distillation_False_teacher_True_mse_False_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_10_end_12_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_116_step_312500.pt"

PARAMS=(
  "--raptor4 --autoreg --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 12 --cls_weight 0.45 --reg_weight 0.10 --patch_weight 0.45 --bp1 $BP1_1 --bp2 $BP2_1 --bp3 $BP3_1 --bp4 $BP4_1 --seed 1201"
  "--raptor4 --autoreg --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 12 --cls_weight 0.45 --reg_weight 0.10 --patch_weight 0.45 --bp1 $BP1_2 --bp2 $BP2_2 --bp3 $BP3_2 --bp4 $BP4_2 --seed 1202"
  "--raptor4 --autoreg --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 12 --cls_weight 0.45 --reg_weight 0.10 --patch_weight 0.45 --bp1 $BP1_3 --bp2 $BP2_3 --bp3 $BP3_3 --bp4 $BP4_3 --seed 1203"
)

ARGS=${PARAMS[$SLURM_ARRAY_TASK_ID]}

python ../trainer.py $ARGS