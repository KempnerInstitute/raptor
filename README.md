# Block-Recurrent Dynamics in ViTs

<div align="center">
<img src="docs/raptor_logo.png" width="25%" />
</div>

**Authors:** Mozes Jacobs\*, Thomas Fel\*, Richard Hakim\*, Alessandra Brondetta, Demba Ba, T. Andy Keller

\* Equal contribution. Correspondence to {mozesjacobs,tfel,rhakim,takeller}@g.harvard.edu

---

**Abstract:** 
As Vision Transformers (ViTs) become standard backbones across vision, a mechanistic account of their computational phenomenology is now essential.
Despite architectural cues that hint at dynamical structure, there is no settled framework that interprets Transformer depth as a well-characterized flow.
In this work, we introduce the **Block-Recurrent Hypothesis (BRH)**, arguing that trained ViTs admit a block-recurrent depth structure such that the computation of the original L blocks can be accurately rewritten using only k << L distinct blocks applied recurrently.
Across diverse ViTs, between-layer representational similarity matrices suggest few contiguous phases. Yet, representational similarity does not necessarily translate to functional similarity.
To determine whether these phases reflect genuinely reusable computation, we operationalize our hypothesis in the form of block recurrent surrogates of pretrained ViTs, which we call **R**ecurrent **A**pproximations to **P**hase-structured **T**ransf**OR**mers (`Raptor`). 
Using small-scale ViTs, we demonstrate that phase-structure metrics correlate with our ability to accurately fit `Raptor`, and identify the role of training and stochastic depth in promoting the recurrent block structure. 
We then provide an empirical existence proof for BRH in foundation models by showing that we can train a `Raptor` model to recover $96\%$ of DINOv2 ImageNet-1k linear probe accuracy in only 2 blocks while maintaining equivalent computational cost. 
To provide a mechanistic account of these observations, we leverage our hypothesis to develop a program of **Dynamical Interpretability**. We find **(i)** directional convergence into class-dependent angular basins with self-correcting trajectories under small perturbations, **(ii)** token-specific dynamics, where `cls` executes sharp late reorientations while `patch` tokens exhibit strong late-stage coherence reminiscent of a mean-field effect and converge rapidly toward their mean direction, and **(iii)** a collapse of the update to low rank in late depth, consistent with convergence to low-dimensional attractors.<br>

Altogether, we find that a compact recurrent program emerges along the depth of ViTs, pointing to a low-complexity normative solution that enables these models to be studied through principled dynamical systems analysis.

---

## Setup

### Environment
To run the code, you will need to create a mamba (or conda) environment from the `environment.yml` file.
Create and activate the environment with 
```bash
mamba env create -f environment.yml
mamba activate raptor
```

### Paths
Edit src/paths.py to have the correct absolute paths to different datasets.

### Extracting DINOv2 Activations for ImageNet-1k
For ImageNet, we precompute the DINOv2 activations so that `Raptor` can train faster. 
We provide a script to extract the activations from the ImageNet-1k dataset. This script is available in the `data` directory.
This script takes around 5 hours to run on 1 H100 GPU, and storing the activations requires a lot of disk space.
```bash
cd data
python 000_precompute_dinov2_act.py
```

### Download Pretrained Classifiers
Download the DINOv2 linear heads from Meta's [repository](https://github.com/facebookresearch/dinov2). 
These are used during training of `Raptor`.

```bash
cd src
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth
cp dinov2_vitb14_reg4_linear_head.pth imagenet_probes/dinov2_vitb14_reg4_linear_head.pth
cp dinov2_vits14_reg4_linear_head.pth imagenet_probes/dinov2_vits14_reg4_linear_head.pth
```

## Usage Example
`Raptor` training follows 4 main steps. Here, we show example usage for a 3-block `Raptor`:

1. Determine max-cut segmentations. This has been done for you in src/000_max_cut_dinov2_base.ipynb.
2. Train each block independently.
```bash
cd src
python trainer.py --teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 7 --seed 100
python trainer.py --teacher_force --mse --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 7 --end_layer 10  --seed 101
python trainer.py --teacher_force --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 10 --end_layer 12 --seed 104
```
3. Train the full model with the pretrained blocks.
```bash
cd src
BP1="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_0_end_7_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_100_step_312500.pt"
BP2="final_weighted_False_autoregressive_False_distillation_False_teacher_True_mse_True_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_7_end_10_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_101_step_312500.pt"
BP3="final_weighted_True_autoregressive_False_distillation_False_teacher_True_mse_False_cosine_False_t_scale_True_swiglu_True_sigma_0.0_start_10_end_12_lr_0.0003_cls_weight_0.34_reg_weight_0.33_patch_weight_0.33_seed_104_step_312500.pt"
python trainer.py --raptor3 --autoreg --weighted --sigma 0 --lr 3e-4 --wandb --t_scale --swiglu --start_layer 0 --end_layer 12 --cls_weight 0.45 --reg_weight 0.10 --patch_weight 0.45 --bp1 $BP1 --bp2 $BP2 --bp3 $BP3 --seed 1101
```
4. Train linear probes on the frozen pretrained checkpoints.
```bash
cd src/imagenet_probes
python train_probe.py --variant raptor3 --model_seed 1101 --seed 4005
```
```bash
cd src/ade20k_probes
python train_probe.py --variant raptor3 --model_seed 1101 --seed 5005
```
```bash
cd src/nyud_probes
python train_probe.py --variant raptor3 --model_seed 1101 --seed 6005
```

## Reproducing Foundation Models Results (Section 3)
To reproduce the results for the foundation models section (Table 1 and Figure 7), do the following:

1. Determine max-cut segmentations. This has been done for you in src/000_max_cut_dinov2_base.ipynb.
2. Train each block independently.
```bash
cd src/runs
sbatch 001_blocks.sh
```
3. Train the full model with the pretrained blocks.
```bash
cd src/runs
sbatch 002_raptor2_pretrained.sh
sbatch 003_raptor3_pretrained.sh
sbatch 004_raptor4_pretrained.sh
```
4. Train linear probes on the frozen pretrained checkpoints.
```bash
cd src/ade20k_probes
sbatch run_all.sh
```
```bash
cd src/imagenet_probes
sbatch run_all.sh
```
```bash
cd src/nyud_probes
sbatch run_all.sh
```
5. Table 1
```bash
cd src
python aggregate_results.py
```
6. Figure 7
Run the notebook in src/imagenet_probes/101_eval_error_bars.ipynb.