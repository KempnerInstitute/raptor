from trainer import train

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train the model with specified configurations.")
    parser.add_argument('--autoreg', action='store_true', help="Enable autoregressive training")
    parser.add_argument('--distill', action='store_true', help="Enable distillation training")
    parser.add_argument('--teacher_force', action='store_true', help="Enable teacher-forced")
    parser.add_argument('--mse', action='store_true', help="Enable mse loss")
    parser.add_argument('--cosine', action='store_true', help="Enable cosine loss")
    parser.add_argument('--t_scale', action='store_true', help="Use layer dependent scaling.")
    parser.add_argument('--swiglu', action='store_true', help="Use Swiglu in MLP (if False, uses GELU)")
    parser.add_argument('--sigma', type=float, default=1e-3, help="Sigma value for noise")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--start_layer', type=int, default=0, help="Start layer for the model (takes in this layer)")
    parser.add_argument('--end_layer', type=int, default=7, help="End layer for the model (outputs this layer)")
    parser.add_argument('--wandb', action='store_true', help="Use wandb for logging (if False, only logs through prints)")
    parser.add_argument('--weighted', action='store_true', help="Use weight on cls, patch, and register")
    parser.add_argument('--cls_weight', type=float, default=0.34, help="Weight of cls token in loss. Applies only if --weighted.")
    parser.add_argument('--reg_weight', type=float, default=0.33, help="Weight of reg token in loss. Applies only if --weighted.")
    parser.add_argument('--patch_weight', type=float, default=0.33, help="Weight of patch token in loss. Applies only if --weighted.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization")
    parser.add_argument('--raptor2', action='store_true', help="Build Raptor2")
    parser.add_argument('--raptor3', action='store_true', help="Build Raptor3")
    parser.add_argument('--raptor4', action='store_true', help="Build Raptor4")
    parser.add_argument('--name_prefix', type=str, default="", help="Prefix for naming the model")
    parser.add_argument('--bp1', type=str, default="", help="Path to block 1")
    parser.add_argument('--bp2', type=str, default="", help="Path to block 2")
    parser.add_argument('--bp3', type=str, default="", help="Path to block 3")
    parser.add_argument('--bp4', type=str, default="", help="Path to block 4")
    parser.add_argument('--proj_name', type=str, default="raptor_foundation_model", help="Name of the project")
    return parser.parse_args()


def build_config(args,
                 init_values=1e-5, dim=768, num_heads=12, mlp_ratio=4.0, wd=1e-4, epochs=20, log_interval=100, save_interval=5000,
                 validate_interval=5000, num_val_samples=2048, batch_size=64, tf_anneal_ratio=0.25, tf_min_ratio=0.0, warmup_iters=10000):
    
    from paths import PRJ_DIR, IMAGENET_VAL_DIR, DATA_DIR
    DINO_MODEL = "dinov2_vitb14_reg"
    CLASSIFIER_PATH = f"{PRJ_DIR}/src/dinov2_vitb14_reg4_linear_head.pth"

    block_paths = [bp for bp in [args.bp1, args.bp2, args.bp3, args.bp4] if bp != ""]
    config = {
        'data_dir': DATA_DIR,
        'dino_model': DINO_MODEL,
        'val_dir': IMAGENET_VAL_DIR,
        'classifier_path': CLASSIFIER_PATH,
        'model_save_dir': 'models',
        'init_values': init_values,
        'sigma': args.sigma,
        'start_layer': args.start_layer,
        'end_layer': args.end_layer,
        'dim': dim,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'lr': args.lr,
        'wd': wd,
        'epochs': epochs,
        'device': 'cuda',
        'log_interval': log_interval, 
        'save_interval': save_interval,
        'validate_interval': validate_interval,
        'mse': args.mse,
        'cosine': args.cosine,
        'autoregressive_training': args.autoreg,
        'distillation_training': args.distill,
        't_scale': args.t_scale,
        'swiglu': args.swiglu,
        'num_val_samples': num_val_samples,
        'batch_size': batch_size,
        'wandb': args.wandb,
        'tf_anneal_ratio': tf_anneal_ratio,
        'tf_min_ratio': tf_min_ratio,
        'teacher_force': args.teacher_force,
        'warmup_iters': warmup_iters,
        'weighted': args.weighted,
        'cls_weight': args.cls_weight,
        'reg_weight': args.reg_weight,
        'patch_weight': args.patch_weight,
        'seed': args.seed,
        'raptor2': args.raptor2,
        'raptor3': args.raptor3,
        'raptor4': args.raptor4,
        'name_prefix': args.name_prefix,
        'block_paths': block_paths,
        'proj_name': args.proj_name,
    }
    return config