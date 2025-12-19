import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
import sys
from datetime import datetime
from overcomplete.metrics import r2_score
from einops import rearrange
import wandb

from scheduler import CosineScheduler
from block import Block
from raptor import Raptor
from dataloader import AsynchZarrLoader, imagenet_transform
from dino_wrapper import DinoModelWrapper

import random

def set_random_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def loss_mse(pred, target):
    return torch.square(pred - target).mean()

def loss_cosine(pred, target):
    return F.cosine_similarity(pred, target, dim=2).mean()

def get_loss_func(mse, cosine, weighted, cls_weight, reg_weight, patch_weight):
    if weighted:
        def loss_func(pred, target):
            cls_loss = loss_mse(pred[:,0], target[:,0])     # before: 1/261
            reg_loss = loss_mse(pred[:,1:5], target[:,1:5]) # before: 4/261
            patch_loss = loss_mse(pred[:,5:], target[:,5:]) # before: 256/261
            return cls_weight * cls_loss + reg_weight * reg_loss + patch_weight * patch_loss
        return loss_func
    if mse and cosine:
        def loss_func(pred, target):
            return loss_mse(pred, target) * 0.5 - loss_cosine(pred, target) * 0.5
        return loss_func
    if mse:
        return loss_mse
    if cosine:
        return loss_cosine
    else:
        print("ERROR: Specify loss function.")
        exit()

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def make_fname_base(config):
    parts = [
        ("weighted", config["weighted"]),
        ("autoregressive", config['autoregressive_training']),
        ("distillation", config['distillation_training']),
        ("teacher", config['teacher_force']),
        ("mse", config["mse"]),
        ("cosine", config["cosine"]),
        ("t_scale", config["t_scale"]),
        ("swiglu", config["swiglu"]),
        ("sigma", config["sigma"]),
        ("start", config["start_layer"]),
        ("end", config["end_layer"]),
        ("lr", config["lr"]),
        ("cls_weight", config["cls_weight"]),
        ("reg_weight", config["reg_weight"]),
        ("patch_weight", config["patch_weight"]),
        ("seed", config["seed"]),
    ]
    prefix = f"{config['name_prefix']}_" if config['name_prefix'] else ""
    
    raptor_prefix = ""
    if config.get('raptor2'):
        raptor_prefix = "raptor2_"
    elif config.get('raptor3'):
        raptor_prefix = "raptor3_"
    elif config.get('raptor4'):
        raptor_prefix = "raptor4_"
        
    return raptor_prefix + prefix + "_".join(f"{k}_{v}" for k, v in parts)
        
def build_wandb_name(config):
    # --- 1. two‑letter codes for every boolean flag -------------
    codes = {
        "autoregressive_training": "ar",
        "distillation_training":   "di",
        "teacher_force":           "tf",
        "mse":                     "mse",
        "cosine":                  "cos",
        "t_scale":                 "ts",
        "swiglu":                  "sw",
        "weighted":                "we",
    }
    # --- 2. collect the codes that are ON -----------------------
    name_parts = [
        code                     # "ar", "di", …
        for key, code in codes.items()
        if config.get(key)       # keeps only the truthy ones
    ]
    # --- 3. always append numeric hints -------------------------
    name_parts.extend([
        f"s{config['sigma']}",   # e.g. s0.01
        f"lr{config['lr']}",     # e.g. lr1e-4
        f"cw{config['cls_weight']}",
        f"rw{config['reg_weight']}",
        f"pw{config['patch_weight']}",
        f"sl{config['start_layer']}",
        f"el{config['end_layer']}",
        f"seed{config['seed']}",
    ])
    prefix = f"{config['name_prefix']}_" if config['name_prefix'] else ""

    raptor_prefix = ""
    if config.get('raptor2'):
        raptor_prefix = "raptor2_"
    elif config.get('raptor3'):
        raptor_prefix = "raptor3_"
    elif config.get('raptor4'):
        raptor_prefix = "raptor4_"

    return raptor_prefix + prefix + "_".join(name_parts)


def autoreg_pass(model, sequence, loss_fn, start_layer, sigma, base):
    base = base.repeat(sequence.size(0))
    loss = 0.0
    x = sequence[:, 0]
    timesteps = sequence.shape[1] - 1
    for i in range(timesteps):
        curr_t = start_layer + i + 1
        x = model(x + torch.randn_like(x).mul_(sigma),
                base * curr_t, curr_t)
        loss += loss_fn(x, sequence[:, i + 1]) / timesteps
    return loss

def distill_pass(model, sequence, loss_fn, sigma, layer_idx):
    layer_idx = layer_idx.repeat(sequence.size(0), 1).reshape(-1)
    input_sequence  = torch.flatten(sequence[:, :-1, :, :], 0, 1)
    target_sequence = torch.flatten(sequence[:, 1:,  :, :], 0, 1)
    # B N D
    input_sequence.add_(torch.randn_like(input_sequence).mul_(sigma))
    pred = model(input_sequence, layer_idx)
    return loss_fn(pred, target_sequence)


def build_block(config, device):
    return Block(
            dim=config.get('dim', 768),
            num_heads=config.get('num_heads', 12),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            init_values=config['init_values'],
            t_scale=config['t_scale'],
            swiglu=config['swiglu']
    ).to(device).float()

def build_model(config, device):
    raptor_variants = [
        ('raptor2', [7], 2),
        ('raptor3', [7, 10], 3),
        ('raptor4', [4, 7, 10], 4),
    ]

    for key, thresholds, num_blocks in raptor_variants:
        if config.get(key):
            block_paths = config.get('block_paths', [])
            if len(block_paths) > 0:
                blocks = [torch.load(f"models/{path}").to(device).float() for path in block_paths]
            else:
                blocks = [build_block(config, device) for _ in range(num_blocks)]
            return Raptor(blocks, thresholds).to(device).float()

    # Default fallback
    return build_block(config, device)

def train(config):
    set_random_seed(config['seed'])
    # SETUP
    required_keys = [
        'data_dir', 'model_save_dir', 'init_values', 'sigma',
        'start_layer', 'end_layer', 'lr', 'wd', 'epochs',
        'dino_model', 'classifier_path', 'val_dir'
    ]
    for key in required_keys:
        assert key in config, f"{key} must be specified in config"
    
    use_wandb = config['wandb']
    if use_wandb:
        name = build_wandb_name(config)
        wandb.init(project=f"{config['proj_name']}", config=config, name=name,
                   group=f"{config['start_layer']}-{config['end_layer']}", 
                   tags = [
                   tag for tag in [
                        "autoreg"  if config["autoregressive_training"] else None,
                        "distill"  if config["distillation_training"]   else None,
                        "teachf"   if config["teacher_force"]           else None,
                        "mse"      if config["mse"]                     else None,
                        "cosine"   if config["cosine"]                  else None,
                        "t_scale"  if config["t_scale"]                 else None,
                        "swiglu"   if config["swiglu"]                  else None,
                        "weighted" if config["weighted"]                else None,
                        f"sigma={config['sigma']}",
                        f"lr={config['lr']}",
                    ]
                    if tag is not None                    # filters out the Nones
                    ]
        )
    device = config['device']
    os.makedirs(config['model_save_dir'], exist_ok=True)

    # FNAME
    fname_base = make_fname_base(config)

    # DATA
    dataloader = AsynchZarrLoader(
        zarr_path=config['data_dir'],
        layer_start=config['start_layer'],
        layer_end=config['end_layer'],
        batch_size=config['batch_size'],
        num_workers=8,
        queue_size=30,
        device=device
    )

    valset = datasets.ImageFolder(root=config['val_dir'], transform=imagenet_transform())
    val_loader = DataLoader(
        valset, batch_size=config['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=20
    )

    # TRAINING
    autoregressive_training = config['autoregressive_training']
    distillation_training = config['distillation_training']
    teacher_force = config['teacher_force']
    if teacher_force:
        assert not autoregressive_training and not distillation_training
    loss_fn = get_loss_func(config['mse'], config['cosine'], config['weighted'],
                            config['cls_weight'], config['reg_weight'], config['patch_weight'])

    # MODELS
    dino = DinoModelWrapper(config['dino_model'], config['classifier_path'], device=device)
    model = build_model(config, device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    steps_per_epoch = 1_000_000 // config['batch_size']
    max_steps = config['epochs'] * steps_per_epoch
    scheduler = CosineScheduler(
        optimizer, config['lr'], 1e-6, max_steps, warmup_iters=config['warmup_iters']
    )

    anneal_max  = int(config.get('tf_anneal_ratio', 1.0) * max_steps)
    tf_min      = config.get('tf_min_ratio', 0.0)
    def teacher_forcing_ratio(step):
        """Linear decay from 1.0 → tf_min over `anneal_max` steps."""
        return max(tf_min, 1.0 - step / anneal_max)

    model.train()
    losses = []

    start_layer = config['start_layer']
    end_layer = config['end_layer']
    
    base = torch.ones((1,), device=device, dtype=torch.float)
    layer_idx = torch.arange(start=start_layer + 1, end=end_layer + 1,
                             device=device, dtype=torch.float).unsqueeze(0)
    
    for step, sequence in enumerate(dataloader):
        if step >= max_steps:
            break

        sequence = sequence.to(device, non_blocking=True)
        if step == 0:
            log(f"Input sequence shape: {sequence.shape}")

        loss = 0.0
        if teacher_force:
            tf_ratio     = teacher_forcing_ratio(step)           # 1 → 0
            ar_loss      = autoreg_pass(model, sequence, loss_fn, start_layer, config['sigma'], base)
            if tf_ratio != 0:
                distill_loss = distill_pass(model, sequence, loss_fn, config['sigma'], layer_idx)
            else:
                distill_loss = 0
            loss         += tf_ratio * distill_loss + (1 - tf_ratio) * ar_loss
        elif autoregressive_training:
            loss = autoreg_pass(model, sequence, loss_fn, start_layer, config['sigma'], base)
        elif distillation_training:
            loss = distill_pass(model, sequence, loss_fn, config['sigma'], layer_idx)

        losses.append(loss.item())
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % config.get('log_interval', 500) == 0:
            avg_loss = np.mean(losses[-100:])
            log(f"Step {step}/{max_steps} | Loss: {avg_loss:.4f}")
            if use_wandb:
                wandb.log({"loss": avg_loss}, step=step, commit=step % config.get('validate_interval', 5000) != 0)
            model.train()

        if step % config.get('validate_interval', 5000) == 0:
            acc, r2 = validation_step(model, dino, val_loader, start_layer, end_layer, config['num_val_samples'])
            log(f"Step {step}/{max_steps} | Acc: {acc:.4f} | R2: {r2:.4f}")
            if use_wandb:
                wandb.log({"acc": acc, "r2": r2}, step=step, commit=True)

        if step % config.get('save_interval', 5000) == 0:
            model.eval()
            fname = f"{fname_base}_step_{step}.pt"
            torch.save(model, os.path.join(config['model_save_dir'], fname))
            log(f"Checkpoint saved at step {step}: {fname}")
            model.train()

    model.eval()
    acc, r2 = validation_step(model, dino, val_loader, start_layer, end_layer, len(valset))
    fname = f"final_{fname_base}_step_{step}.pt"
    torch.save(model, os.path.join(config['model_save_dir'], fname))
    log(f"Final validation accuracy: {acc:.4f}")
    log(f"Final validation r2      : {r2:.4f}")
    log(f"Final model saved: {fname}")

    if use_wandb:
        wandb.log({"Validation Acc": acc, "Validation R2": r2})
        wandb.finish()


def validation_step(model, dino, dataloader, start_layer, end_layer, max_samples):
    # Setup
    model.eval()
    dino = dino.eval()
    accs = 0
    r2s = 0
    num_samples = 0
    with torch.no_grad():  
        for i, batch in enumerate(dataloader):
            # Prepare batch
            x = batch[0].cuda()
            y = batch[1].cuda()
            batch_size = x.size(0)
            num_samples += batch_size

            # Forward
            _, a_dino = dino(x, layer_start=0, layer_end=end_layer)
            pred = a_dino[:,start_layer]
            for output_layer in range(start_layer + 1, end_layer + 1): # 0 + 1 = 1, 7 + 1 = 8 so (1, 7 inclusive so 7 iters)
                t = torch.ones(batch_size, device=x.device, dtype=torch.float) * output_layer
                pred = model(pred, t, output_layer)
            logits, _ = dino(pred, layer_start=end_layer + 1) # 7 + 1 = 8, index 7 for dino which is layer 8
            pred_label = logits.argmax(dim=1)
            accs += (pred_label == y).float().sum().cpu()

            a_end = a_dino[:,end_layer] # b t d
            r2 = r2_score(pred.reshape(-1, pred.size(2)), a_end.reshape(-1, a_end.size(2))) # 'b t d -> (b t) d'
            r2s += r2 * batch_size

            if num_samples >= max_samples:
                break

        return accs / num_samples, r2s / num_samples


if __name__ == "__main__":
    from config import parse_args, build_config
    args = parse_args()
    config = build_config(args)
    train(config)