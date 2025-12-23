import torch
from torch import nn
import timm
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random

import sys
sys.path.append("../")
from dataset import ADE20KDataset, ComposePair, RandomResizedCropSeg, RandomHorizontalFlipSeg
from dino_wrapper import DinoModelWrapper
from raptor_wrapper import RaptorWrapper
from ignite.metrics import ConfusionMatrix, IoU
import json
import os

def save_results_to_jsonl(result_data, filename="results.jsonl"):
    """Appends a dictionary of results to a JSON Lines file."""
    with open(filename, "a") as f:
        f.write(json.dumps(result_data) + "\n")

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


def run_train(model, device, DATA_PATH, CLASSIFIER_PATH, seed, variant, model_seed=None):
    model.eval()
    IGNORE_INDEX = 255
    set_random_seed(seed)

    target_size = (518, 518)
    train_tf = ComposePair([
        RandomResizedCropSeg(size=target_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        RandomHorizontalFlipSeg(p=0.5),
    ])
    trainset = ADE20KDataset(root=DATA_PATH, split="training", transform=train_tf, reduce_zero_label=True, ignore_index=IGNORE_INDEX)
    valset = ADE20KDataset(root=DATA_PATH, split="validation", target_size=target_size, reduce_zero_label=True, ignore_index=IGNORE_INDEX)
    train_loader = DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        valset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    EPOCHS = 10
    num_epochs      = EPOCHS
    base_lr         = 1e-3
    weight_decay    = 1e-2
    grad_clip_norm  = 1.0
    warmup_iters    = 100
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=weight_decay)
    iters_per_epoch = len(train_loader)
    total_iters = num_epochs * iters_per_epoch
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=base_lr * 1e-2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters]
    )


    best_iou = 0.0
    def run_epoch(model, loader, train=True):
        model.eval()
        if train:
            model.classifier.train()
        else:
            model.classifier.eval()
        
        cm = ConfusionMatrix(num_classes=150)
        iou_metric = IoU(cm).to(device)
        iou_metric.reset()

        running_loss, correct, total = 0.0, 0, 0
        torch.set_grad_enabled(train)

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                x = model.get_layer_0(imgs)
                x = model.iterate_layers(x)
            logits = model.classify(x)
            logits = F.interpolate(logits.float(), size=target_size, mode="bilinear", align_corners=False)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += ((preds == labels).sum((1, 2)) / (518 * 518)).sum().item()
            total += labels.size(0)
            iou_metric.update((logits, labels))

        avg_loss = running_loss / max(total, 1)
        acc = correct / max(total, 1)

        per_class_iou = iou_metric.compute()            # tensor of shape [num_classes]
        miou = per_class_iou.mean().item()
        return avg_loss, acc, miou


    # Initial validation
    print("Running initial validation...")
    val_loss, val_acc, val_iou = run_epoch(model, val_loader, train=False)
    print(f"Initial val loss {val_loss:.4f} acc {val_acc:.4f} iou {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_iou = run_epoch(model, train_loader, train=True)
        val_loss,   val_acc, val_iou  = run_epoch(model, val_loader,   train=False)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} iou {train_iou:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} iou {val_iou:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)
            
    print(f"Best val iou: {best_iou:.4f}")

    # Save results
    result_data = {
        "variant": variant,
        "seed": seed, # This is the probe training seed
        "model_seed": model_seed, # This is the pretraining seed (if raptor)
        "best_val_iou": best_iou
    }
    save_results_to_jsonl(result_data)