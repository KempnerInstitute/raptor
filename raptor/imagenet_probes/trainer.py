import torch
from torch import nn
import timm
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import random
from PIL import Image


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


def get_transforms():
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                std=IMAGENET_DEFAULT_STD),
        ])
    val_tf = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                            std=IMAGENET_DEFAULT_STD),
    ])
    return train_tf, val_tf


def run_train(model, device, train_path, val_path, classifier_save_path, seed, variant, model_seed=None):
    model.eval()
    set_random_seed(seed)

    BATCH_SIZE = 512
    
    train_tf, val_tf = get_transforms()
    trainset = datasets.ImageFolder(train_path, transform=train_tf)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    valset = datasets.ImageFolder(val_path, transform=val_tf)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=False)

    EPOCHS = 15
    base_lr         = 1e-4
    weight_decay    = 1e-2
    grad_clip_norm  = 1.0
    warmup_iters    = 100         # NEW – set this to whatever you like
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=weight_decay)
    iters_per_epoch = len(train_loader)
    total_iters = EPOCHS * iters_per_epoch
    warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=base_lr * 1e-2)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters]
    )

    best_acc = 0.0
    def run_epoch(model, loader, train=True):
        model.eval()
        if train:
            model.classifier.train()
        else:
            model.classifier.eval()

        running_loss, correct, total = 0.0, 0, 0
        torch.set_grad_enabled(train)

        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            batch_size = x.size(0)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                x = model.get_layer_0(x)
                x = model.iterate_layers(x)
            logits = model.classify(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), grad_clip_norm)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            curr_correct = (preds == y).sum().item()
            correct += curr_correct
            total += batch_size

            if train and i != 0 and i % 100 == 0:
                print(i, len(loader))
                print(curr_correct / batch_size)

        avg_loss = running_loss / max(total, 1)
        acc = correct / total
        return avg_loss, acc


    # Initial validation
    print("Running initial validation...")
    val_loss, val_acc = run_epoch(model, val_loader, train=False)
    print(f"Initial val loss {val_loss:.4f} acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.classifier.state_dict(), classifier_save_path)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, train=True)
        val_loss,   val_acc  = run_epoch(model, val_loader,   train=False)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.classifier.state_dict(), classifier_save_path)
            
    print(f"Best val acc: {best_acc:.4f}")
    
    # Save results
    result_data = {
        "variant": variant,
        "seed": seed, # This is the probe training seed
        "model_seed": model_seed, # This is the pretraining seed (if raptor)
        "best_val_acc": best_acc
    }
    save_results_to_jsonl(result_data)