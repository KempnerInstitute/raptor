import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random
from torchvision import transforms
import numpy as np
import itertools
import json
import os
from dataset import NYUv2CSVWithBins, PairedRandomResizedCrop, PairedHorizontalFlip, make_eigen_valid_mask


def save_results_to_jsonl(result_data, filename="results.jsonl"):
    """Appends a dictionary of results to a JSON Lines file."""
    with open(filename, "a") as f:
        f.write(json.dumps(result_data) + "\n")


class CenterPadding(nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


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


def adabins_loss(logits, depth_pred, gt_depth, gt_bins, ignore_index):
    mask = gt_depth > 0
    if mask.sum() == 0:
        return depth_pred, torch.tensor(0.0)
    pred_masked = depth_pred[mask]
    gt_masked = gt_depth[mask]
    log_diff = torch.log(pred_masked + 1e-3) - torch.log(gt_masked + 1e-3)
    si_loss = torch.mean(log_diff**2) - 0.85 * torch.mean(log_diff)**2
    ce_loss = F.cross_entropy(logits, gt_bins, ignore_index=ignore_index, reduction='mean')
    total_loss = 10.0 * si_loss + ce_loss
    return total_loss


def run_train(model, device, DATA_FOLDER, CLASSIFIER_PATH, IMG_SIZE, NYU_MIN, NYU_MAX, seed, variant, model_seed=None):
    set_random_seed(seed)
    IGNORE_INDEX = 256
    BATCH_SIZE = 128
    VAL_BATCH_SIZE = 8
    EPOCHS = 25

    trainset = NYUv2CSVWithBins(
        dataset_folder=DATA_FOLDER,
        train=True,
        paired_random_resized_crop=PairedRandomResizedCrop(size=IMG_SIZE, scale=(0.5, 2.0), ratio=(4/3, 4/3)),
        paired_hflip=PairedHorizontalFlip(p=0.5),
        color_jitter=transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        dmin=NYU_MIN,
        dmax=NYU_MAX
    )
    valset = NYUv2CSVWithBins(
        dataset_folder=DATA_FOLDER,
        train=False,
        dmin=NYU_MIN,
        dmax=NYU_MAX
    )
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset,   batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def run_epoch(loader, train: bool):
        model.eval()
        if train:
            model.classifier.train()
        else:
            model.classifier.eval()

        running_loss, total = 0.0, 0
        sqerr_sum, valid_count = 0.0, 0
        padding = CenterPadding(14)

        for i, batch in enumerate(loader):
            if train and i % 10 == 0:
                print(i, len(loader), flush=True)
            imgs, depth_gt, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            # (B,1,H,W) — H,W=224 for train; ~480x640 for val
            depth_gt = depth_gt.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()        # (B,H,W)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    imgs = padding(imgs)
                    x = model.get_layer_0(imgs)
                    x = model.iterate_layers(x)
                logits = model.classify(x)   # [B,K,H'*4,W'*4] = [B, 256, 120, 160]

            with torch.cuda.amp.autocast(False):
                labels_ds = F.interpolate(
                    labels.unsqueeze(1).float(),
                    size=logits.shape[-2:],
                    mode="nearest").squeeze(1).long()
                logits = logits.float()
                pred_depth = model.decode_depth(logits)
                pred_depth = F.interpolate(pred_depth, size=(depth_gt.size(-2), depth_gt.size(-1)), mode="bilinear")
                loss = adabins_loss(logits, pred_depth, depth_gt, labels_ds, IGNORE_INDEX)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            total += bs

            # -------- Validation RMSE at native GT resolution with Eigen crop --------
            if not train:
                with torch.no_grad():
                    valid = make_eigen_valid_mask(depth_gt, NYU_MIN, NYU_MAX)
                    if valid.any():
                        diff = (pred_depth - depth_gt)[valid]
                        sqerr_sum += (diff * diff).sum().item()
                        valid_count += valid.sum().item()

        avg_loss = running_loss / max(total, 1)

        if train:
            return avg_loss, None
        else:
            rmse = math.sqrt(sqerr_sum / max(valid_count, 1)) if valid_count > 0 else float("nan")
            return avg_loss, rmse

    # ---------------- train ------------------
    best_val = float("inf")

    # Initial validation
    print("Running initial validation...")
    val_loss, val_rmse = run_epoch(val_loader, train=False)
    print(f"Initial val loss {val_loss:.4f} | val RMSE {val_rmse:.4f} m")

    if val_rmse < best_val:
        best_val = val_rmse
        torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)

    for epoch in range(1, EPOCHS + 1):
        train_loss, _ = run_epoch(train_loader, train=True)
        val_loss, val_rmse = run_epoch(val_loader,   train=False)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | "
              f"val RMSE {val_rmse:.4f} m")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.classifier.state_dict(), CLASSIFIER_PATH)

    print(f"Best val RMSE: {best_val:.4f}")

    # Save results
    result_data = {
        "variant": variant,
        "seed": seed,  # This is the probe training seed
        "model_seed": model_seed,  # This is the pretraining seed (if raptor)
        "best_val_rmse": best_val
    }
    save_results_to_jsonl(result_data)
