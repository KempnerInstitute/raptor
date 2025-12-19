import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import pandas as pd
from paths import NYUD_DIR

# --- Standard NYU Depth v2 range ---
NYU_MIN, NYU_MAX = 0.001, 10.0

# ---------- Eval-time helper: standard Eigen valid region ----------
def make_eigen_valid_mask(depth: torch.Tensor, dmin: float, dmax: float):
    """
    depth: (B,1,H,W) or (1,H,W) tensor in meters (native 480x640 expected).
    Returns:
      mask: (B,1,H,W) bool — true for valid pixels in the Eigen crop and within [dmin, dmax].
    """
    assert depth.ndim in (3,4)
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)

    B, _, H, W = depth.shape
    # Eigen crop box for 480x640: rows [45:471), cols [41:601)
    top, bottom = 45, 471
    left, right = 41, 601
    crop_mask = torch.zeros((B,1,H,W), dtype=torch.bool, device=depth.device)
    h0, h1 = max(0, top), min(H, bottom)
    w0, w1 = max(0, left), min(W, right)
    crop_mask[:, :, h0:h1, w0:w1] = True

    finite = torch.isfinite(depth) & (depth > 0)
    in_range = (depth >= dmin) & (depth <= dmax)
    return crop_mask & finite & in_range


# -----------------------------
# Paired (RGB+Depth) transforms
# -----------------------------

class PairedRandomResizedCrop:
    """
    RandomResizedCrop that applies the same crop to (img, depth).
    Target is (420, 560). ratio is W/H. NYU (640×480) has W/H ≈ 4/3.
    We keep the ratio tight around 4/3 to preserve FOV.
    """
    def __init__(self,
                 size=(420, 560),
                 scale=(0.5, 2.0),
                 ratio=(4/3, 4/3),
                 interpolation_img=InterpolationMode.BILINEAR,
                 interpolation_depth=InterpolationMode.NEAREST):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interp_img = interpolation_img
        self.interp_depth = interpolation_depth

    def __call__(self, img_pil, depth_pil):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img_pil, scale=self.scale, ratio=self.ratio
        )
        img_c = TF.resized_crop(img_pil, i, j, h, w, self.size, self.interp_img)
        dep_c = TF.resized_crop(depth_pil, i, j, h, w, self.size, self.interp_depth)
        return img_c, dep_c


class PairedHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_pil, depth_pil):
        if torch.rand(1).item() < self.p:
            img_pil = TF.hflip(img_pil)
            depth_pil = TF.hflip(depth_pil)
        return img_pil, depth_pil


class PairedResize:
    """
    Resize image to exactly (420, 560). (Depth is NOT touched in val.)
    """
    def __init__(self, size=(420, 560),
                 interpolation_img=InterpolationMode.BILINEAR,
                 interpolation_depth=InterpolationMode.NEAREST):
        self.size = size
        self.interp_img = interpolation_img
        self.interp_depth = interpolation_depth

    def __call__(self, img_pil):
        return TF.resize(img_pil, self.size, self.interp_img)


# -----------------------------
#  Binning utilities (unchanged)
# -----------------------------

def depth_to_bin_indices(
    depth: torch.Tensor,
    num_bins: int = 256,
    dmin: float = NYU_MIN,
    dmax: float = NYU_MAX,
    ignore_index: int = 255,
):
    """
    depth: (B,1,H,W) or (1,H,W) in meters.
    Returns:
      bin_idx: (B,H,W) long with values in [0, num_bins-1] or ignore_index
      valid_mask: (B,1,H,W) bool
    """
    assert depth.ndim in (3,4)
    if depth.ndim == 3:
        depth = depth.unsqueeze(0)  # (1,1,H,W)

    valid = torch.isfinite(depth) & (depth > 0)
    in_range = (depth >= dmin) & (depth <= dmax)
    valid = valid & in_range

    edges = torch.linspace(dmin, dmax, num_bins + 1, device=depth.device, dtype=depth.dtype)
    bin_idx = torch.bucketize(depth.clamp(dmin, dmax - 1e-6), edges, right=False) - 1  # (B,1,H,W)
    bin_idx = bin_idx.squeeze(1).to(torch.long)  # (B,H,W)

    bin_idx[~valid.squeeze(1)] = ignore_index
    return bin_idx, valid


# -----------------------------
# Combined dataset
# -----------------------------

class NYUv2CSVWithBins(Dataset):
    """
    One-stop dataset:
      - Loads from CSVs (train/test in given folder)
      - Train: apply paired crop (+optional flip) to both image & depth
      - Val:   resize image only; depth untouched
      - Converts depth mm -> meters
      - Returns (img, depth_m, bin_idx)
    """
    def __init__(
        self,
        dataset_folder=NYUD_DIR,
        train=True,
        paired_random_resized_crop=None,  # PairedRandomResizedCrop
        paired_hflip=None,                # PairedHorizontalFlip or None
        color_jitter=None,
        num_bins=256,
        dmin=NYU_MIN,
        dmax=NYU_MAX,
        ignore_index=255,
        depth_scale_mm_to_m=True,
    ):
        self.train = train
        self.dataset_folder = dataset_folder
        csv_name = "data/nyu2_train.csv" if train else "data/nyu2_test.csv"
        self.df = pd.read_csv(os.path.join(dataset_folder, csv_name), header=None)

        # Transforms
        self.paired_random_resized_crop = paired_random_resized_crop
        self.paired_hflip = paired_hflip
        self.color_jitter = color_jitter

        # Binning params
        self.num_bins = num_bins
        self.dmin = dmin
        self.dmax = dmax
        self.ignore_index = ignore_index

        # Depth scaling
        self.depth_scale_mm_to_m = depth_scale_mm_to_m

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def _load_rgb(self, path):
        return Image.open(self.dataset_folder + "/" + path).convert("RGB")

    def _load_depth(self, path):
        d = Image.open(self.dataset_folder + "/" + path)
        if d.mode != "I;16":
            d = d.convert("I;16")
        return d

    def __getitem__(self, idx):
        rgb_path = self.df.iloc[idx, 0]
        depth_path = self.df.iloc[idx, 1]

        img = self._load_rgb(rgb_path)
        depth = self._load_depth(depth_path)

        if self.train:
            img, depth = self.paired_random_resized_crop(img, depth)
            img, depth = self.paired_hflip(img, depth)
            img = self.color_jitter(img)

        depth_np = np.array(depth, dtype=np.float32)  # raw, likely millimeters
        if self.train:
            depth_np = depth_np / 255.0
            depth_np = depth_np * 10.0
        else:
            depth_np = depth_np / 1000.0  # meters
        depth_t = torch.from_numpy(depth_np).unsqueeze(0)               # [1,H,W], meters

        # bins from meters
        bin_idx, _ = depth_to_bin_indices(
            depth_t, num_bins=self.num_bins, dmin=self.dmin, dmax=self.dmax, ignore_index=self.ignore_index
        )
        bin_idx = bin_idx.squeeze(0)  # (H,W)

        # Convert to tensors
        img_t = TF.to_tensor(img)  # [3,H,W], float32 [0,1]
        img_t = self.normalize(img_t)

        return img_t, depth_t, bin_idx