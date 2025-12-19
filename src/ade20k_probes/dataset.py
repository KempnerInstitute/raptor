import os, glob
from typing import Optional, Tuple, List, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop as TVRandomResizedCrop
import random

# ---------- Joint transform helpers (image & mask together) ----------

class ComposePair:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    def __call__(self, img: Image.Image, mask: Image.Image):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomResizedCropSeg:
    """
    RandomResizedCrop for segmentation: same crop/resize for img & mask.
    `size` may be int or (h, w) like torchvision.
    """
    def __init__(self, size, scale=(0.5, 1.0), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self, img: Image.Image, mask: Image.Image):
        i, j, h, w = TVRandomResizedCrop.get_params(img, self.scale, self.ratio)
        img  = TF.resized_crop(img,  i, j, h, w, self.size, Image.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, Image.NEAREST)
        return img, mask

class RandomHorizontalFlipSeg:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, img: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask

# ---------- Dataset ----------

class ADE20KDataset(Dataset):
    """
    ADE20K loader (SceneParse150) with 150 classes and background ignored.
    Expected structure (MIT release / SceneParse150):
        <root>/
          images/
            training/*.jpg
            validation/*.jpg
          annotations/
            training/*.png   (or *_seg.png)
            validation/*.png (or *_seg.png)

    Labels in raw masks: 0=background, 1..150=classes.
    We map to: 255=ignore (background), and 1..150 -> 0..149.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        split: str = "training",               # "training" or "validation"
        target_size: Optional[Tuple[int,int]] = None,  # (W,H) resize if no joint transform handles size
        reduce_zero_label: bool = True,        # keep True to get 150 classes w/ bg ignored
        ignore_index: int = 255,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
    ):
        assert split in ("training", "validation")
        self.root = root
        self.split = split
        self.target_size = target_size
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index
        self.transform = transform

        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        if not (os.path.isdir(self.img_dir) and os.path.isdir(self.ann_dir)):
            raise FileNotFoundError(
                f"Couldn't find '{self.img_dir}' and '{self.ann_dir}'. "
                f"Make sure you pointed root to the ADE20K 'ADEChallengeData2016' (SceneParse150) folder."
            )

        self.img_paths: List[str] = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No JPGs found in {self.img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def _mask_path_for(self, img_path: str) -> str:
        base = os.path.splitext(os.path.basename(img_path))[0]
        p1 = os.path.join(self.ann_dir, base + ".png")
        p2 = os.path.join(self.ann_dir, base + "_seg.png")
        if os.path.isfile(p1): return p1
        if os.path.isfile(p2): return p2
        raise FileNotFoundError(f"Mask for {img_path} not found in {self.ann_dir}")

    @staticmethod
    def _reduce_zero_label(mask: np.ndarray, ignore_index: int) -> np.ndarray:
        # map 0 -> ignore_index, and shift 1..150 -> 0..149
        out = mask.copy()
        zero = (out == 0)
        out = out - 1
        out[zero] = ignore_index
        return out

    def __getitem__(self, i: int):
        img_path = self.img_paths[i]
        msk_path = self._mask_path_for(img_path)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path)  # 'P' or 'L' with class indices

        # If a joint transform is provided (e.g., RandomResizedCropSeg), it should take care of size.
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        elif self.target_size is not None:
            # Fallback deterministic resize
            img  = img.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)

        # To tensor & normalize AFTER spatial transforms
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

        mask_np = np.array(mask, dtype=np.uint8)
        if self.reduce_zero_label:
            mask_np = self._reduce_zero_label(mask_np, self.ignore_index)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return img, mask_t