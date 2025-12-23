import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDepthClassifier(nn.Module):
    """
    Monocular depth classifier head (lin. 1):
      - Input: last-layer tokens B x N x D with CLS at index 0 and 4 register tokens (1..4).
      - Uses patch tokens starting from index 5.
      - Concatenates CLS feature to each patch token along feature dim (-> 2D).
      - Reshapes to spatial grid (H, W), upsamples by factor=4 (bilinear).
      - Applies a 1x1 conv (linear per-pixel classifier) -> num_bins logits.

    Args:
        num_bins (int): Number of uniformly spaced depth bins. Default: 256.
        grid_size (Optional[Tuple[int, int]]): (H, W) of the patch grid BEFORE upsampling.
            If None, we infer a square grid from #patch_tokens. (Requires perfect square.)
        upsample_factor (int): Spatial upsample factor. Default: 4.
        min_depth (float): Minimum depth (meters) for linear bins. Default: 0.1.
        max_depth (float): Maximum depth (meters) for linear bins. Default: 10.0.

    Forward:
        tokens: FloatTensor of shape (B, N, D)
        Returns: logits of shape (B, num_bins, H*upsample_factor, W*upsample_factor)
    """

    def __init__(
        self,
        num_bins: int = 256,
        grid_size: Tuple[int, int] = (16, 16),   # 256 patch tokens from 224×224 @ patch size 14
        upsample_factor: int = 4,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.grid_size = grid_size
        self.upsample_factor = upsample_factor
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.embed_dim = embed_dim

        self.H = grid_size[0]
        self.W = grid_size[1]

        self.classifier = nn.Conv2d(self.embed_dim * 2, self.num_bins, kernel_size=1, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor (B, N, D). tokens[:, 0] is CLS; tokens[:, 1:5] are registers; tokens[:, 5:] are patches.
        Returns:
            logits: Tensor (B, num_bins, H*scale, W*scale)
        """
        B, N, D = tokens.shape

        cls = tokens[:, 0:1, :]               # (B, 1, D)
        patch_tokens = tokens[:, 5:, :]       # (B, P, D)
        P = patch_tokens.shape[1]

        # Concatenate CLS to each patch token along feature dim -> (B, P, 2D)
        cls_expanded = cls.expand(B, P, D)
        patch_feat = torch.cat([patch_tokens, cls_expanded], dim=-1)  # (B, P, 2D)

        # Reshape to image-like tensor (B, 2D, H, W)
        patch_feat = patch_feat.view(B, self.H, self.W, 2 * D).permute(0, 3, 1, 2).contiguous()

        # Bilinear upsample by factor=4 (as specified)
        if self.upsample_factor != 1:
            patch_feat = F.interpolate(
                patch_feat,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )

        # Build the 1x1 linear classifier if needed, then apply
        logits = self.classifier(patch_feat)  # (B, num_bins, H*scale, W*scale)
        return logits

    def decode_depth(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Decode classifier logits to a continuous depth map via expected value over linear bins.

        Args:
            logits: (B, num_bins, H, W)
        Returns:
            depths: (B, 1, H, W) in meters
        """
        probs = torch.softmax(logits, dim=1)  # (B, K, H, W)
        centers = self.get_bin_centers(device=logits.device, dtype=logits.dtype)  # (K,)
        depths = (probs * centers.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return depths

    def get_bin_centers(self, device=None, dtype=None) -> torch.Tensor:
        """Bin centers (midpoints), shape (num_bins,)."""
        edges = self.get_bin_edges(device=device, dtype=dtype)
        return 0.5 * (edges[:-1] + edges[1:])

    def get_bin_edges(self, device=None, dtype=None) -> torch.Tensor:
        """Linearly spaced bin edges in [min_depth, max_depth], shape (num_bins+1,)."""
        return torch.linspace(
            self.min_depth, self.max_depth, self.num_bins + 1, device=device, dtype=dtype
        )
