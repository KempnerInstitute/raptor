import torch
import torch.nn as nn
from typing import List

from classifier import PatchDepthClassifier

class RaptorWrapper(nn.Module):

    def __init__(self, raptor, dino, grid_size=(16, 16), dmin=1e-3, dmax=10.0):
        super().__init__()

        self.raptor = raptor
        self.dino = dino
        self.classifier = PatchDepthClassifier(
            num_bins=256,
            grid_size=grid_size,
            upsample_factor=4,
            min_depth=dmin,
            max_depth=dmax,
            embed_dim=768,
        )

    """
    Run every block from layer_start to layer_end and return all the activations and the logits of the classifier.

    Params:
        x: Tensor of shape B x N x D or (B x 3 x H x W). If (B x 3 x H x W), runs patching and feature extraction first.
        layer_start: layer index index in the range(1, 12) (one-based layer). Defaults to 1.
        layer_end: layer index index in the range of 1, 12 (one-based layer). Defaults to 12.
    Returns:
        Tuple of logits (B x C) and tensor of activations (B x L x N x D).
    """
    def forward(self, x, layer_start=1, layer_end=12):
        base = torch.ones(x.size(0), device=x.device, dtype=torch.float)
        activations = []
        if layer_start == 0:
            x = self.dino.prepare_tokens_with_masks(x) # B x N x D
            activations.append(x)
            layer_start = 1
        for t in range(layer_start, layer_end + 1):
            x = self.raptor(x, base * t, t)
            activations.append(x)
        if layer_start < 13:
            activations = torch.stack(activations, dim=1)
        logits = self.classify(x)
        return logits, activations

    """
    Run every block from block_start to layer_end (inclusive) and return the final representation.

    Params:
        x: Tensor of shape B x N x D
        layer_start: layer index index in the range of 1, 12 (one-based layer)
        layer_end: layer index index in the range of 1, 12 (one-based layer)
    Returns:
        Layer 12 activation: tensor of shape B x N x D. If block_start > 12, returns x
    """
    def iterate_layers(self, x: torch.Tensor, layer_start: int = 1, layer_end: int=12) -> torch.Tensor:
        base = torch.ones(x.size(0), device=x.device, dtype=torch.float)
        for t in range(layer_start, layer_end + 1):
            x = self.raptor(x, base * t, t)
        return x

    def get_layer_0(self, x):
        return self.dino.prepare_tokens_with_masks(x) # B x N x D
            
    """
    Replicates the standard DINOv2 classifier recipe (normalise + CLS & patch mean).

    Params:
        x: Tensor of shape B x N x D
    Returns:
        logits of shape B x C (C == 1000 for 1000 INet1k classes)
    """
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.dino.norm(x)
        logits = self.classifier(x)
        return logits # B x 256 x 64 x 64

    def decode_depth(self, logits: torch.Tensor) -> torch.Tensor:
        return self.classifier.decode_depth(logits)

    def _load_dino(self, source="dinov2_vitb14_reg"):
        return torch.hub.load("facebookresearch/dinov2", source)