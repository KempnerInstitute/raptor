import torch
import torch.nn as nn
from typing import List

class DinoModelWrapper(nn.Module):

    def __init__(self, dino_model="dinov2_vitb14_reg", classifier_path="dinov2_vitg14_reg4_linear_head", device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.dino = self._load_dino(dino_model).eval().to(self.device)
        W, b = self._load_classifier(classifier_path)
        self.W = W.to(device)
        self.b = b.to(device)

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
        activations = []
        if layer_start == 0:
            x = self.dino.prepare_tokens_with_masks(x) # B x N x D
            activations.append(x)
            layer_start = 1
        for i in range(layer_start - 1, layer_end):
            x = self.dino.blocks[i](x)
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
        for i in range(layer_start - 1, layer_end):
            x = self.dino.blocks[i](x)
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
        # Keep CLS token (index 0) and the mean of patch tokens after a 4‑token prompt
        x = torch.cat([x[:, 0], x[:, 1 + 4 :].mean(dim=1)], dim=1)
        return x @ self.W.T + self.b

    def _load_classifier(self, fpath='dinov2_vitg14_reg4_linear_head.pth'):
        classifier = torch.load(fpath, map_location="cpu")
        W = classifier['weight']
        b = classifier['bias']
        return W, b

    def _load_dino(self, source="dinov2_vitb14_reg"):
        return torch.hub.load("facebookresearch/dinov2", source)