import torch
import torch.nn as nn

class Raptor(nn.Module):
    def __init__(self, blocks, thresholds):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.thresholds = thresholds
        # Ensure we have one more block than thresholds
        assert len(self.blocks) == len(self.thresholds) + 1

    def forward(self, x, t, t_integer):
        # Iterate through thresholds to find the matching block
        for i, threshold in enumerate(self.thresholds):
            if t_integer <= threshold:
                return self.blocks[i](x, t)
        # If greater than all thresholds, use the last block
        return self.blocks[-1](x, t)