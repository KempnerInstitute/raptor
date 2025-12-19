import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, init_values=1e-5,
                 t_scale=True, swiglu=False):
        super().__init__()
        self.t_scale = t_scale

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ls1 = LayerScale(dim, init_values) if init_values is not None else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        if swiglu:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hid * 2),
                SwiGLU(),
                nn.Linear(hid, dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hid),
                nn.GELU(),
                nn.Linear(hid, dim),
            )
        self.ls2 = LayerScale(dim, init_values) if init_values is not None else nn.Identity()

        if t_scale:
            self.t_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, 3 * dim),
            )
            nn.init.normal_(self.t_proj[-1].weight, mean=0.0, std=1e-4)
            nn.init.normal_(self.t_proj[-1].bias, mean=0.0, std=1e-4)

    def forward(self, x, t=None, t_integer=None):
        x_norm = self.norm1(x)
        if self.t_scale:
            t = t.unsqueeze(1).unsqueeze(2)  # broadcast over sequence length
            t = self.t_proj(t) + 1           # (B,1,3D), center at 1 so we start as identity
            t1, t2, t3 = t.chunk(3, dim=-1)

            x = x + t1 * self.ls1(self.attn(x_norm, x_norm, x_norm)[0])
            x = x + t2 * self.ls2(self.mlp(self.norm2(x)))
            return t3 * x

        x = x + self.ls1(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x