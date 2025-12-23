import torch

from block import Block


def test_block_forward_with_t_scale():
    block = Block(dim=16, num_heads=4, t_scale=True, swiglu=False)
    x = torch.randn(2, 6, 16)
    t = torch.tensor([1.0, 2.0])

    out = block(x, t, t_integer=1)

    assert out.shape == x.shape


def test_block_forward_without_t_scale():
    block = Block(dim=16, num_heads=4, t_scale=False, swiglu=True)
    x = torch.randn(2, 6, 16)

    out = block(x)

    assert out.shape == x.shape
