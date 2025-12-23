import torch
import torch.nn as nn

from raptor import Raptor


class DummyBlock(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x, t):
        return x + self.value


def test_raptor_selects_expected_block():
    blocks = [DummyBlock(1.0), DummyBlock(2.0), DummyBlock(3.0)]
    model = Raptor(blocks, thresholds=[2, 5])
    x = torch.zeros(1, 1, 1)
    t = torch.tensor([1.0])

    out1 = model(x, t, t_integer=1)
    out2 = model(x, t, t_integer=3)
    out3 = model(x, t, t_integer=9)

    assert out1.item() == 1.0
    assert out2.item() == 2.0
    assert out3.item() == 3.0
