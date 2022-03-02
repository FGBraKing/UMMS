import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, norm_layer=None):
        super(SEModule, self).__init__()

        self.ave_pool = nn.AdaptiveAvgPool3d(1)
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv3d(channels, rd_channels, kernel_size=1, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv3d(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = self.ave_pool(x)
        x_se = self.act(self.bn(self.fc1(x_se)))
        x_se = self.gate(self.fc2(x_se))
        return x * x_se


SqueezeExcite = SEModule  # alias


class EffectiveSEModule(nn.Module):
    """ 'Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """
    def __init__(self, channels):
        super(EffectiveSEModule, self).__init__()
        self.fc = nn.Conv3d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)

        x_se = self.gate(self.fc(x_se))
        return x * x_se


EffectiveSqueezeExcite = EffectiveSEModule  # alias


if __name__ == '__main__':
    input=torch.randn(50, 512, 7, 7, 7)
    ker_size = input.shape[2]
    se = SEModule(channels=512)
    output = se(input)
    print(output.shape)










