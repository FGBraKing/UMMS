import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


from models.modules.blocks.blocks3d import same_convlution, downsample_convlution, upsample_deconvlution


class InConvBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InConvBlock, self).__init__()


class DownBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DownBlock, self).__init__()


class CustomEncoder(nn.Module):
    def __init__(self, in_channels, depth, f_maps=16):
        super(CustomEncoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        if isinstance(f_maps, int):
            f_maps = [f_maps*2**x for x in range(self.depth + 1)]
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert self.depth + 1 == len(f_maps), "Required at least 2 levels in the U-Net when once downsample"
        self.features_channels = f_maps

        self.in_conv = InConvBlock(in_channels, f_maps[0])
        self.down_blocks = nn.ModuleList([DownBlock(f_maps[i], f_maps[i+1]) for i in range(self.depth)])

    def forward(self, x):
        encoders_features = []
        x = self.in_conv(x)
        encoders_features.append(x)
        for encode in self.down_blocks:
            x = encode(x)
            encoders_features.append(x)
        return encoders_features

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self.features_channels




