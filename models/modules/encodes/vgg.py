import torch.nn as nn
from models.modules.blocks.blocks3d import create_conv_block


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='crb', num_groups=1, padding=1):
        super(DoubleConv, self).__init__()
        conv1_in_channels = in_channels
        conv1_out_channels = max(in_channels, out_channels // 2)
        conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        self.conv1 = create_conv_block(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups, padding)
        self.conv2 = create_conv_block(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, order='crb'):
        super(Down, self).__init__()
        self.downsample = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_channels, out_channels, order=order)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        return x


class VggEncode(nn.Module):
    def __init__(self, in_channels, depth, f_maps=16):
        super(VggEncode, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        if isinstance(f_maps, int):
            f_maps = [f_maps*2**x for x in range(self.depth + 1)]
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert self.depth + 1 == len(f_maps), "Required at least 2 levels in the U-Net when once downsample"
        self.features_channels = f_maps

        self.in_conv = DoubleConv(in_channels, f_maps[0])
        self.down_blocks = nn.ModuleList([Down(f_maps[i], f_maps[i+1]) for i in range(self.depth)])

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




