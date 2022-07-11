import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.modules.blocks.blocks3d import conv3x3x3, conv1x1x1, create_conv_block, same_convlution
from models.auxiliary_funs import get_normalization3d, get_activation
from .resnext import ResnextBottleneck, ConvBlock


class Downsample(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1, groups=1, down_type="C", norm_type='batch', act_type='lrelu'):
        super(Downsample, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        if down_type == "A":
            self.conv_down = nn.Sequential(
                nn.AvgPool3d(2) if stride == 2 else nn.Identity(),
                nn.Conv3d(in_chs, out_chs, 1, 1, 0, groups=groups, bias=use_bias),
                get_normalization3d(out_chs, norm_type),
                get_activation(act_type)
            )
        elif down_type == "B":
            self.conv_down = nn.Sequential(
                nn.MaxPool3d(2) if stride == 2 else nn.Identity(),
                nn.Conv3d(in_chs, out_chs, 1, 1, 0, groups=groups, bias=use_bias),
                get_normalization3d(out_chs, norm_type),
                get_activation(act_type)
            )
        else:
            self.conv_down = nn.Sequential(
                nn.Conv3d(in_chs, out_chs, 3, stride, 1, groups=groups, bias=use_bias),
                get_normalization3d(out_chs, norm_type),
                get_activation(act_type)
            )

    def forward(self, x):
        return self.conv_down(x)


class CrossStage(nn.Module):
    """Cross Stage."""
    def __init__(
            self,
            in_chs,
            out_chs,
            stride,
            depth,
            block_ratio=1.,
            expand_ratio=1.,
            groups=1,
            block_fn=ResnextBottleneck,
            norm_type='batch', act_type='lrelu',
            **block_kwargs
    ):
        super(CrossStage, self).__init__()
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))

        if stride != 1:
            self.conv_down = Downsample(in_chs, in_chs, stride, groups)
        else:
            self.conv_down = nn.Identity()
        self.conv_exp = ConvBlock(in_chs, exp_chs, 1, norm_type, act_type)

        prev_chs = exp_chs // 2  # output of conv_exp is always split in two
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(prev_chs, block_out_chs))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition_b = ConvBlock(prev_chs, exp_chs // 2,  1, norm_type, act_type)
        self.conv_transition = ConvBlock(exp_chs, out_chs, 1, norm_type, act_type)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        xs, xb = x.split(self.expand_chs // 2, dim=1)
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb).contiguous()
        out = self.conv_transition(torch.cat([xs, xb], dim=1))
        return out


class CrossStage3(nn.Module):
    """Cross Stage 3.
    Similar to CrossStage, but with only one transition conv for the output.
    """
    def __init__(
            self,
            in_chs,
            out_chs,
            stride,
            depth,
            block_ratio=1.,
            expand_ratio=1.,
            groups=1,
            block_fn=ResnextBottleneck,
            norm_type='batch', act_type='lrelu',
            **block_kwargs
    ):
        super(CrossStage3, self).__init__()
        self.expand_chs = exp_chs = int(round(out_chs * expand_ratio))
        block_out_chs = int(round(out_chs * block_ratio))

        if stride != 1:
            self.conv_down = Downsample(in_chs, in_chs, stride, groups)
        else:
            self.conv_down = nn.Identity()
        # expansion conv
        self.conv_exp = ConvBlock(in_chs, exp_chs, 1, norm_type, act_type)

        prev_chs = exp_chs // 2  # expanded output is split in 2 for blocks and cross stage
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_module(str(i), block_fn(prev_chs, block_out_chs))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition = ConvBlock(exp_chs, out_chs, 1, norm_type, act_type)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.conv_exp(x)
        x1, x2 = x.split(self.expand_chs // 2, dim=1)
        x1 = self.blocks(x1)
        out = self.conv_transition(torch.cat([x1, x2], dim=1))
        return out

