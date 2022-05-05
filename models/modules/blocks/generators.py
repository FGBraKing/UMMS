import torch
import functools
import torch.nn as nn
from torch.nn import functional as F
from models.modules.blocks.blocks3d import same_convlution, downsample_convlution, upsample_deconvlution
from models.auxiliary_funs import get_normalization3d, get_activation


# torch.nn.functional.pad ( input, pad , mode='reflect' , value=(1, 1, 1, 1, 1, 1) )
class InConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, stride=1,
                 kernel_size=7, dilation=1, groups=1,
                 padding_mode='zeros', norm_type="none", act_type="leakyrelu", **norm_kwargs):
        # ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        if stride == 1:
            conv = same_convlution(in_planes, out_planes, kernel_size, dilation, groups, use_bias, padding_mode)
        else:
            conv = downsample_convlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type, **norm_kwargs)
        act = get_activation(act_type)
        super(InConv, self).__init__(conv, norm, act)


class EncoderBase(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=4,
                 padding_mode='reflect', norm_type="batch", act_type="leakyrelu"):
        # ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        bias = not (norm_type == 'batch' or norm_type == 'group')
        conv = downsample_convlution(in_planes, out_planes, kernel_size, use_bias=bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type)
        act = get_activation(act_type)
        super(EncoderBase, self).__init__(conv, norm, act)


class Encoder(nn.Sequential):
    def __init__(self, num_blocks, in_planes, kernel_size=4,
                 padding_mode='reflect', norm_type="batch", act_type="leakyrelu"):
        encoders = []
        for i in range(num_blocks):
            in_ch = in_planes*2**i if in_planes*2**i < 512 else 512
            out_ch = in_planes*2**i*2 if in_planes*2**i < 512 else 512
            encoders.append(EncoderBase(in_ch, out_ch, kernel_size, padding_mode, norm_type, act_type))

        super(Encoder, self).__init__(*encoders)


class DecoderBase(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=4,
                 padding_mode='reflect', norm_type="batch", act_type="leakyrelu"):
        bias = not (norm_type == 'batch' or norm_type == 'group')
        conv = upsample_deconvlution(in_planes, out_planes, kernel_size, use_bias=bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type)
        act = get_activation(act_type)
        super(DecoderBase, self).__init__(conv, norm, act)


# kernel_size = img_size // 2 ** n_strided


