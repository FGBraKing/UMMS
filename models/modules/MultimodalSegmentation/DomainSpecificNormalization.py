import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict, defaultdict
from models.modules.blocks.blocks3d import create_conv_block, same_convlution
from models.auxiliary_funs import get_normalization3d, get_activation
from torch.nn import functional as F


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest', align_corners=True):
        '''
        :param mode: 'nearest' | 'linear'`| 'bilinear' | 'bicubic'`|'trilinear' | 'area'. Default: ``'nearest'``
        '''
        upsample = partial(self._interpolate, mode=mode, align_corners=align_corners)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode, align_corners):
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=4, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class NormalizationDict(nn.Module):
    def __init__(self, domains, norm_type='instance', **kwargs):
        super(NormalizationDict, self).__init__()
        domains = [n for n in domains if isinstance(n, str)]
        self.num = len(domains)
        # like nn.ModuleDict
        self._norm_dict = OrderedDict()
        for domain in domains:
            if norm_type.lower() == 'instance':
                self._norm_dict[domain] = nn.InstanceNorm3d(**kwargs)
            elif norm_type.lower() == 'batch':
                self._norm_dict[domain] = nn.BatchNorm3d(**kwargs)
            elif norm_type.lower() == 'layer':
                self._norm_dict[domain] = nn.LayerNorm(**kwargs)
            elif norm_type.lower() == 'group':
                self._norm_dict[domain] = nn.GroupNorm(**kwargs)
            else:
                self._norm_dict[domain] = nn.BatchNorm3d(**kwargs)
        self.norm = nn.ModuleDict(self._norm_dict)

    def forward(self, x, domain):
        return self.norm[domain](x)















