import torch
import torch.nn as nn
from torch.nn import functional as F
import warnings
# bias = not ('g' in order or 'b' in order)


class Mish(nn.Module):
    '''
    x * torch.tanh(torch.nn.functional.softplus(x))
    '''
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (F.tanh(F.softplus(x)))
        return x


def make_divisible(v, divisor=2, min_value=0):
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v


# ===================Convlution==============
def conv3x3x3(in_planes, out_planes,
              kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
              use_bias=False, padding_mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups,
                     bias=use_bias, padding_mode=padding_mode)


def conv1x1x1(in_planes, out_planes,
              kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
              use_bias=False, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups,
                     bias=use_bias, padding_mode=padding_mode)


def same_convlution(in_planes, out_planes, kernel_size, dilation=1,
                    groups=1, use_bias=False, padding_mode='zeros'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 3
    if isinstance(dilation, int):
        dilation = (dilation,) * 3
    assert len(kernel_size) == len(dilation) == 3, '3d convlution'
    padding = tuple([(d*k-d)//2 for (d, k) in zip(dilation, kernel_size)])
    return nn.Conv3d(in_planes, out_planes, kernel_size, 1, padding, dilation, groups,
                     bias=use_bias, padding_mode=padding_mode)


def downsample_convlution(in_planes, out_planes, kernel_size, stride=2, dilation=1,
                          groups=1, use_bias=False, padding_mode='zeros'):
    # (4,2,1,1), (4,2,3k+1, 2k+1)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 3
    if isinstance(dilation, int):
        dilation = (dilation,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3
    assert len(kernel_size) == len(dilation) == len(stride) == 3, '3d convlution'

    padding = []
    for k, s, d in zip(kernel_size, stride, dilation):
        if s == 1:
            padding.append((d*k-d)//2)      # 总希望k是奇数
        elif s == 2:
            padding.append((d*k-d-1)//2)    # （d，k）是（奇数， 偶数）
        else:
            warnings.warn('you are using the stride which over two')
            padding.append((d*k-d+1-s)//2)  # （d，k，s）是（偶，偶/奇，奇）或（奇，奇，奇）或（奇，偶，偶）

    return nn.Conv3d(in_planes, out_planes, kernel_size, stride, tuple(padding), dilation, groups,
                     bias=use_bias, padding_mode=padding_mode)


def full_convlution(in_planes, out_planes, kernel_size, stride=1, dilation=1, groups=1,
                    use_bias=False, padding_mode='zeros'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 3
    assert len(kernel_size) == 3, '3d convlution'
    padding = tuple([k-1 for k in kernel_size])
    return nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups,
                     bias=use_bias, padding_mode=padding_mode)


def half_convlution(in_planes, out_planes, kernel_size, stride=1, dilation=1, groups=1,
                    use_bias=False, padding_mode='zeros'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 3
    assert len(kernel_size) == 3, '3d convlution'
    padding = tuple([k//2 for k in kernel_size])
    return nn.Conv3d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups,
                     bias=use_bias, padding_mode=padding_mode)


def valid_convlution(in_planes, out_planes, kernel_size, stride=1, dilation=1, groups=1,
                     use_bias=False, padding_mode='zeros'):
    return nn.Conv3d(in_planes, out_planes, kernel_size, stride, (0, 0, 0), dilation, groups,
                     bias=use_bias, padding_mode=padding_mode)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=False):
        if dilation != 1 and kernel_size==3:
            padding = dilation
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=use_bias)
        self.point_conv = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=use_bias)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        out = None
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


# ===================Convlution + Normolization + Activation==============

class CBR(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, use_bias=False):
        super(CBR, self).__init__()
        padding = dilation
        blocks = [
            conv3x3x3(in_planes, out_planes, padding=padding, dilation=dilation, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)


class CBL(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, use_bias=False, slope=1e-2):
        super(CBL, self).__init__()
        padding = dilation
        blocks = [
            conv3x3x3(in_planes, out_planes, padding=padding, dilation=dilation, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(negative_slope=slope, inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)


class CBS(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, use_bias=False):
        super(CBS, self).__init__()
        padding = dilation
        blocks = [
            conv3x3x3(in_planes, out_planes, padding=padding, dilation=dilation, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.SiLU(inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)


def create_conv_block(in_channels, out_channels, kernel_size, order, num_groups, padding):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    model = nn.Sequential()

    for i, char in enumerate(order):
        if char == 'r':
            model.add_module('ReLU', nn.ReLU(inplace=True))
        elif char == 'l':
            model.add_module('LeakyReLU', nn.LeakyReLU(inplace=True))
        elif char == 'e':
            model.add_module('ELU', nn.ELU(inplace=True))
        elif char == 's':
            model.add_module('Swish', nn.SiLU(inplace=True))
        elif char == 'm':
            model.add_module('mish', Mish())
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            model.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by ' \
                                                   f'num_groups. num_channels={num_channels}, num_groups={num_groups}'
            model.add_module('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                model.add_module('batchnorm', nn.BatchNorm3d(in_channels))
            else:
                model.add_module('batchnorm', nn.BatchNorm3d(out_channels))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                model.add_module('instancenorm', nn.InstanceNorm3d(in_channels))
            else:
                model.add_module('instancenorm', nn.InstanceNorm3d(out_channels))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'i',"
                             f" 'r', 'l', 'e', 's', 'm', 'c']")
        return model


class ConvBnRelu(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            use_bn: bool = True,
            add_relu: bool = True,
            interpolate: bool = False
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.use_bn = use_bn
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return x


# ===================Convlution Blocks==============
class ResBlock(nn.Module):
    def __init__(self, in_channels, conv_block):
        super(ResBlock, self).__init__()
        mid_channels = in_channels//2
        self.conv1 = conv_block(in_channels, mid_channels)
        self.conv2 = conv_block(in_channels, mid_channels)

    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x+out


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, norm_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        return x


# ===========================================Downsample=========================================================
class CBLDown(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1, use_bias=False, slope=1e-2):
        super(CBLDown, self).__init__()
        assert (dilation-1) % 2 == 0
        padding = (3*dilation-1)/2

        blocks = [
            conv3x3x3(in_planes, out_planes, 4, 2, padding, dilation, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(negative_slope=slope, inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)


# ===========================================Unsample=================================
def upsampleing_trilinear3d(scale):
    # size scale_factor mode align_corners
    return nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True) if scale > 1 else nn.Identity()


def upsampling_nearest3d(scale):
    # size scale_factor mode align_corners
    return nn.Upsample(scale_factor=scale, mode='nearest', align_corners=True) if scale > 1 else nn.Identity()


# (1,2,0,1) (2,2,1,2) (3,2,1,1) (4,2,1,0) (5,2,2,1)
def deconvlution(in_planes, out_planes, kernel_size=4, stride=2, padding=1, output_padding=0, use_bias=False):
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size, stride, padding, output_padding,
                              groups=1, bias=use_bias, dilation=1, padding_mode='zeros')


def upsample_deconvlution(in_planes, out_planes, kernel_size=4, stride=2, dilation=1, use_bias=False):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 3
    if isinstance(dilation, int):
        dilation = (dilation,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3
    assert len(kernel_size) == len(dilation) == len(stride) == 3, '3d convlution'
    out_padding = []
    in_padding = []

    for k, s, d in zip(kernel_size, stride, dilation):
        op = max(0, int(d*k-d+2-s) // 2 * 2) - (d*k-d+1-s)
        out_padding.append(op)
        in_padding.append((d*k-d+1-s+op)//2)

    # print(kernel_size[0], stride[0], in_padding[0], out_padding[0], dilation[0])
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size, stride, tuple(in_padding), tuple(out_padding),
                              groups=1, bias=use_bias, dilation=dilation, padding_mode='zeros')


class DCBR(nn.Module):
    def __init__(self, in_planes, out_planes, use_bias=False):
        super(DCBR, self).__init__()
        blocks = [
            deconvlution(in_planes, out_planes, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)


class DCBL(nn.Module):
    def __init__(self, in_planes, out_planes, use_bias=False, slope=1e-2):
        super(DCBL, self).__init__()
        blocks = [
            deconvlution(in_planes, out_planes, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(negative_slope=slope, inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)


class DCBS(nn.Module):
    def __init__(self, in_planes, out_planes, use_bias=False):
        super(DCBS, self).__init__()
        blocks = [
            deconvlution(in_planes, out_planes, use_bias=use_bias),
            nn.BatchNorm3d(out_planes),
            nn.SiLU(inplace=True)
        ]
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.model(inputs)



