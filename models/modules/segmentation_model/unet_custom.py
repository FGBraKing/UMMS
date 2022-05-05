import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


from models.modules.blocks.blocks3d import same_convlution, downsample_convlution, upsample_deconvlution


def conv_bn_lrelu_3d(in_planes, out_planes, kernel_size=3, dilation=1, use_bias=False, slope=2e-1):
    return nn.Sequential(
        same_convlution(in_planes, out_planes, kernel_size, dilation, use_bias=use_bias),
        nn.BatchNorm3d(out_planes, affine=True),
        nn.LeakyReLU(negative_slope=slope, inplace=True)
    )


def conv_in_lrelu_3d(in_planes, out_planes, kernel_size=3, dilation=1, use_bias=False, slope=2e-1):
    return nn.Sequential(
        same_convlution(in_planes, out_planes, kernel_size, dilation, use_bias=use_bias),
        nn.InstanceNorm3d(out_planes, affine=True),
        nn.LeakyReLU(negative_slope=slope, inplace=True)
    )


def deconv_bn_relu3d(in_channels, out_channels, kernel_size=4, stride=2, dilation=1, use_bias=False, slope=2e-1):
    return nn.Sequential(
        upsample_deconvlution(in_channels, out_channels, kernel_size, stride, dilation=dilation, bias=use_bias),
        nn.BatchNorm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=slope, inplace=True)
    )


def deconv_in_relu3d(in_channels, out_channels, kernel_size=4, stride=2, dilation=1, use_bias=False, slope=2e-1):
    return nn.Sequential(
        upsample_deconvlution(in_channels, out_channels, kernel_size, stride, dilation=dilation, bias=use_bias),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=slope, inplace=True)
    )


class NormalizationLayer(nn.Module):
    def __init__(self, norm_type='instance', **kwargs):
        super(NormalizationLayer, self).__init__()
        # like nn.ModuleDict
        if norm_type.lower() == 'instance':
            self.norm = nn.InstanceNorm3d(**kwargs)
        elif norm_type.lower() == 'batch':
            self.norm = nn.BatchNorm3d(**kwargs)
        elif norm_type.lower() == 'layer':
            self.norm = nn.LayerNorm(**kwargs)
        elif norm_type.lower() == 'group':
            self.norm = nn.GroupNorm(**kwargs)
        elif norm_type == 'none':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    def forward(self, x):
        return self.norm(x)


class ConvNormLrelu3d(nn.Module):
    def __init__(self, in_planes, out_planes, norm_type='batch',
                 kernel_size=3, dilation=1, use_bias=None, slope=2e-1):
        super(ConvNormLrelu3d, self).__init__()
        if use_bias is None:
            use_bias = not norm_type == 'batch'
        self.conv = same_convlution(in_planes, out_planes, kernel_size, dilation, use_bias=use_bias)
        self.norm = NormalizationLayer(norm_type, num_features=out_planes, affine=True)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DeconvNormLrelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type,
                 kernel_size=4, stride=2, dilation=1, use_bias=None, slope=2e-1):
        super(DeconvNormLrelu3d, self).__init__()
        if use_bias is None:
            use_bias = not norm_type == 'batch'
        self.deconv = upsample_deconvlution(in_channels, out_channels, kernel_size, stride, dilation=dilation, use_bias=use_bias)
        self.norm = NormalizationLayer(norm_type, num_features=out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, kernel_size=3,
                 dilation=1, use_bias=None, slope=2e-1):
        super(DoubleConv, self).__init__()
        if use_bias is None:
            use_bias = not norm_type == 'batch'

        conv1_in_channels = in_channels
        conv2_in_channels = conv1_out_channels = out_channels//2 if in_channels < out_channels else out_channels
        conv2_out_channels = out_channels

        self.conv1 = ConvNormLrelu3d(conv1_in_channels, conv1_out_channels, norm_type,
                                     kernel_size, dilation, use_bias, slope)
        self.conv2 = ConvNormLrelu3d(conv2_in_channels, conv2_out_channels, norm_type,
                                     kernel_size, dilation, use_bias, slope)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InConv(nn.Module):
    def __init__(self, norm_type, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = DoubleConv(norm_type, in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)


class EncodeBlock(nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, conv_kernel_size=3,
                 is_max_pool=True, max_pool_kernel_size=(2, 2, 2)):
        super(EncodeBlock, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, padding=0) if is_max_pool else None
        self.double_conv = DoubleConv(norm_type, in_channels, out_channels, conv_kernel_size)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, norm_type, in_channels, out_channels, conv_kernel_size=3,
                 deconv_kernel_size=(4, 4, 4), scale_factor=(2, 2, 2),):
        super(DecodeBlock, self).__init__()

        self.upsample = DeconvNormLrelu3d(in_channels, out_channels, norm_type,
                                          kernel_size=deconv_kernel_size, stride=scale_factor)
        self.double_conv = DoubleConv(norm_type, in_channels, out_channels, conv_kernel_size)

    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), activation=False, upsampling=1):
        super(OutConv, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=[a//2 for a in kernel_size])
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.conv3d(x)
        x = self.upsampling(x)
        if self.activation:
            x = torch.sigmoid(x)
        return x


class UnetCustom(nn.Module):
    def __init__(self, norm_type, in_channels, n_class,
                 deptp=4, init_channel_number=64, final_sigmoid=False):
        super(UnetCustom, self).__init__()
        f_maps = [init_channel_number*2**x for x in range(deptp + 1)]
        self.in_conv = InConv(norm_type, in_channels, f_maps[0])

        self.encoders = nn.ModuleList([EncodeBlock(norm_type, f_maps[i], f_maps[i+1],
                                                   conv_kernel_size=(3, 3, 3),
                                                   max_pool_kernel_size=(2,2,2))
                                       for i in range(deptp)])

        # self.encoders = nn.ModuleList([
        #     EncodeBlock(norm_type, f_maps[0], f_maps[1], conv_kernel_size=(3, 3, 3), max_pool_kernel_size=(2,2,2)),
        #     EncodeBlock(norm_type, f_maps[1], f_maps[2], conv_kernel_size=(3, 3, 3), max_pool_kernel_size=(2,2,2)),
        #     EncodeBlock(norm_type, f_maps[2], f_maps[3], conv_kernel_size=(3, 3, 3), max_pool_kernel_size=(2,2,2)),
        #     EncodeBlock(norm_type, f_maps[3], f_maps[4], conv_kernel_size=(3, 3, 3), max_pool_kernel_size=(2,2,2)),
        # ])

        self.decoders = nn.ModuleList(
            [
                DecodeBlock(norm_type, f_maps[deptp - i], f_maps[deptp - 1 - i],
                            conv_kernel_size=(3, 3, 3),
                            deconv_kernel_size=(4, 4, 4),
                            scale_factor=(2, 2, 2)) for i in range(deptp)
            ]

        )
        # self.decoders = nn.ModuleList([
        #     DecodeBlock(norm_type, f_maps[4], f_maps[3],
        #                 conv_kernel_size=(3, 3, 3), deconv_kernel_size=(4,4,4), scale_factor=(2,2,2)),
        #     DecodeBlock(norm_type, f_maps[3], f_maps[2],
        #                 conv_kernel_size=(3, 3, 3), deconv_kernel_size=(4,4,4), scale_factor=(2,2,2)),
        #     DecodeBlock(norm_type, f_maps[2], f_maps[1],
        #                 conv_kernel_size=(3, 3, 3), deconv_kernel_size=(4,4,4), scale_factor=(2,2,2)),
        #     DecodeBlock(norm_type, f_maps[1], f_maps[0],
        #                 conv_kernel_size=(3, 3, 3), deconv_kernel_size=(4,4,4), scale_factor=(2,2,2)),
        # ])
        self.out_conv = OutConv(f_maps[0], n_class, kernel_size=(1, 1, 1), activation=final_sigmoid)

    def forward(self, x):
        encoders_features = []
        x = self.in_conv(x)
        encoders_features.append(x)
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)
        encoders_features = encoders_features[::-1]
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.out_conv(x)
        return x

    def get_L2_norm(self):
        l2_norm = torch.sum(torch.Tensor([torch.sum(torch.pow(parameter, 2))/2
                                          for parameter in self.parameters() if parameter.requires_grad]))
        return l2_norm


if __name__ == "__main__":
    from torchsummary import summary
    from functools import partial
    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    torch.cuda.set_device('cuda:1')
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = UnetCustom(norm_type='batch', in_channels=1, n_class=1, deptp=4,
                     init_channel_number=16, final_sigmoid=True).cuda()

    # print('---------------------------------------------------------')
    # for name, layer in net.named_modules():
    #     print(name, type(layer))
    print('---------------------------------------------------------')
    for name, layer in net.named_children():
        print(name, type(layer))
    #
    # for k, v in net.named_parameters():
    #     print(k, v.size())
    #
    #     print(v.nelement())
    #
    # for k, v in net.named_buffers():
    #     print(k, v.size())

    # func_net = partial(net, domain='target')

    summary(net, input_size=(1, 80, 96, 96), batch_size=1, device='cuda')

    inputs = torch.rand((4, 1, 80, 96, 96), requires_grad=True).cuda()
    print_model_parm_nums(net)  # 40.15M

    out = net(inputs)
    print(out.size())

    print(net.get_L2_norm())
    # print_model_parm_flops(func_net, inputs, need_idx=False)  # 751.84G
    # summary(func_net, input_size=(1, 112, 128, 128), batch_size=1, device='cuda')



