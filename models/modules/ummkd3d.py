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


class ConvNormDictLrelu3d(nn.Module):
    def __init__(self, in_planes, out_planes, domains, norm_type='batch',
                 kernel_size=3, dilation=1, use_bias=None, slope=2e-1):
        super(ConvNormDictLrelu3d, self).__init__()
        if use_bias is None:
            use_bias = not norm_type == 'batch'
        self.conv = same_convlution(in_planes, out_planes, kernel_size, dilation, use_bias=use_bias)
        self.norm = NormalizationDict(domains, norm_type, num_features=out_planes, affine=True)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x, domain):
        x = self.conv(x)
        x = self.norm(x, domain)
        x = self.act(x)
        return x


class DeconvNormDictLrelu3d(nn.Module):
    def __init__(self, in_channels, out_channels, domains, norm_type,
                 kernel_size=4, stride=2, dilation=1, use_bias=None, slope=2e-1):
        super(DeconvNormDictLrelu3d, self).__init__()
        if use_bias is None:
            use_bias = not norm_type == 'batch'
        self.deconv = upsample_deconvlution(in_channels, out_channels, kernel_size, stride, dilation=dilation, use_bias=use_bias)
        self.norm = NormalizationDict(domains, norm_type, num_features=out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x, domain):
        x = self.deconv(x)
        x = self.norm(x, domain)
        x = self.act(x)
        return x


class DoubleConvWithNormDict(nn.Module):
    def __init__(self, domains, norm_type, in_channels, out_channels, kernel_size=3,
                 dilation=1, use_bias=None, slope=2e-1):
        super(DoubleConvWithNormDict, self).__init__()
        if use_bias is None:
            use_bias = not norm_type == 'batch'

        conv1_in_channels = in_channels
        conv2_in_channels = conv1_out_channels = out_channels//2 if in_channels < out_channels else out_channels
        conv2_out_channels = out_channels

        self.conv1 = ConvNormDictLrelu3d(conv1_in_channels, conv1_out_channels, domains, norm_type,
                                         kernel_size, dilation, use_bias, slope)
        self.conv2 = ConvNormDictLrelu3d(conv2_in_channels, conv2_out_channels, domains, norm_type,
                                         kernel_size, dilation, use_bias, slope)

    def forward(self, x, domain):
        x = self.conv1(x, domain)
        x = self.conv2(x, domain)
        return x


class InConv(nn.Module):
    def __init__(self, domains, norm_type, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = DoubleConvWithNormDict(domains, norm_type, in_channels, out_channels)

    def forward(self, x, domain):
        return self.conv(x, domain)


# class InConv(nn.Module):
#     def __init__(self, domains, norm_type, in_channels, out_channels, use_bias=None, slope=2e-1):
#         super(InConv, self).__init__()
#         if use_bias is None:
#             use_bias = not norm_type == 'batch'
#
#         conv1_in_channels = in_channels
#         conv2_in_channels = conv1_out_channels = out_channels//2 if in_channels < out_channels else out_channels
#         conv2_out_channels = out_channels
#
#         self.conv1 = nn.Conv3d(conv1_in_channels, conv1_out_channels, (3,3,3), (1,1,1), (1,3,3), bias=use_bias)
#         self.norm1 = NormalizationDict(domains, norm_type, num_features=conv1_out_channels, affine=True)
#
#         self.conv2 = nn.Conv3d(conv2_in_channels, conv2_out_channels, (3,3,3), (1,1,1), (0,3,3), bias=use_bias)
#         self.norm2 = NormalizationDict(domains, norm_type, num_features=conv2_out_channels, affine=True)
#
#         self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
#
#     def forward(self, x, domain):
#         x = self.conv1(x)
#         x = self.norm1(x, domain)
#         x = self.act(x)
#
#         x = self.conv2(x)
#         x = self.norm2(x, domain)
#         x = self.act(x)
#         return x


class EncodeBlockWithNormDict(nn.Module):
    def __init__(self, domains, norm_type, in_channels, out_channels, conv_kernel_size=3,
                 is_max_pool=True, max_pool_kernel_size=(2, 2, 2)):
        super(EncodeBlockWithNormDict, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=max_pool_kernel_size, padding=0) if is_max_pool else None
        self.double_conv = DoubleConvWithNormDict(domains, norm_type, in_channels, out_channels, conv_kernel_size)

    def forward(self, x, domain):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x, domain)
        return x


class DecodeBlockWithNormDict(nn.Module):
    def __init__(self, domains, norm_type, in_channels, out_channels, conv_kernel_size=3,
                 deconv_kernel_size=(4, 4, 4), scale_factor=(2, 2, 2),):
        super(DecodeBlockWithNormDict, self).__init__()

        self.upsample = DeconvNormDictLrelu3d(in_channels, out_channels, domains, norm_type,
                                              kernel_size=deconv_kernel_size, stride=scale_factor)
        self.double_conv = DoubleConvWithNormDict(domains, norm_type, in_channels, out_channels, conv_kernel_size)

    def forward(self, encoder_features, x, domain):
        x = self.upsample(x, domain)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x, domain)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), activation=False, upsampling=1):
        super(OutConv, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=[a//2 for a in kernel_size])
        # self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.conv3d(x)
        x = self.upsampling(x)
        if self.activation:
            x = torch.sigmoid(x)
        return x


class UnetWithNormSpecficity(nn.Module):
    def __init__(self, domains, norm_type, in_channels, n_class,
                 deptp=4, init_channel_number=64, final_sigmoid=False):
        super(UnetWithNormSpecficity, self).__init__()
        f_maps = [init_channel_number*2**x for x in range(deptp + 1)]
        self.in_conv = InConv(domains, norm_type, in_channels, f_maps[0])
        self.encoders = nn.ModuleList([
            EncodeBlockWithNormDict(domains, norm_type, f_maps[0], f_maps[1],
                                    conv_kernel_size=(3, 3, 3), max_pool_kernel_size=(2,2,2)),
            EncodeBlockWithNormDict(domains, norm_type, f_maps[1], f_maps[2],
                                    conv_kernel_size=(3, 3, 3), max_pool_kernel_size=(2,2,2)),
            EncodeBlockWithNormDict(domains, norm_type, f_maps[2], f_maps[3],
                                    conv_kernel_size=(1, 3, 3), max_pool_kernel_size=(1,2,2)),
            EncodeBlockWithNormDict(domains, norm_type, f_maps[3], f_maps[4],
                                    conv_kernel_size=(1, 3, 3), max_pool_kernel_size=(1,2,2)),
        ])
        self.decoders = nn.ModuleList([
            DecodeBlockWithNormDict(domains, norm_type, f_maps[4], f_maps[3],
                                    conv_kernel_size=(1, 3, 3), deconv_kernel_size=(3,4,4), scale_factor=(1,2,2)),
            DecodeBlockWithNormDict(domains, norm_type, f_maps[3], f_maps[2],
                                    conv_kernel_size=(1, 3, 3), deconv_kernel_size=(3,4,4), scale_factor=(1,2,2)),
            DecodeBlockWithNormDict(domains, norm_type, f_maps[2], f_maps[1],
                                    conv_kernel_size=(3, 3, 3), deconv_kernel_size=(4,4,4), scale_factor=(2,2,2)),
            DecodeBlockWithNormDict(domains, norm_type, f_maps[1], f_maps[0],
                                    conv_kernel_size=(3, 3, 3), deconv_kernel_size=(4,4,4), scale_factor=(2,2,2)),
        ])
        self.out_conv = OutConv(f_maps[0], n_class, kernel_size=(1,1,1), activation=final_sigmoid)

    def forward(self, x, domain):
        encoders_features = []
        x = self.in_conv(x, domain)
        encoders_features.append(x)
        for encoder in self.encoders:
            x = encoder(x, domain)
            encoders_features.append(x)
        encoders_features = encoders_features[::-1]
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x, domain)

        x = self.out_conv(x)
        return x

    def get_L2_norm(self):
        L2_norm = torch.sum(torch.Tensor([torch.sum(torch.pow(parameter, 2))/2
                                          for parameter in self.parameters() if parameter.requires_grad]))
        return L2_norm


if __name__ == "__main__":
    from torchsummary import summary
    from functools import partial
    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
    net = UnetWithNormSpecficity(domains=['target', 'source'], norm_type='batch',
                                 in_channels=1, n_class=1, init_channel_number=16, final_sigmoid=True).to(device)

    # print('---------------------------------------------------------')
    # for name, layer in net.named_modules():
    #     print(name, type(layer))
    # print('---------------------------------------------------------')
    # for name, layer in net.named_children():
    #     print(name, type(layer))
    #
    # for k, v in net.named_parameters():
    #     print(k, v.size())
    #
    #     print(v.nelement())
    #
    # for k, v in net.named_buffers():
    #     print(k, v.size())

    func_net = partial(net, domain='target')

    inputs = torch.rand((4, 1, 80, 96, 96), requires_grad=True).to(device)  # 64,96,96
    print_model_parm_nums(net)  # 40.15M

    out = net(inputs, 'target')
    print(out.size())

    print(net.get_L2_norm())
    # print_model_parm_flops(func_net, inputs, need_idx=False)  # 751.84G
    # summary(func_net, input_size=(1, 64, 96, 96), batch_size=1, device='cuda')




