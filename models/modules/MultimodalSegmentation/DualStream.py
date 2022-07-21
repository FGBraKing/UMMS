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


# total (1+3+4)=8, cbr+maxpool/brc+avgpool
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm_type="batch", act_type="lrelu", use_norm=True, use_act=True, **norm_kwargs):
        super(ConvBlock, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.conv = same_convlution(in_channels, out_channels, kernel_size, use_bias=use_bias)
        self.norm = get_normalization3d(out_channels, norm_type, **norm_kwargs)
        self.act = get_activation(act_type)
        self.use_norm = use_norm
        self.use_act = use_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, in_encoder, kernel_size=3, norm_type="batch", act_type="lrelu"):
        super(DoubleConv, self).__init__()
        if in_encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = max(in_channels, out_channels // 2)
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, kernel_size, norm_type, act_type)  # cbr
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, kernel_size, norm_type, act_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, norm_type="batch", act_type="lrelu"):
        super(DownBlock, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels, True, conv_kernel_size, norm_type, act_type)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels,
                 conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 mode='trilinear', upsample=True, interpolation=True, norm_type="batch", act_type="lrelu"):
        super(UpBlock, self).__init__()
        if upsample:
            if interpolation:
                self.upsampling = InterpolateUpsampling(mode=mode)
                self.joining = partial(self._joining, concat=True)
            else:
                # kernel_size改成4
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=4, scale_factor=scale_factor)
                self.joining = partial(self._joining, concat=True)
                in_channels = out_channels
        else:
            self.upsampling = NoUpsampling()
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels + feature_channels, out_channels,
                                         False, conv_kernel_size, norm_type, act_type)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class Inconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type="batch", act_type="lrelu"):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, True, kernel_size, norm_type, act_type)

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), upsampling=1,
                 with_activation=False, final_sigmoid=True):
        super(OutConv, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                padding=[a // 2 for a in kernel_size])
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='trilinear',
                                      align_corners=True) if upsampling > 1 else nn.Identity()
        if with_activation:
            if final_sigmoid:
                self.activation = nn.Sigmoid()
            else:
                self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv3d(x)
        x = self.upsampling(x)
        x = self.activation(x)
        return x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = DownBlock(in_channels, out_feature_num,
                                apply_pooling=False,
                                conv_kernel_size=conv_kernel_size,
                                pool_kernel_size=pool_kernel_size,
                                basic_module=basic_module,
                                norm_type=norm_type,
                                act_type=act_type)
        else:
            encoder = DownBlock(f_maps[i - 1], out_feature_num,
                                conv_kernel_size=conv_kernel_size,
                                pool_kernel_size=pool_kernel_size,
                                basic_module=basic_module,
                                norm_type=norm_type,
                                act_type=act_type)
        encoders.append(encoder)
    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i]
        cat_feature_num = reversed_f_maps[i + 1]
        out_feature_num = reversed_f_maps[i + 1]

        decoder = UpBlock(in_feature_num, cat_feature_num, out_feature_num, conv_kernel_size,
                          basic_module=basic_module, interpolation=interpolation,
                          norm_type=norm_type, act_type=act_type)

        decoders.append(decoder)
    return nn.ModuleList(decoders)


class DualStreamUnetV1(nn.Module):
    def __init__(self, in_channels, out_channels, domains, basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(DualStreamUnetV1, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        domains = [n for n in domains if isinstance(n, str)]
        self.num = len(domains)
        domain_encoders = defaultdict()
        for domain in domains:
            domain_encoders[domain] = create_encoders(in_channels, f_maps, basic_module,
                                                      conv_kernel_size, pool_kernel_size, norm_type, act_type)

        self.independent_encoders = nn.ModuleDict(domain_encoders)
        self.shared_decoders = create_decoders(f_maps, basic_module, conv_kernel_size,
                                               interpolation, norm_type, act_type)
        self.shared_outconv = OutConv(f_maps[0], out_channels,
                                      with_activation=with_activation, final_sigmoid=final_sigmoid)

    def forward(self, x, domain):
        encoders_features = []
        for encoder in self.independent_encoders[domain]:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.shared_decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.shared_outconv(x)

        return x


class DualStreamUnetV2(nn.Module):
    def __init__(self, in_channels, out_channels, domains, basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(DualStreamUnetV2, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        domains = [n for n in domains if isinstance(n, str)]
        self.num = len(domains)
        domain_inconv = defaultdict()
        for domain in domains:
            domain_inconv[domain] = Inconv(in_channels, f_maps[0], conv_kernel_size, norm_type, act_type)
        self.independent_inconv = nn.ModuleDict(domain_inconv)

        self.shared_encoders = nn.ModuleList(
            [
                DownBlock(f_maps[i], f_maps[i + 1], basic_module=basic_module,
                          conv_kernel_size=conv_kernel_size, pool_kernel_size=pool_kernel_size,
                          norm_type=norm_type, act_type=act_type) for i in range(num_levels - 1)
            ]
        )
        self.shared_decoders = create_decoders(f_maps, basic_module, conv_kernel_size,interpolation, norm_type, act_type)
        self.shared_outconv = OutConv(f_maps[0], out_channels,
                                      with_activation=with_activation, final_sigmoid=final_sigmoid)

    def forward(self, x, domain):
        encoders_features = []
        x = self.independent_inconv[domain](x)
        encoders_features.append(x)
        for encoder in self.shared_encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.shared_decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.shared_outconv(x)

        return x


class DualStreamUnetV3(nn.Module):
    def __init__(self, in_channels, out_channels, domains, basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(DualStreamUnetV3, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.shared_encoders = create_encoders(in_channels, f_maps, basic_module,
                                               conv_kernel_size, pool_kernel_size, norm_type, act_type)

        domains = [n for n in domains if isinstance(n, str)]
        self.num = len(domains)
        domain_decoders = defaultdict()
        domain_outconvs = defaultdict()
        for domain in domains:
            domain_decoders[domain] = create_decoders(f_maps, basic_module, conv_kernel_size,
                                                      interpolation, norm_type, act_type)
            domain_outconvs[domain] = OutConv(f_maps[0], out_channels,
                                              with_activation=with_activation, final_sigmoid=final_sigmoid)

        self.independent_decoders = nn.ModuleDict(domain_decoders)
        self.independent_outconv = nn.ModuleDict(domain_outconvs)

    def forward(self, x, domain):
        encoders_features = []
        for encoder in self.shared_encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.independent_decoders[domain], encoders_features):
            x = decoder(encoder_features, x)

        x = self.independent_outconv[domain](x)

        return x


class DualStreamUnetV4(nn.Module):
    def __init__(self, in_channels, out_channels, domains, basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(DualStreamUnetV4, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        domains = [n for n in domains if isinstance(n, str)]
        self.num = len(domains)
        domain_encoders = defaultdict()
        domain_decoders = defaultdict()
        domain_outconvs = defaultdict()
        for domain in domains:
            domain_encoders[domain] = create_encoders(in_channels, f_maps[:-1], basic_module,
                                                      conv_kernel_size, pool_kernel_size, norm_type, act_type)
            domain_decoders[domain] = create_decoders(f_maps, basic_module, conv_kernel_size,
                                                      interpolation, norm_type, act_type)
            domain_outconvs[domain] = OutConv(f_maps[0], out_channels,
                                              with_activation=with_activation, final_sigmoid=final_sigmoid)

        self.independent_encoders = nn.ModuleDict(domain_encoders)
        self.shared_middle_block = DownBlock(f_maps[-2], f_maps[-1], basic_module=basic_module,
                                             conv_kernel_size=conv_kernel_size, pool_kernel_size=pool_kernel_size,
                                             norm_type=norm_type, act_type=act_type)

        self.independent_decoders = nn.ModuleDict(domain_decoders)
        self.independent_outconv = nn.ModuleDict(domain_outconvs)

    def forward(self, x, domain):
        encoders_features = []
        for encoder in self.independent_encoders[domain]:
            x = encoder(x)
            encoders_features.insert(0, x)
        x = self.shared_middle_block(x)
        for decoder, encoder_features in zip(self.independent_decoders[domain], encoders_features):
            x = decoder(encoder_features, x)

        x = self.independent_outconv[domain](x)

        return x


class SingleUnet(nn.Module):
    def __init__(self, in_channels, out_channels, domains=None, basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(SingleUnet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type)
        self.outconv = OutConv(f_maps[0], out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)

    def forward(self, x, domain=None):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.outconv(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    from functools import partial
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor

    device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # net = DualStreamUnetV1(1, 1, ['target', 'source'],
    #                        f_maps=16, num_levels=5, with_activation=False, final_sigmoid=True,
    #                        norm_type="batch", act_type="lrelu").to(device)    # 5.8865M

    # net = DualStreamUnetV2(1, 1, ['target', 'source'],
    #                        f_maps=16, num_levels=5, with_activation=False, final_sigmoid=True,
    #                        norm_type="batch", act_type="lrelu").to(device)      # 4.1225M

    # net = DualStreamUnetV3(1, 1, ['target', 'source'],
    #                        f_maps=16, num_levels=5, with_activation=False, final_sigmoid=True,
    #                        norm_type="batch", act_type="lrelu").to(device)  # 6.4698M

    net = DualStreamUnetV4(1, 1, ['target', 'source'],
                           f_maps=16, num_levels=5, with_activation=False, final_sigmoid=True, interpolation=True,
                           norm_type="batch", act_type="lrelu").to(device)  # 6.9097M

    single_net = SingleUnet(1, 1, ['target', 'source'], DoubleConv, 16, 5,
                            with_activation=False, final_sigmoid=True,norm_type="batch", act_type="lrelu").to(device)

    print('--------------------------named_modules-------------------------------')
    for name, layer in net.named_modules():
        print(name, type(layer))
    print('------------------------named_children---------------------------------')
    for name, layer in net.named_children():
        print(name, type(layer))
    print('-------------------------named_parameters--------------------------------')
    for k, v in net.named_parameters():
        print(k, v.size())

        print(v.nelement())
    print('--------------------------named_buffers-------------------------------')
    for k, v in net.named_buffers():
        print(k, v.size())
    func_net = partial(net, domain='target')

    inputs = torch.rand((2, 1, 80, 112, 112), requires_grad=True).to(device)  # 64,96,96
    print_model_parm_nums(net)
    # print(net)
    out = net(inputs, 'source')
    print(out.size())
    print('--------------------------single model-------------------------------')
    single_out = single_net(inputs)
    print(single_out.size())
    print_model_parm_flops(single_net, inputs, need_idx=False)  # 751.84G
    # summary(single_net, input_size=(1, 80, 112, 112), batch_size=2, device='cuda')
    print(single_net)