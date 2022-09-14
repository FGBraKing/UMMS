import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict, defaultdict
from models.auxiliary_funs import get_normalization3d, get_activation
from models.modules.blocks.blocks3d import conv3x3x3, conv1x1x1, same_convlution, create_conv_block, downsample_convlution
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
    def __init__(self, num_features, domains, norm_type='batch', **kwargs):
        super(NormalizationDict, self).__init__()
        domains = [n for n in domains if isinstance(n, str)]
        self.num = len(domains)
        self._norm_dict = OrderedDict()
        for domain in domains:
            if norm_type.lower() == 'instance' or norm_type.lower() == "in":
                self._norm_dict[domain] = nn.InstanceNorm3d(num_features, **kwargs)
            elif norm_type.lower() == 'batch' or norm_type.lower() == "bn":
                self._norm_dict[domain] = nn.BatchNorm3d(num_features, **kwargs)
            elif norm_type.lower() == 'layer' or norm_type.lower() == "ln":
                self._norm_dict[domain] = nn.LayerNorm(**kwargs)
            elif norm_type.lower() == 'group' or norm_type.lower() == "gn":
                self._norm_dict[domain] = nn.GroupNorm(num_channels=num_features, **kwargs)
            else:
                self._norm_dict[domain] = nn.Identity()
        self.norm = nn.ModuleDict(self._norm_dict)

    def forward(self, x, domain):
        return self.norm[domain](x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, domains, kernel_size=3,
                 norm_type="batch", act_type="lrelu", use_norm=True, use_act=True, **norm_kwargs):
        super(ConvBlock, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.conv = same_convlution(in_channels, out_channels, kernel_size, use_bias=use_bias)
        self.norm = NormalizationDict(out_channels, domains, norm_type, **norm_kwargs)
        self.act = get_activation(act_type)
        self.use_norm = use_norm
        self.use_act = use_act

    def forward(self, x, domain):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x, domain)
        if self.use_act:
            x = self.act(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, domains, in_encoder, kernel_size=3, norm_type="batch", act_type="lrelu"):
        super(DoubleConv, self).__init__()
        if in_encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = max(in_channels, out_channels // 2)
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, domains, kernel_size, norm_type, act_type)  # cbr
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, domains, kernel_size, norm_type, act_type)

    def forward(self, x, domain):
        x = self.conv1(x, domain)
        x = self.conv2(x, domain)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, domains, conv_kernel_size=3, apply_pooling=True,
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

        self.basic_module = basic_module(in_channels, out_channels, domains, True, conv_kernel_size, norm_type, act_type)

    def forward(self, x, domain):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x, domain)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels, domains,
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

        self.basic_module = basic_module(in_channels + feature_channels, out_channels, domains,
                                         False, conv_kernel_size, norm_type, act_type)

    def forward(self, encoder_features, x, domain):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x, domain)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class RegressionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=4, apply_pooling=False,
                 pool_kernel_size=2, pool_type='max', norm_type="batch", act_type="lrelu"):
        super(RegressionBlock, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        conv_stride_size = 1 if apply_pooling else pool_kernel_size
        self.conv = downsample_convlution(in_channels, out_channels, conv_kernel_size, conv_stride_size, use_bias=use_bias)
        self.norm = get_normalization3d(out_channels, norm_type)
        self.act = get_activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.pooling:
            x = self.pooling(x)
        return x


class FunetuneBlockA(nn.Module):
    def __init__(self, features_channels, addition_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), mode='trilinear', upsample=True, interpolation=True,
                 norm_type="batch", act_type="lrelu"):
        super(FunetuneBlockA, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.conv = same_convlution(features_channels+addition_channels, out_channels, kernel_size, use_bias=use_bias)
        self.norm = get_normalization3d(out_channels, norm_type)
        self.act = get_activation(act_type)
        if upsample:
            if interpolation:
                self.upsampling = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
            else:
                self.upsampling = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=4, stride=scale_factor, padding=1)
        else:
            self.upsampling = nn.Identity()

    def forward(self, orgin_features, additive_feature):
        x = torch.cat((orgin_features, additive_feature), dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.upsampling(x)
        return x


class FunetuneBlockB(nn.Module):
    def __init__(self, features_channels, addition_channels, out_channels,
                 kernel_size=3, norm_type="batch", act_type="lrelu"):
        super(FunetuneBlockB, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.conv = same_convlution(features_channels + addition_channels, out_channels, kernel_size, use_bias=use_bias)
        self.norm = get_normalization3d(out_channels, norm_type)
        self.act = get_activation(act_type)

    def forward(self, orgin_features, additive_feature):
        x = torch.cat((orgin_features, additive_feature), dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Inconv(nn.Module):
    def __init__(self, in_channels, out_channels, domains, kernel_size=3, norm_type="batch", act_type="lrelu"):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, domains, True, kernel_size, norm_type, act_type)

    def forward(self, x, domain):
        return self.conv(x, domain)


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


class OutRegression(nn.Module):
    def __init__(self, in_channels, out_dim, with_activation=False, pool_type='avg'):
        super(OutRegression, self).__init__()
        assert pool_type in ['max', 'avg']
        if pool_type == 'max':
            self.adapooling = nn.AdaptiveMaxPool3d(3)
        else:
            self.adapooling = nn.AdaptiveAvgPool3d(3)
        self.liner = nn.Linear(in_channels*27, out_dim)
        if with_activation:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        bs = x.size(0)
        x = self.adapooling(x)
        x = self.liner(x.view(bs, -1))
        x = self.activation(x)
        return x


def create_encoders(domains, in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = DownBlock(in_channels, out_feature_num, domains,
                                apply_pooling=False,
                                conv_kernel_size=conv_kernel_size,
                                pool_kernel_size=pool_kernel_size,
                                basic_module=basic_module,
                                norm_type=norm_type,
                                act_type=act_type)
        else:
            encoder = DownBlock(f_maps[i - 1], out_feature_num, domains,
                                conv_kernel_size=conv_kernel_size,
                                pool_kernel_size=pool_kernel_size,
                                basic_module=basic_module,
                                norm_type=norm_type,
                                act_type=act_type)
        encoders.append(encoder)
    return nn.ModuleList(encoders)


def create_decoders(domains, f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i]
        cat_feature_num = reversed_f_maps[i + 1]
        out_feature_num = reversed_f_maps[i + 1]

        decoder = UpBlock(in_feature_num, cat_feature_num, out_feature_num, domains, conv_kernel_size,
                          basic_module=basic_module, interpolation=interpolation,
                          norm_type=norm_type, act_type=act_type)

        decoders.append(decoder)
    return nn.ModuleList(decoders)


def create_sizedecoders(f_maps, conv_kernel_size=4, apply_pooling=False, norm_type="batch", act_type="lrelu"):
    regressors = []
    forward_f_maps = list(f_maps)
    for i in range(len(forward_f_maps)-1):
        in_feature_num = 2 * forward_f_maps[i] if i == 0 else 3*forward_f_maps[i]
        out_feature_num = forward_f_maps[i + 1]
        sizedecoder = RegressionBlock(in_feature_num, out_feature_num, conv_kernel_size, apply_pooling,
                                      norm_type=norm_type, act_type=act_type)
        regressors.append(sizedecoder)
    return nn.ModuleList(regressors)


def create_funetuners_v1(f_maps, out_maps, conv_kernel_size, interpolation, norm_type, act_type):
    funetuners = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i]
        cat_feature_num = reversed_f_maps[i]
        # out_feature_num = reversed_f_maps[i + 1]
        out_feature_num = out_maps
        funetuner = FunetuneBlockA(in_feature_num, cat_feature_num, out_feature_num, conv_kernel_size,
                                   interpolation=interpolation, norm_type=norm_type, act_type=act_type)
        funetuners.append(funetuner)
    return nn.ModuleList(funetuners)


def create_funetuners_v2(f_maps, out_maps, conv_kernel_size, interpolation, norm_type, act_type):
    funetuners_v1 = []
    funetuners_v2 = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i]
        cat_feature_num = reversed_f_maps[i]
        out_feature_num = reversed_f_maps[i + 1]
        funetunerv1 = FunetuneBlockA(in_feature_num, cat_feature_num, out_feature_num, conv_kernel_size,
                                     interpolation=interpolation, norm_type=norm_type, act_type=act_type)
        funetuners_v1.append(funetunerv1)

        funetunerv2 = FunetuneBlockB(out_feature_num, reversed_f_maps[i + 1], out_maps, conv_kernel_size,
                                     norm_type=norm_type, act_type=act_type)
        funetuners_v2.append(funetunerv2)

    return nn.ModuleList(funetuners_v1), nn.ModuleList(funetuners_v2)


class ChilopodUnetWithRegression(nn.Module):
    def __init__(self, in_channels, out_channels, domains=('source', 'target'),
                 basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(ChilopodUnetWithRegression, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.encoders = create_encoders(domains, in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)
        self.decoders = create_decoders(domains, f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type)
        self.regressors = create_sizedecoders(f_maps, 4, False, norm_type, act_type)
        self.outconv = OutConv(f_maps[0], out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)
        self.outregress = OutRegression(f_maps[-1]*3, 1, True)

    def forward(self, source, target, domains=('source', 'target')):
        source_encoders_features = []
        target_encoders_features = []
        for encoder in self.encoders:
            source = encoder(source, domains[0])
            source_encoders_features.insert(0, source)
            target = encoder(target, domains[1])
            target_encoders_features.insert(0, target)

        source_encoders_features = source_encoders_features[1:]
        target_encoders_features = target_encoders_features[1:]
        source_decoders_features = []
        target_decoders_features = []
        for decoder, source_encoder_features, target_encoder_features in zip(self.decoders, source_encoders_features, target_encoders_features):
            source_decoders_features.insert(0, source)
            target_decoders_features.insert(0, target)
            source = decoder(source_encoder_features, source, domains[0])
            target = decoder(target_encoder_features, target, domains[1])

        fused_feature = torch.cat([source, target], dim=1)
        for regressor, source_decoder_features, target_decoder_features in zip(self.regressors, source_decoders_features, target_decoders_features):
            fused_feature = regressor(fused_feature)
            fused_feature = torch.cat((fused_feature, source_decoder_features, target_decoder_features), dim=1)

        roi_rate = self.outregress(fused_feature)

        source = self.outconv(source)
        target = self.outconv(target)
        return source, target, roi_rate


class ChilopodUnetWithRegressionFinetuneV1(nn.Module):
    def __init__(self, in_channels, out_channels, domains=('source', 'target'),
                 basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(ChilopodUnetWithRegressionFinetuneV1, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.encoders = create_encoders(domains, in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)
        self.decoders = create_decoders(domains, f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type)
        self.regressors = create_sizedecoders(f_maps, 4, False, norm_type, act_type)
        self.finetuners = create_funetuners_v1(f_maps, f_maps[0], 3, interpolation, norm_type, act_type)
        self.outconv = OutConv(f_maps[0], out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)
        self.outregress = OutRegression(f_maps[-1]*3, 1, True)

        self.feature_upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='trilinear', align_corners=True) for i in range(num_levels-2, -1, -1)
        ])
        self.finally_out = OutConv(f_maps[0]*num_levels, out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)

    def forward(self, source, target, domains=('source', 'target')):
        source_encoders_features = []
        target_encoders_features = []
        for encoder in self.encoders:
            source = encoder(source, domains[0])
            source_encoders_features.insert(0, source)
            target = encoder(target, domains[1])
            target_encoders_features.insert(0, target)

        source_encoders_features = source_encoders_features[1:]
        target_encoders_features = target_encoders_features[1:]
        source_decoders_features = []
        target_decoders_features = []
        for decoder, source_encoder_features, target_encoder_features in zip(self.decoders, source_encoders_features, target_encoders_features):
            source_decoders_features.insert(0, source)
            target_decoders_features.insert(0, target)
            source = decoder(source_encoder_features, source, domains[0])
            target = decoder(target_encoder_features, target, domains[1])
        source_predict1 = self.outconv(source)
        target_predict1 = self.outconv(target)

        regressors_features = []
        fused_feature = torch.cat([source, target], dim=1)
        for regressor, source_decoder_features, target_decoder_features in zip(self.regressors, source_decoders_features, target_decoders_features):
            fused_feature = regressor(fused_feature)
            regressors_features.insert(0, fused_feature)
            fused_feature = torch.cat((fused_feature, source_decoder_features, target_decoder_features), dim=1)
        roi_rate = self.outregress(fused_feature)

        source_refines_features = []
        target_refines_features = []
        for finetuner, regressor_features, source_decoder_features, target_decoder_features, upsample in zip(self.finetuners, regressors_features, source_decoders_features[::-1], target_decoders_features[::-1], self.feature_upsample):
            source_refine_features = finetuner(source_decoder_features, regressor_features)
            target_refine_features = finetuner(target_decoder_features, regressor_features)
            source_refines_features.insert(0, upsample(source_refine_features))
            target_refines_features.insert(0, upsample(target_refine_features))

        source_predict2 = self.finally_out(torch.cat(source_refines_features+[source], dim=1))
        target_predict2 = self.finally_out(torch.cat(target_refines_features+[target], dim=1))

        return source_predict1, target_predict1, roi_rate, source_predict2, target_predict2


class ChilopodUnetWithRegressionFinetuneV2(nn.Module):
    def __init__(self, in_channels, out_channels, domains=('source', 'target'),
                 basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(ChilopodUnetWithRegressionFinetuneV2, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.encoders = create_encoders(domains, in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)
        self.decoders = create_decoders(domains, f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type)
        self.regressors = create_sizedecoders(f_maps, 4, False, norm_type, act_type)
        self.finetuners_v1, self.finetuners_v2 = create_funetuners_v2(f_maps, f_maps[0], 3, interpolation, norm_type, act_type)

        self.outconv = OutConv(f_maps[0], out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)
        self.outregress = OutRegression(f_maps[-1]*3, 1, True)

        self.feature_upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='trilinear', align_corners=True) for i in range(num_levels-2, -1, -1)
        ])
        self.finally_out = OutConv(f_maps[0]*4, out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)

    def forward(self, source, target, domains=('source', 'target')):
        source_encoders_features = []
        target_encoders_features = []
        for encoder in self.encoders:
            source = encoder(source, domains[0])
            source_encoders_features.insert(0, source)
            target = encoder(target, domains[1])
            target_encoders_features.insert(0, target)

        source_encoders_features = source_encoders_features[1:]
        target_encoders_features = target_encoders_features[1:]
        source_decoders_features = []
        target_decoders_features = []
        for decoder, source_encoder_features, target_encoder_features in zip(self.decoders, source_encoders_features, target_encoders_features):
            source_decoders_features.insert(0, source)
            target_decoders_features.insert(0, target)
            source = decoder(source_encoder_features, source, domains[0])
            target = decoder(target_encoder_features, target, domains[1])
        source_predict1 = self.outconv(source)
        target_predict1 = self.outconv(target)

        regressors_features = []
        fused_feature = torch.cat([source, target], dim=1)
        for regressor, source_decoder_features, target_decoder_features in zip(self.regressors, source_decoders_features, target_decoders_features):
            fused_feature = regressor(fused_feature)
            regressors_features.insert(0, fused_feature)
            fused_feature = torch.cat((fused_feature, source_decoder_features, target_decoder_features), dim=1)
        roi_rate = self.outregress(fused_feature)

        source_refines_features = []
        target_refines_features = []
        for finetuner, regressor_features, source_decoder_features, target_decoder_features in zip(self.finetuners_v1, regressors_features, source_decoders_features[::-1], target_decoders_features[::-1]):
            source_refine_features = finetuner(source_decoder_features, regressor_features)
            target_refine_features = finetuner(target_decoder_features, regressor_features)
            source_refines_features.append(source_refine_features)
            target_refines_features.append(target_refine_features)

        source_refined_features = []
        target_refined_features = []
        source_decoders_features_torefine = source_decoders_features[:-1][::-1] + [source]
        target_decoders_features_torefine = target_decoders_features[:-1][::-1] + [target]
        for finetuner, source_refine_features, source_decoder_features_torefine, target_refine_features, target_decoder_features_torefine in zip(self.finetuners_v2, source_refines_features, source_decoders_features_torefine, target_refines_features, target_decoders_features_torefine):
            source_refined_features.append(finetuner(source_refine_features, source_decoder_features_torefine))
            target_refined_features.append(finetuner(target_refine_features, target_decoder_features_torefine))

        source_predicts_features = []
        target_predicts_features = []
        for upsample, s_refined_features, t_refined_features in zip(self.feature_upsample, source_refined_features, target_refined_features):
            source_predicts_features.append(upsample(s_refined_features))
            target_predicts_features.append(upsample(t_refined_features))

        source_predict2 = self.finally_out(torch.cat(source_predicts_features, dim=1))
        target_predict2 = self.finally_out(torch.cat(target_predicts_features, dim=1))

        return source_predict1, target_predict1, roi_rate, source_predict2, target_predict2


class FeatureMapExtractor:
    def __init__(self, model, layers=None):
        self.features_out = []
        self.module_name_out = []
        self.handles = []

        if layers:
            for module_name, module in model.named_modules():
                if module_name in layers:
                    self.handles.append(module.register_forward_hook(self.feature_out_hooker))
        else:
            for module in model.modules():
                self.handles.append(module.register_forward_hook(self.feature_out_hooker))

    def feature_out_hooker(self, module, feature_in, feature_out):
        self.module_name_out.append(module.__class__)
        self.features_out.append(feature_out)

    def reset(self):
        self.features_out = []
        self.module_name_out = []
        self.remove_hooker()
        self.handles = []

    def remove_hooker(self):
        for handle in self.handles:
            handle.remove()

    def get_out_feature(self):
        if len(self.features_out) > 0:
            feature = self.features_out
        else:
            feature = None
        self.features_out = []
        self.module_name_out = []
        return feature


if __name__ == "__main__":
    from torchsummary import summary
    from functools import partial
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor

    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    net = ChilopodUnetWithRegression(1, 1, ['target', 'source'], DoubleConv,
                                     f_maps=16, num_levels=5, with_activation=False, final_sigmoid=True, interpolation=True,
                                     norm_type="batch", act_type="lrelu")#.to(device)  # 4.1212M

    # finetunenetv1 = ChilopodUnetWithRegressionFinetuneV1(1, 1, ['target', 'source'], DoubleConv,
    #                                                      f_maps=16, num_levels=5, with_activation=False,
    #                                                      final_sigmoid=True, interpolation=True,
    #                                                      norm_type="batch", act_type="lrelu").to(device)
    #
    # finetunenetv2 = ChilopodUnetWithRegressionFinetuneV2(1, 1, ['target', 'source'], DoubleConv,
    #                                                      f_maps=16, num_levels=5, with_activation=False,
    #                                                      final_sigmoid=True, interpolation=True,
    #                                                      norm_type="batch", act_type="lrelu").to(device)
    print(torch.cuda.memory_reserved(), torch.cuda.memory_allocated())      # 25165824 19150848/ 24576 KB 18702 KB
    print(torch.cuda.memory_summary())
    # feature_extractor = FeatureMapExtractor(net, ['decoders.0', 'decoders.1', 'decoders.2', 'decoders.3'])
    # 'decoders.1.upsampling', 'decoders.1.basic_module.conv2', 'decoders.2.basic_module.conv2'
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

    # with torch.no_grad():
    #     s_in = torch.rand((1, 1, 80, 112, 112), requires_grad=True).to(device)  # 64,96,96
    #     t_in = torch.rand((1, 1, 80, 112, 112), requires_grad=True).to(device)  # 64,96,96
    #     # s_out, t_out, rate = net(s_in, t_in)
    #     s_out1, t_out1, rate, s_out2, t_out2 = finetunenetv2(s_in, t_in)
    #     print(s_out1.size(), t_out1.size(), rate, s_out2.size(), t_out2.size())
    #     # print('--------------------------single model-------------------------------')
    #     # print_model_parm_nums(net)
    #
    # # print_model_parm_flops(net, inputs, need_idx=False, domain="source")  # 751.84G
    # # print(net)
    #
    # # feature_all = feature_extractor.get_out_feature()
    # # print(len(feature_all))
    # # for feature in feature_all:
    # #     print(feature.size(), feature.min(), feature.max(), feature.sum())
    # # print(torch.cuda.memory_summary())





