import os
import torch
import itertools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.modules.blocks.blocks3d import same_convlution, downsample_convlution, upsample_deconvlution, create_conv_block, conv1x1x1, conv3x3x3
from models.auxiliary_funs import get_normalization3d, get_activation
from functools import partial
from collections import OrderedDict, defaultdict


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 padding_mode='reflect', norm_type='batch', act_type='leakyrelu'):
        # replicate circular reflect constant
        super(NLayerDiscriminator, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        kw = 4
        padw = 1
        sequence = [
            downsample_convlution(input_nc, ndf, kw, use_bias=use_bias, padding_mode=padding_mode),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                downsample_convlution(ndf * nf_mult_prev, ndf * nf_mult, kw, use_bias=use_bias, padding_mode=padding_mode),
                get_normalization3d(ndf * nf_mult, norm_type),
                get_activation(act_type)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            same_convlution(ndf * nf_mult_prev, ndf * nf_mult, 3, use_bias=use_bias, padding_mode=padding_mode),
            get_normalization3d(ndf * nf_mult, norm_type),
            get_activation(act_type),
            same_convlution(ndf * nf_mult, 1, 3, use_bias=use_bias, padding_mode=padding_mode)
        ]

        # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# copy from CycleGAN
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            loss = None
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        # torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def cal_gradient_penalty_feature(net_df, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    def mix_data(real, fake, alph):
        alph = alph.expand(real.shape[0], real.nelement() // real.shape[0]).contiguous().view(*real.shape)
        interpol = alph * real + ((1 - alph) * fake)
        return interpol
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            interpolatesv = []
            for rr, ff in zip(real_data, fake_data):
                alpha = torch.rand(rr.shape[0], 1, device=device)
                interpolatesv.append(mix_data(rr, ff, alpha))
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        for inter_feature in interpolatesv:
            inter_feature.requires_grad_(True)
        disc_interpolates = net_df(*interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)

        gradients_norm = []
        for gradient in gradients:
            gradient = gradient.view(gradient.size(0), -1)
            gradients_norm.append((gradient + 1e-16).norm(2, dim=1))
        gradients_norm = torch.stack(gradients_norm).mean(0)

        gradient_penalty = ((gradients_norm - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# # Clip weights of discriminator, clip_value=0.01
# for p in discriminator.parameters():
#     p.data.clamp_(-opt.clip_value, opt.clip_value)
# -------------------------------------------------------------------------------------------------------------------
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
    def __init__(self, in_channels, out_channels, in_encoder, kernel_size=3, norm_type="batch", act_type="lrelu", **kwargs):
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


class MCMDA(nn.Module):
    def __init__(self, in_channels, out_channels, domains=None, basic_module=DoubleConv, f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(MCMDA, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.source_encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)
        self.target_encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)

        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, interpolation, norm_type, act_type)
        self.outconv = OutConv(f_maps[0], out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)

    def get_target_encoders_parameters(self):
        encoders_params = []
        for encoder in self.target_encoders:
            encoders_params.append(encoder.parameters())
        return itertools.chain(*encoders_params)

    def set_requires_grad_source_encoders(self, requires_grad=False):
        for encoder in self.source_encoders:
            for param in encoder.parameters():
                param.requires_grad = requires_grad

    def set_requires_grad_target_encoders(self, requires_grad=False):
        for encoder in self.target_encoders:
            for param in encoder.parameters():
                param.requires_grad = requires_grad

    def set_requires_grad_decoders(self, requires_grad=False):
        for decoder in self.decoders:
            for param in decoder.parameters():
                param.requires_grad = requires_grad

    def set_requires_grad_outconv(self, requires_grad=False):
        for param in self.outconv.parameters():
            param.requires_grad = requires_grad

    def forward(self, x, domain=None):
        encoders = self.target_encoders if domain == "target" else self.source_encoders

        encoders_features = []
        for encoder in encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        x = self.outconv(x)

        return x


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


def process_singleweights_to_multiweights(src_weight):
    out_result = r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/AdversarialDA/sourceWeights/'
    if not os.path.isdir(out_result):
        os.makedirs(out_result)
    out_state_dict = OrderedDict()

    state_dict = torch.load(src_weight, map_location='cpu')['umms']
    for key, value in state_dict.items():
        if key.find('encoders') == 0:
            sourcekey = key.replace('encoders', 'source_encoders')
            targetkey = key.replace('encoders', 'target_encoders')
            out_state_dict[sourcekey] = value
            out_state_dict[targetkey] = value
        else:
            out_state_dict[key] = value
    filename = os.path.basename(src_weight)
    save_path = os.path.join(out_result, filename)

    torch.save(out_state_dict, save_path)


def batch_process_source_weights():
    fpaths = [
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold0_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_1080Ti/latest_net_mrus11211280_fold0_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_1080Ti.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold0_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti/latest_net_mrus11211280_fold0_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold1_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_1080Ti/latest_net_mrus11211280_fold1_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_1080Ti.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold1_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti/latest_net_mrus11211280_fold1_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold2_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_Tesla/latest_net_mrus11211280_fold2_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_Tesla.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold2_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti/latest_net_mrus11211280_fold2_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold3_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_Tesla/latest_net_mrus11211280_fold3_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_Tesla.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold3_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti/latest_net_mrus11211280_fold3_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_2080Ti.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold4_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_TITAN/latest_net_mrus11211280_fold4_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_TITAN.pth',
        r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/SingleOne/mrus11211280_fold4_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_TITAN/latest_net_mrus11211280_fold4_bs4_SingleTarget_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_TITAN.pth'
    ]
    for fpath in fpaths:
        process_singleweights_to_multiweights(fpath)


class FeatureDiscriminator(nn.Module):
    def __init__(self, fmap=16, mid_ch=8, ndf=16, n_layers=4,
                 padding_mode='replicate', norm_type='batch', act_type='leakyrelu'):
        super(FeatureDiscriminator, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.conv_down1 = nn.Sequential(
            conv1x1x1(fmap*16, mid_ch, use_bias=use_bias),
            get_normalization3d(mid_ch, norm_type),
            get_activation(act_type)
        )
        self.conv_down2 = nn.Sequential(
            conv1x1x1(fmap*8, mid_ch, use_bias=use_bias),
            get_normalization3d(mid_ch, norm_type),
            get_activation(act_type)
        )
        self.conv_down3 = nn.Sequential(
            conv1x1x1(fmap*4, mid_ch, use_bias=use_bias),
            get_normalization3d(mid_ch, norm_type),
            get_activation(act_type)
        )
        self.conv_down4 = nn.Sequential(
            conv1x1x1(fmap*2, mid_ch, use_bias=use_bias),
            get_normalization3d(mid_ch, norm_type),
            get_activation(act_type)
        )
        self.conv_down5 = nn.Sequential(
            conv1x1x1(fmap, mid_ch, use_bias=use_bias),
            get_normalization3d(mid_ch, norm_type),
            get_activation(act_type)
        )
        self.conv_fuse = nn.Sequential(
            conv3x3x3(mid_ch*5, mid_ch, use_bias=use_bias),
            get_normalization3d(mid_ch, norm_type),
            get_activation(act_type)
        )
        self.discriminator = NLayerDiscriminator(mid_ch, ndf, n_layers, padding_mode, norm_type, act_type)

    def forward(self, en_last, de0, de1, de2, de3):
        out_size = de3.size()
        en_last = F.interpolate(self.conv_down1(en_last), out_size[2:])
        de0 = F.interpolate(self.conv_down2(de0), out_size[2:])
        de1 = F.interpolate(self.conv_down3(de1), out_size[2:])
        de2 = F.interpolate(self.conv_down4(de2), out_size[2:])
        de3 = self.conv_down5(de3)

        x = torch.cat((en_last, de0, de1, de2, de3), dim=1)
        x = self.conv_fuse(x)
        x = self.discriminator(x)
        return x


class MaskDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=16, n_layers=4,
                 padding_mode='replicate', norm_type='batch', act_type='leakyrelu'):
        super(MaskDiscriminator, self).__init__()
        self.mask_discriminator = NLayerDiscriminator(input_nc, ndf, n_layers, padding_mode, norm_type, act_type)

    def forward(self, mask):
        return self.mask_discriminator(mask)


if __name__ == "__main__":
    from torchsummary import summary
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    net = MCMDA(1, 1, ['target', 'source'], DoubleConv, 16, 5,
                with_activation=False, final_sigmoid=True, norm_type="batch", act_type="lrelu").to(device)

    weight_path = r'/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/AdversarialDA/sourceWeights/latest_net_mrus11211' \
                  r'280_fold0_bs4_SingleSource_ch16_combo_1_1_1_l2_1e-4_adam_1e-4_poly_3x30_0.6_AugPlusPlus_1080Ti.pth'
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict, strict=True)

    feature_extractor = FeatureMapExtractor(net, ['source_encoders.4',
                                                  'target_encoders.4',
                                                  'decoders.0',
                                                  'decoders.1',
                                                  'decoders.2',
                                                  'decoders.3'])
    logit_extractor = FeatureMapExtractor(net, ['outconv.conv3d'])

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

    inputs = torch.rand((2, 1, 80, 112, 112), requires_grad=True).to(device)  # 64,96,96
    print_model_parm_nums(net)
    # print(net)
    out = net(inputs, 'source')
    print(out.size())
    print('--------------------------single model-------------------------------')
    # print_model_parm_flops(net, inputs, need_idx=False)  # 751.84G
    # summary(net, input_size=(1, 80, 112, 112), batch_size=2, device='cuda')
    # print(net)

    features = feature_extractor.get_out_feature()
    print(len(features))
    for feature in features:
        print(feature.size())

    df = FeatureDiscriminator().to(device)
    logit = df(*features)
    print(logit.size())
    print('--------------------------named_modules-------------------------------')
    for name, layer in df.named_modules():
        print(name, type(layer))
    print('-------------------------named_parameters--------------------------------')
    for k, v in df.named_parameters():
        print(k, v.size())
        print(v.nelement())

    dm = MaskDiscriminator().to(device)
    mask_logit = logit_extractor.get_out_feature()
    mask_prob = dm(*mask_logit)
    print(mask_prob.size())
    print('--------------------------named_modules-------------------------------')
    for name, layer in dm.named_modules():
        print(name, type(layer))
    print('-------------------------named_parameters--------------------------------')
    for k, v in dm.named_parameters():
        print(k, v.size())
        print(v.nelement())

