import torch
import torch.nn as nn
from models.modules.blocks.blocks3d import same_convlution, downsample_convlution, upsample_deconvlution
from models.auxiliary_funs import get_normalization3d, get_activation, get_3dnorm_layer


# # no need to use bias as BatchNorm3d has affine parameters, 用了affine就不需要bias
def define_G(input_nc, output_nc, ngf, netG, use_dropout=False,
             padding_mode='zeros', norm_type='batch', act_type='relu'):

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, n_blocks=9, use_dropout=use_dropout, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, n_blocks=6, use_dropout=use_dropout, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type)
    elif netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, ngf, use_dropout=use_dropout, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type)
    # elif netG == 'unet_256':
    #     net = UnetGenerator(input_nc, output_nc, 8, ngf, use_dropout=use_dropout, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, padding_mode='zeros', norm_type='batch', act_type='leakyrelu'):

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, 3, padding_mode, norm_type, act_type)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, padding_mode, norm_type, act_type)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, padding_mode, norm_type, act_type)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return net


# ===================================================================================================================
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, padding_mode='zeros', norm_type='batch', act_type='leakyrelu'):
        super(PixelDiscriminator, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')

        self.net = [
            same_convlution(input_nc, ndf, 1, use_bias=use_bias, padding_mode=padding_mode),
            get_activation(act_type),
            same_convlution(ndf, ndf*2, 1, use_bias=use_bias, padding_mode=padding_mode),
            get_normalization3d(ndf * 2, norm_type),
            get_activation(act_type),
            same_convlution(ndf*2, 1, 1, use_bias=use_bias, padding_mode=padding_mode)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Standard forward."""
        return self.net(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 padding_mode='reflect', norm_type='batch', act_type='leakyrelu'):
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


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout=False, use_bias=False,
                 padding_mode='reflect', norm_type='batch', act_type='relu'):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [same_convlution(dim, dim, 3, use_bias=use_bias, padding_mode=padding_mode),
                       get_normalization3d(dim, norm_type),
                       get_activation(act_type)]
        if use_dropout:
            conv_block += [nn.Dropout3d(0.5)]

        conv_block += [same_convlution(dim, dim, 3, use_bias=use_bias, padding_mode=padding_mode),
                       get_normalization3d(dim, norm_type)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # 注意：没有激活
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_blocks=6, use_dropout=False,
                 padding_mode='reflect', norm_type='batch', act_type='relu'):
        super(ResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.in_conv = nn.Sequential(
            same_convlution(input_nc, ngf, 7,  use_bias=use_bias, padding_mode=padding_mode),
            get_normalization3d(ngf, norm_type),
            get_activation(act_type)
        )

        self.encoder_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        n_downsampling = 2
        for i in range(n_downsampling):
            self.encoder_blocks.append(self.encoder_block(ngf*2**i, 2*ngf*2**i, 4, use_bias, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type))
            self.decoder_blocks.insert(0, self.decoder_block(2*ngf*2**i, ngf*2**i, 4, use_bias, padding_mode=padding_mode, norm_type=norm_type, act_type=act_type))
        for i in range(n_blocks):
            self.res_blocks.append(ResnetBlock(ngf*2**n_downsampling, use_dropout=use_dropout, use_bias=use_bias,
                                               padding_mode=padding_mode, norm_type=norm_type, act_type=act_type))

        self.out_conv = nn.Sequential(
            same_convlution(ngf, output_nc, 7, use_bias=use_bias, padding_mode=padding_mode),
            get_activation('tanh')
        )

    def forward(self, x):
        x = self.in_conv(x)
        for encoder in self.encoder_blocks:
            x = encoder(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        for decoder in self.decoder_blocks:
            x = decoder(x)
        x = self.out_conv(x)
        return x

    @staticmethod
    def encoder_block(in_planes, out_planes, kernel_size=4, use_bias=False, use_dropout=False,
                      padding_mode='reflect', norm_type='batch', act_type='leakyrelu'):
        conv = downsample_convlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type)
        act = get_activation(act_type)
        if out_planes >= 512 and use_dropout:
            return nn.Sequential(conv, norm, act, nn.Dropout(0.5))
        else:
            return nn.Sequential(conv, norm, act)

    @staticmethod
    def decoder_block(in_planes, out_planes, kernel_size=4, use_bias=False, use_dropout=False,
                      padding_mode='reflect', norm_type='batch', act_type='leakyrelu', interpolation=False):
        if interpolation:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                same_convlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
            )
        else:
            upconv = upsample_deconvlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type)
        act = get_activation(act_type)
        if out_planes >= 512 and use_dropout:
            return nn.Sequential(upconv, norm, act, nn.Dropout(0.5))
        else:
            return nn.Sequential(upconv, norm, act)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32,
                 padding_mode='reflect', norm_type='batch', act_type='leakyrelu', use_dropout=False):
        super(UnetGenerator, self).__init__()
        ks = 4
        use_bias = not (norm_type == 'batch' or norm_type == 'group')

        self.in_conv = self.encoder_block(input_nc, ngf, ks, use_bias, use_dropout,
                                          padding_mode, norm_type='none', act_type=act_type)
        self.down1 = self.encoder_block(ngf, ngf*2, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.down2 = self.encoder_block(ngf*2, ngf*4, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.down3 = self.encoder_block(ngf*4, ngf*8, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.down4 = self.encoder_block(ngf*8, ngf*16, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.down5 = self.encoder_block(ngf*16, ngf*16, ks, use_bias, use_dropout,
                                        padding_mode, norm_type='none', act_type=act_type)
        #  一般unet是先上采样再cat。cyclegan里的是忽略最里边的下采样，然后先cat再上采样
        self.up1 = self.decoder_block(ngf*16, ngf*16, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.up2 = self.decoder_block(ngf*32, ngf*8, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.up3 = self.decoder_block(ngf*16, ngf*4, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.up4 = self.decoder_block(ngf*8, ngf*2, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)
        self.up5 = self.decoder_block(ngf*4, ngf*1, ks, use_bias, use_dropout, padding_mode, norm_type, act_type)

        self.out_conv = self.decoder_block(ngf*2, output_nc, ks, use_bias, use_dropout,
                                           padding_mode, norm_type='none', act_type='tanh')

    def forward(self, x):
        d1 = self.in_conv(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)

        u1 = torch.cat([d5, self.up1(d6)], dim=1)
        u2 = torch.cat([d4, self.up2(u1)], dim=1)
        u3 = torch.cat([d3, self.up3(u2)], dim=1)
        u4 = torch.cat([d2, self.up4(u3)], dim=1)
        u5 = torch.cat([d1, self.up5(u4)], dim=1)
        out = self.out_conv(u5)
        return out

    @staticmethod
    def encoder_block(in_planes, out_planes, kernel_size=4, use_bias=False, use_dropout=False,
                      padding_mode='reflect', norm_type='batch', act_type='leakyrelu'):
        conv = downsample_convlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type)
        act = get_activation(act_type)
        if out_planes >= 512 and use_dropout:
            return nn.Sequential(conv, norm, act, nn.Dropout(0.5))
        else:
            return nn.Sequential(conv, norm, act)

    @staticmethod
    def decoder_block(in_planes, out_planes, kernel_size=4, use_bias=False, use_dropout=False,
                      padding_mode='reflect', norm_type='batch', act_type='leakyrelu', interpolation=False):
        if interpolation:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                same_convlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
            )
        else:
            upconv = upsample_deconvlution(in_planes, out_planes, kernel_size, use_bias=use_bias, padding_mode=padding_mode)
        norm = get_normalization3d(out_planes, norm_type)
        act = get_activation(act_type)
        if out_planes >= 512 and use_dropout:
            return nn.Sequential(upconv, norm, act, nn.Dropout(0.5))
        else:
            return nn.Sequential(upconv, norm, act)


def main():
    from torchsummary import summary
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')

    # net = ResnetGeneratorz(input_nc=1, output_nc=1, ngf=32).to(device)
    # resnet_9blocks resnet_6blocks unet_64
    net_gen = define_G(input_nc=1, output_nc=1, ngf=32, netG='resnet_9blocks').to(device)
    net_dis = define_D(1, 32, 'n_layers').to(device)
    # for name, module in net.named_modules():  # named_children():
    #     print(name, type(module))
    # print('---------------------------------------------------------')
    # for name, layer in net.named_children():
    #     print(name, type(layer))
    # print('---------------------------------------------------------')
    # for k, v in net.named_parameters():
    #     print(k, v.size())
    #     print(v.nelement())

    # inputs = torch.rand((16, 1, 64, 64, 64), requires_grad=True).to(device)
    # print_model_parm_nums(net)  # 40.15M
    # print_model_parm_flops(net, inputs, need_idx=False)  # 751.84G

    summary(net_gen, input_size=(1, 256, 256, 256), batch_size=1, device='cuda')
    summary(net_dis, input_size=(1, 256, 256, 256), batch_size=1, device='cuda')


if __name__ == "__main__":
    main()
