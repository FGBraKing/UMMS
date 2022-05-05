import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks.blocks3d import same_convlution, downsample_convlution, upsample_deconvlution
from models.auxiliary_funs import get_normalization3d, get_activation


##############################
#        Discriminator
##############################
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, n_layer=3, num_scales=3, dim=32,
                 norm='adain', activ='relu', pad_type='reflect', gan_type='lsgan'):
        super(MsImageDis, self).__init__()
        self.gan_type = gan_type
        self.cnns = nn.ModuleList()
        for _ in range(num_scales):
            self.cnns.append(self._make_net(input_dim, dim, n_layer, norm, activ, pad_type))
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)

    def _make_net(self, input_dim, dim, n_layer, norm_type, act_type, padding_mode):
        cnn_x = []
        cnn_x += [Conv3dBlock(input_dim, dim, 4, 2, 1, norm_type='none', act_type=act_type, padding_mode=padding_mode)]
        for i in range(n_layer - 1):
            cnn_x += [Conv3dBlock(dim*2**i, dim*2*2**i, 4, 2, 1, norm_type=norm_type, act_type=act_type, padding_mode=padding_mode)]
        cnn_x += [nn.Conv3d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(out0)
                all1 = torch.ones_like(out1)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out0)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, dim, style_dim, mlp_dim=128, n_downsample=4, n_res=3, activ='relu', pad_type='reflect'):
        super(AdaINGen, self).__init__()
        # encoders
        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)

        # decoders
        self.dec = Decoder(n_downsample, n_res, dim*2**n_downsample, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    @staticmethod
    def assign_adain_params(adain_params, model):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                # Extract mean and std predictions
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    @staticmethod
    def get_num_adain_params(model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm3d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, dim, n_downsample, n_res, activ, pad_type):
        super(VAEGen, self).__init__()
        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution
        # with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training:
            noise = torch.randint_like(hiddens)
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        return hiddens

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm='none', activ='relu', pad_type='zeros'):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm_type=norm, act_type=activ, padding_mode=pad_type)]
        for i in range(2):
            self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm_type=norm, act_type=activ, padding_mode=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv3dBlock(dim, dim, 4, 2, 1, norm_type=norm, act_type=activ, padding_mode=pad_type)]
        self.model += [nn.AdaptiveAvgPool3d(1)]  # global average pooling
        self.model += [nn.Conv3d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim,  norm='in', activ='relu', pad_type='zeros'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm_type=norm, act_type=activ, padding_mode=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv3dBlock(dim*2**i, 2*dim*2**i, 4, 2, 1, norm_type=norm, act_type=activ, padding_mode=pad_type)]
        # residual blocks
        self.model += [ResBlocks(n_res, dim*2**n_downsample, norm_type=norm, act_type=activ, padding_mode=pad_type)]
        self.model = nn.Sequential(*self.model)
        # self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv3dBlock(dim, dim // 2, 5, 1, 2, 'ln', activ, pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm_type='none', act_type='tanh', padding_mode=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm_type='in', act_type='relu', padding_mode='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm_type=norm_type, act_type=act_type, padding_mode=padding_mode)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


#   MLP (predicts AdaIn parameters)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm_type=norm, act_type=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm_type=norm, act_type=activ)]
        self.model += [LinearBlock(dim, output_dim, norm_type='none', act_type='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm_type='in', act_type='relu', padding_mode='zeros'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv3dBlock(dim, dim, 3, 1, 1, norm_type=norm_type, act_type=act_type, padding_mode=padding_mode)]
        model += [Conv3dBlock(dim, dim, 3, 1, 1, norm_type=norm_type, act_type='none', padding_mode=padding_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


ResidualBlock = ResBlock


class Conv3dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 norm_type='none', act_type='relu', padding_mode='zeros', use_bias=True):
        super(Conv3dBlock, self).__init__()

        # initialize normalization
        if norm_type == 'ln':
            self.norm = LayerNorm(output_dim)
            self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=use_bias, padding_mode=padding_mode)
        elif norm_type == 'adain':
            self.norm = AdaptiveInstanceNorm3d(output_dim)
            self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=use_bias, padding_mode=padding_mode)
        elif norm_type == 'sn':
            self.norm = None
            self.conv = SpectralNorm(nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=use_bias, padding_mode=padding_mode))
        else:
            self.norm = get_normalization3d(output_dim, norm_type)
            self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, padding, bias=use_bias, padding_mode=padding_mode)

        # initialize activation
        self.activation = get_activation(act_type)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_type='batch', act_type='relu', use_bias=True):
        super(LinearBlock, self).__init__()
        # initialize normalization and fully connected layer
        if norm_type == 'ln':
            self.norm = LayerNorm(output_dim)
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        elif norm_type == 'sn':
            self.norm = None
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.norm = get_normalization3d(output_dim, norm_type)
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize activation
        self.activation = get_activation(act_type)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(Vgg16, self).__init__()

        self.conv1_1 = same_convlution(input_nc, ngf, 3, use_bias=True, padding_mode='zeros')
        self.conv1_2 = same_convlution(ngf, ngf, 3, use_bias=True, padding_mode='zeros')
        self.pool1 = nn.MaxPool3d(2, 2)

        self.conv2_1 = same_convlution(ngf, ngf*2, 3, use_bias=True, padding_mode='zeros')
        self.conv2_2 = same_convlution(ngf*2, ngf*2, 3, use_bias=True, padding_mode='zeros')
        self.pool2 = nn.MaxPool3d(2, 2)

        self.conv3_1 = same_convlution(ngf*2, ngf*4, 3, use_bias=True, padding_mode='zeros')
        self.conv3_2 = same_convlution(ngf*4, ngf*4, 3, use_bias=True, padding_mode='zeros')
        self.conv3_3 = same_convlution(ngf*4, ngf*4, 3, use_bias=True, padding_mode='zeros')
        self.pool3 = nn.MaxPool3d(2, 2)

        self.conv4_1 = same_convlution(ngf*4, ngf*8, 3, use_bias=True, padding_mode='zeros')
        self.conv4_2 = same_convlution(ngf*8, ngf*8, 3, use_bias=True, padding_mode='zeros')
        self.conv4_3 = same_convlution(ngf*8, ngf*8, 3, use_bias=True, padding_mode='zeros')

        self.conv5_1 = same_convlution(ngf*8, ngf*8, 3, use_bias=True, padding_mode='zeros')
        self.conv5_2 = same_convlution(ngf*8, ngf*8, 3, use_bias=True, padding_mode='zeros')
        self.conv5_3 = same_convlution(ngf*8, ngf*8, 3, use_bias=True, padding_mode='zeros')

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.conv1_1(x))
        h = self.act(self.conv1_2(h))
        # relu1_2 = h
        h = self.pool1(h)

        h = self.act(self.conv2_1(h))
        h = self.act(self.conv2_2(h))
        # relu2_2 = h
        h = self.pool2(h)

        h = self.act(self.conv3_1(h))
        h = self.act(self.conv3_2(h))
        h = self.act(self.conv3_3(h))
        # relu3_3 = h
        h = self.pool3(h)

        h = self.act(self.conv4_1(h))
        h = self.act(self.conv4_2(h))
        h = self.act(self.conv4_3(h))
        # relu4_3 = h

        h = self.act(self.conv5_1(h))
        h = self.act(self.conv5_2(h))
        h = self.act(self.conv5_3(h))
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]


##################################################################################
#  Custom Normalization layers
##################################################################################
# # 内置的LayerNorm的affine参数是elementwise的，这里的affine是整体. num_features==bs
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# 默认的instanceNorm是不用affine和track_running_stats的。这里用了affine，
class AdaptiveInstanceNorm3d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (self.weight is not None and self.bias is not None
                ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

    @staticmethod
    def l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)
