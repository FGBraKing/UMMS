import torch.nn as nn
from models.modules.blocks.blocks3d import create_conv_block, same_convlution
from models.auxiliary_funs import get_normalization3d, get_activation


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm_type="batch", act_type="lrelu", use_norm=True, use_act=True):
        super(ConvBlock, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.conv = same_convlution(in_channels, out_channels, kernel_size, use_bias=use_bias)
        self.norm = get_normalization3d(out_channels, norm_type)
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
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_type="batch", act_type="lrelu"):
        super(DoubleConv, self).__init__()
        conv1_in_channels = in_channels
        conv1_out_channels = max(in_channels, out_channels // 2)
        conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, kernel_size, norm_type, act_type)  # cbr
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, kernel_size, norm_type, act_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type="batch", act_type="lrelu"):
        super(Down, self).__init__()
        self.downsample = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_channels, out_channels, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        return x


class Vgg13Encode(nn.Module):
    def __init__(self, in_channels, depth, f_maps=16, norm_type="batch", act_type="lrelu"):
        super(Vgg13Encode, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        if isinstance(f_maps, int):
            f_maps = [f_maps*2**x for x in range(self.depth + 1)]
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert self.depth + 1 == len(f_maps), "Required at least 2 levels in the U-Net when once downsample"
        self.features_channels = f_maps

        self.in_conv = DoubleConv(in_channels, f_maps[0], norm_type=norm_type, act_type=act_type)
        self.down_blocks = nn.ModuleList([Down(f_maps[i], f_maps[i+1], norm_type, act_type) for i in range(self.depth)])

    def forward(self, x):
        encoders_features = []
        x = self.in_conv(x)
        encoders_features.append(x)
        for encode in self.down_blocks:
            x = encode(x)
            encoders_features.append(x)
        return encoders_features

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self.features_channels


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from functools import partial
    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    torch.cuda.set_device('cuda:1')
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = Vgg13Encode(in_channels=1, depth=4, f_maps=16, norm_type='batch', act_type="lrelu").cuda()

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

    inputs = torch.rand((4, 1, 80, 96, 96), requires_grad=True).cuda()
    print_model_parm_nums(net)  # 1.7677M

    out = net(inputs)
    print(len(out))
    for oo in out:
        print(oo.size())
    print(net.out_channels)


