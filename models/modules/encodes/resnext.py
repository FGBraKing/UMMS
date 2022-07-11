import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks.blocks3d import conv3x3x3, conv1x1x1, create_conv_block, same_convlution
from models.auxiliary_funs import get_normalization3d, get_activation


def get_inplanes(depth=4, initial_channel=64):
    return [initial_channel*2**i for i in range(depth)]


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


class ResnextBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, base_width=4, cardinality=32,
                 norm_type='batch', act_type='lrelu'):
        super(ResnextBasicBlock, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')

        self.conv1 = conv3x3x3(in_planes, planes, kernel_size=stride+2, stride=stride, padding=1, use_bias=use_bias)
        self.norm1 = get_normalization3d(planes, norm_type)

        self.conv2 = conv3x3x3(planes, planes*self.expansion, groups=cardinality)
        self.norm2 = get_normalization3d(planes, norm_type)

        self.act = get_activation(act_type)

        self.downsample = nn.Sequential()
        if in_planes != planes * self.expansion or stride != 1:
            self.downsample.add_module("shortcut_conv", conv1x1x1(in_planes, planes*self.expansion, stride=stride, use_bias=use_bias))
            self.downsample.add_module("shortcut_norm", get_normalization3d(planes*self.expansion), norm_type)

    def forward(self, x):
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        out = self.act(residual+x)

        return out


class ResnextBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, base_width=4, cardinality=32,
                 norm_type='batch', act_type='lrelu'):
        super(ResnextBottleneck, self).__init__()

        use_bias = not (norm_type == 'batch' or norm_type == 'group')

        middle_channel = cardinality * int(base_width * planes / 64)        # 128的时候，middle_channel=planes*2

        self.conv_reduce = conv1x1x1(in_planes, middle_channel, use_bias=use_bias)
        self.norm_reduce = get_normalization3d(middle_channel, norm_type)
        self.conv_mid = conv3x3x3(middle_channel, middle_channel,
                                  kernel_size=stride+2, stride=stride, padding=1, groups=cardinality, use_bias=use_bias)
        self.norm_mid = get_normalization3d(middle_channel, norm_type)
        self.conv_expand = conv1x1x1(middle_channel, planes * self.expansion, use_bias=use_bias)
        self.norm_expand = get_normalization3d(planes * self.expansion, norm_type)
        self.act = get_activation(act_type)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != planes*self.expansion:
            self.downsample.add_module("shortcut_conv",
                                       conv1x1x1(in_planes, planes * self.expansion, stride=stride, use_bias=use_bias))
            self.downsample.add_module("shortcut_norm", get_normalization3d(planes * self.expansion, norm_type))

    def forward(self, x):
        residual = self.downsample(x)

        x = self.act(self.norm_reduce(self.conv_reduce(x)))
        x = self.act(self.norm_mid(self.conv_mid(x)))
        x = self.norm_expand(self.conv_expand(x))

        out = self.act(residual + x)

        return out


class ResNext(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 base_width=4,
                 cardinality=32,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 widen_factor=1.0,
                 n_classes=1):
        super(ResNext, self).__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], base_width=base_width, cardinality=cardinality)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2, base_width=base_width, cardinality=cardinality)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2, base_width=base_width, cardinality=cardinality)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2, base_width=base_width, cardinality=cardinality)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.initial()

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, base_width=4, cardinality=32):

        layers = [block(self.in_planes, planes, stride, base_width, cardinality)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, base_width=base_width, cardinality=cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNext(ResnextBasicBlock, [1, 1, 1, 1], get_inplanes(4, 64), **kwargs)
    elif model_depth == 18:
        model = ResNext(ResnextBasicBlock, [2, 2, 2, 2], get_inplanes(4, 64), **kwargs)
    elif model_depth == 34:
        model = ResNext(ResnextBasicBlock, [3, 4, 6, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 50:
        model = ResNext(ResnextBottleneck, [3, 4, 6, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 101:
        model = ResNext(ResnextBottleneck, [3, 4, 23, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 152:
        model = ResNext(ResnextBottleneck, [3, 8, 36, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 200:
        model = ResNext(ResnextBottleneck, [3, 24, 36, 3], get_inplanes(4, 64), **kwargs)
    else:
        raise RuntimeError('failure to built model')
    return model


class ResnextEncoder(nn.Module):
    def __init__(self, in_channels, depth, f_maps=16, base_width=4, cardinality=32,
                 norm_type="batch", act_type="lrelu", block_style="basic", repeat_style='consistence'):
        super(ResnextEncoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        if isinstance(f_maps, int):
            f_maps = [f_maps * 2 ** x for x in range(self.depth + 1)]
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert self.depth + 1 <= len(f_maps), "Required at least 2 levels in the U-Net when once downsample"

        self.repeat_style = repeat_style
        repeat_num = self.get_repeat_num()

        if block_style == "basic":
            basic_block = ResnextBasicBlock
        elif block_style == "bottle":
            basic_block = ResnextBottleneck
        else:
            raise NotImplementedError

        self.in_conv = DoubleConv(in_channels, f_maps[0], kernel_size=3, norm_type=norm_type, act_type=act_type)
        self.features_channels = [f_maps[0]]
        down_blocks = []
        for i in range(depth):
            if i == 0:
                down_blocks.append(
                    nn.Sequential(
                        nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                        self.make_layer(f_maps[i], f_maps[i+1], 1, norm_type, act_type,
                                        basic_block, repeat_num[0], base_width, cardinality)
                    )
                )
            else:
                down_blocks.append(
                    self.make_layer(f_maps[i]*basic_block.expansion, f_maps[i+1], 2, norm_type, act_type,
                                    basic_block, repeat_num[0], base_width, cardinality)
                )
            self.features_channels.append(f_maps[i+1]*basic_block.expansion)
        self.down_blocks = nn.ModuleList(down_blocks)

    @staticmethod
    def make_layer(in_plane, out_plane, stride, norm_type, act_type, block, blocks_num, base_width, cardinality):
        modules = [block(in_plane, out_plane, stride, base_width, cardinality, norm_type, act_type)]

        in_plane = out_plane * block.expansion
        for i in range(1, blocks_num):
            modules.append(block(in_plane, out_plane, 1, base_width, cardinality, norm_type, act_type))
        return nn.Sequential(*modules)

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

    def get_repeat_num(self):
        if self.repeat_style == "consistence":
            repeat_num = (2,) * self.depth
        else:
            if self.depth == 4:
                repeat_num = [3, 4, 6, 3]
            elif self.depth == 3:
                repeat_num = [3, 4, 3]
            else:
                raise NotImplementedError
        return repeat_num


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from functools import partial
    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    torch.cuda.set_device('cuda:1')
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = ResnextEncoder(in_channels=1, depth=4, f_maps=16, norm_type='batch', act_type="lrelu",
                         block_style="bottle", repeat_style='not consistence').cuda()

    print('---------------------------------------------------------')
    for name, layer in net.named_children():
        print(name, type(layer))

    inputs = torch.rand((4, 1, 80, 96, 96), requires_grad=True).cuda()
    print_model_parm_nums(net)  # 5.8266M

    outputs = net(inputs)
    print(len(outputs))
    for oo in outputs:
        print(oo.size())
    print(net.out_channels)






