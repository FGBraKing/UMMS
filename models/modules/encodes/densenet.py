# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/densenet.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.modules.blocks.blocks3d import conv3x3x3, conv1x1x1, create_conv_block, same_convlution
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


class SingleLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, growth_size=1, drop_rate=0.0, norm_type="batch", act_type="lrelu"):
        super(SingleLayer, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.norm = get_normalization3d(in_channels, norm_type)
        self.act = get_activation(act_type)
        self.conv = conv3x3x3(in_channels, growth_rate, use_bias=use_bias)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(p=self.drop_rate, inplace=False)

    def forward(self, x):
        residual = x
        x = self.conv((self.act(self.norm(x))))
        if self.drop_rate > 0:
            x = self.dropout(x)
        out = torch.cat((x, residual), 1)
        return out


class DoubleLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, growth_size=4, drop_rate=0.0, norm_type="batch", act_type="lrelu"):
        super(DoubleLayer, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.norm1 = get_normalization3d(in_channels, norm_type)
        self.conv1 = conv1x1x1(in_channels, growth_rate*growth_size, use_bias=use_bias)

        self.norm2 = get_normalization3d(growth_size*growth_rate, norm_type)
        self.conv2 = conv3x3x3(growth_size*growth_rate, growth_rate, use_bias=use_bias)

        self.act = get_activation(act_type)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(p=self.drop_rate, inplace=False)

    def forward(self, x):
        residual = x
        x = self.conv1((self.act(self.norm1(x))))
        x = self.conv2((self.act(self.norm2(x))))
        if self.drop_rate > 0:
            x = self.dropout(x)
        out = torch.cat((x, residual), 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0, norm_type="batch", act_type="lrelu"):
        super(Transition, self).__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')
        self.norm = get_normalization3d(in_channels, norm_type)
        self.conv = conv1x1x1(in_channels, out_channels, use_bias=use_bias)
        self.act = get_activation(act_type)

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(p=self.drop_rate, inplace=False)
        self.pool = nn.AvgPool3d(2)

    def forward(self, x):
        x = self.conv(self.act(self.norm(x)))
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, growth_size=4, drop_rate=0.0,
                 norm_type="batch", act_type="lrelu", block_type="double"):
        super(DenseBlock, self).__init__()
        if block_type == "double":
            block = DoubleLayer
        else:
            block = SingleLayer

        for i in range(num_layers):
            layer = block(in_channels + i * growth_rate, growth_rate, growth_size, drop_rate, norm_type, act_type)
            self.add_module('denselayer{}'.format(i + 1), layer)


class DenseNet(nn.Module):
    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 trans_theta=0.5,
                 drop_rate=0,
                 num_classes=1000):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               in_channels=num_features,
                               growth_size=bn_size,
                               growth_rate=growth_rate,
                               drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(in_channels=num_features,
                                   out_channels=int(num_features * trans_theta))
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = int(num_features * trans_theta)

        # Final batch norm and relu
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))

        # Final adaptive_avg_pool3d
        self.final_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def initial(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.final_pool(features).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def generate_model(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 24, 16),
                         **kwargs)
    elif model_depth == 169:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 32, 32),
                         **kwargs)
    elif model_depth == 201:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 48, 32),
                         **kwargs)
    elif model_depth == 264:
        model = DenseNet(num_init_features=64,
                         growth_rate=32,
                         block_config=(6, 12, 64, 48),
                         **kwargs)

    return model


# note！：这里输出的feature是没有经过norm和activation的convlution的特征
class DenseNetEncoder(nn.Module):
    def __init__(self, in_channels, depth, init_features=16, growth_rate=8, growth_size=4, trans_theta=0.5, drop_rate=0,
                 norm_type="batch", act_type="lrelu", block_style="double", repeat_style='dense'):
        super(DenseNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.depth = depth

        self.features_channels = [init_features]

        self.in_conv = DoubleConv(in_channels, init_features, kernel_size=3, norm_type=norm_type, act_type=act_type)

        repeat_num = self.get_repeat_num(repeat_style)

        down_blocks = []
        num_features = init_features
        for i in range(self.depth):
            if i == 0:
                down_blocks.append(nn.Sequential(
                    nn.MaxPool3d(2),
                    DenseBlock(repeat_num[i], num_features, growth_rate, growth_size, drop_rate, norm_type, act_type, block_style)
                ))
                num_features = num_features + repeat_num[i] * growth_rate
            else:
                down_blocks.append(nn.Sequential(
                    Transition(num_features, int(num_features * trans_theta), drop_rate, norm_type, act_type),
                    DenseBlock(repeat_num[i], int(num_features * trans_theta), growth_rate, growth_size, drop_rate, norm_type, act_type, block_style)
                ))
                num_features = int(num_features * trans_theta) + repeat_num[i] * growth_rate
            self.features_channels.append(num_features)
        self.down_blocks = nn.ModuleList(down_blocks)

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

    def get_repeat_num(self, repeat_style):
        if repeat_style == "consistence":
            repeat_num = [6*(2**a) for a in range(self.depth)]
        else:
            if self.depth == 2:
                repeat_num = (6, 12)
            elif self.depth == 3:
                repeat_num = (6, 12, 24)
            elif self.depth == 4:
                repeat_num = (6, 12, 24, 16)
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
    net = DenseNetEncoder(1, 4, 16, 8).cuda()

    print('---------------------------------------------------------')
    for name, net_layer in net.named_children():
        print(name, type(net_layer))

    inputs = torch.rand((4, 1, 80, 96, 96), requires_grad=True).cuda()
    print_model_parm_nums(net)  # 0.7202M

    output = net(inputs)
    print(len(output))
    for oo in output:
        print(oo.size())
    print(net.out_channels)
