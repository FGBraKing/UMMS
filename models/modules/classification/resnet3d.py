# from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
import math
from functools import partial

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_type='batch', act_type='lrelu'):
        super().__init__()
        use_bias = not (norm_type == 'batch' or norm_type == 'group')

        self.conv1 = conv3x3x3(in_planes, planes, kernel_size=stride+2, stride=stride, padding=1, use_bias=use_bias)
        self.norm1 = get_normalization3d(planes, norm_type)
        self.act = get_activation(act_type)

        self.conv2 = conv3x3x3(planes, planes)
        self.norm2 = get_normalization3d(planes, norm_type)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, norm_type='batch', act_type='lrelu'):
        super().__init__()

        use_bias = not (norm_type == 'batch' or norm_type == 'group')

        self.conv1 = conv1x1x1(in_planes, planes, use_bias=use_bias)
        self.norm1 = get_normalization3d(planes, norm_type)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=stride+2, stride=stride, padding=1, use_bias=use_bias)
        self.norm2 = get_normalization3d(planes, norm_type)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, use_bias=use_bias)
        self.norm3 = get_normalization3d(planes * self.expansion, norm_type)
        self.act = get_activation(act_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1,
                 norm_type='batch',
                 act_type='lrelu'):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = get_normalization3d(self.in_planes, norm_type)
        self.relu = get_activation(act_type)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type,
                                       norm_type=norm_type, act_type=act_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2,
                                       norm_type=norm_type, act_type=act_type)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2,
                                       norm_type=norm_type, act_type=act_type)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2,
                                       norm_type=norm_type, act_type=act_type)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        self.finally_activate = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _downsample_basic_block(x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm_type='batch', act_type='lrelu'):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride=stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample,
                        norm_type=norm_type, act_type=act_type)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

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
        x = self.finally_activate(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet3D(BasicBlock, [1, 1, 1, 1], get_inplanes(4, 64), **kwargs)
    elif model_depth == 18:
        model = ResNet3D(BasicBlock, [2, 2, 2, 2], get_inplanes(4, 64), **kwargs)
    elif model_depth == 34:
        model = ResNet3D(BasicBlock, [3, 4, 6, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 50:
        model = ResNet3D(Bottleneck, [3, 4, 6, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 101:
        model = ResNet3D(Bottleneck, [3, 4, 23, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 152:
        model = ResNet3D(Bottleneck, [3, 8, 36, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 200:
        model = ResNet3D(Bottleneck, [3, 24, 36, 3], get_inplanes(4, 64), **kwargs)
    else:
        raise RuntimeError('failure to built model')
    return model


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from functools import partial
    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    torch.cuda.set_device('cuda:1')
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net = generate_model(50, n_input_channels=1, n_classes=1, norm_type='batch', act_type="lrelu").cuda()

    print('---------------------------------------------------------')
    for name, layer in net.named_children():
        print(name, type(layer))

    inputs = torch.rand((4, 1, 80, 96, 96), requires_grad=True).cuda()
    print_model_parm_nums(net)  # 14.7302M

    output = net(inputs)
    print(len(output))
    for oo in output:
        print(oo)






