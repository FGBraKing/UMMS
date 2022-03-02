# from https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks.blocks3d import conv3x3x3, conv1x1x1, create_conv_block


def get_inplanes(depth=4, initial_channel=64):
    return [initial_channel*2**i for i in range(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, kernel_size=stride+2, stride=stride, padding=1, use_bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=stride+2, stride=stride, padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=1):
        super().__init__()

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

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

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

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)]
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

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(4, 64), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(4, 64), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(4, 64), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(4, 64), **kwargs)
    else:
        raise RuntimeError('failure to built model')
    return model


class ResnetEncoder(nn.Module):
    def __init__(self, in_channels, depth, f_maps=16, layers=(2, 2, 2, 2, 2), basic_block=BasicBlock):
        super(ResnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        if isinstance(f_maps, int):
            f_maps = [f_maps*2**x for x in range(self.depth + 1)]
        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert self.depth + 1 <= len(f_maps), "Required at least 2 levels in the U-Net when once downsample"
        assert self.depth <= len(layers), "Required at least 2 levels in the U-Net when once downsample"
        self.features_channels = f_maps[:self.depth + 1]
        self.layers = layers[:self.depth]

        self.in_conv = create_conv_block(in_channels, f_maps[0], kernel_size=3, order='cbr', padding=1, num_groups=1)
        self.layer1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            self.make_layer(f_maps[0], f_maps[1], stride=1, blocks_num=layers[0], basic_block=basic_block)
        )
        self.rest_down_blocks = nn.ModuleList([
            self.make_layer(f_maps[i], f_maps[i+1], 2, layers[i], basic_block) for i in range(1, self.depth)
        ])

    @staticmethod
    def make_layer(in_plane, out_plane, stride, blocks_num, basic_block):
        downsample = None
        if stride != 1 or in_plane != out_plane * basic_block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_plane, out_plane * basic_block.expansion, stride, stride, 0, 1, 1, bias=False),
                nn.BatchNorm3d(out_plane * basic_block.expansion))
        modules = [basic_block(in_planes=in_plane, planes=out_plane, stride=stride, downsample=downsample)]

        in_plane = out_plane * basic_block.expansion
        for i in range(1, blocks_num):
            modules.append(basic_block(in_planes=in_plane, planes=out_plane))
        return nn.Sequential(*modules)

    def forward(self, x):
        encoders_features = []
        x = self.in_conv(x)
        encoders_features.append(x)
        x = self.layer1(x)
        encoders_features.append(x)
        for encode in self.rest_down_blocks:
            x = encode(x)
            encoders_features.append(x)
        return encoders_features

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self.features_channels











