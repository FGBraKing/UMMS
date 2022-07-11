import json
import math
import warnings
from collections import OrderedDict, namedtuple
from copy import copy

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp
from models.modules.blocks.blocks3d import CBS, create_conv_block


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels

        self.cv1 = create_conv_block(c1, c_, 1, 'cbs')
        self.cv2 = create_conv_block(c_ * (len(k) + 1), c2, 1, 'cbs')
        self.m = nn.ModuleList([nn.MaxPool3d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = create_conv_block(c1, c_, 1, 'cbs')
        self.cv2 = create_conv_block(c_ * 4, c2, 1, 'cbs')
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)

        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(c1 * 8, c2, k, s, p, groups=g, bias=False),
            nn.BatchNorm3d(c2),
            nn.SiLU()
        )

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2, ::2],
                                    x[..., 1::2, ::2, ::2],
                                    x[..., ::2, 1::2, ::2],
                                    x[..., 1::2, 1::2, ::2],
                                    x[..., ::2, ::2, 1::2],
                                    x[..., 1::2, ::2, 1::2],
                                    x[..., ::2, 1::2, 1::2],
                                    x[..., 1::2, 1::2, 1::2]], 1))














