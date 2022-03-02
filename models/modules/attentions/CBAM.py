import numpy as np
import torch
from torch import nn
from torch.nn import init


# n c 1 1 1
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel//reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out+avg_out)
        return output


class LightChannelAttn(ChannelAttention):
    def __init__(self, channels, reduction=16):
        super(LightChannelAttn, self).__init__(channels, reduction)

    def forward(self, x):
        x_pool = self.maxpool(x) + self.avgpool(x)
        x_attn = self.sigmoid(self.se(x_pool))
        return x_attn


# n 1 d h w
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """
    def __init__(self, kernel_size=7):
        super(LightSpatialAttn, self).__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_attn = 0.5 * x.mean(dim=1, keepdim=True) + 0.5 * x.amax(dim=1, keepdim=True)
        x_attn = self.sigmoid(self.conv(x_attn))
        return x_attn


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # b, c, _, _, _ = x.size()
        residual = x
        out = x*self.ca(x)      # # n c 1 1 1
        out = out*self.sa(out)  # # n 1 d h w
        return out+residual


class LightCbamModule(nn.Module):
    def __init__(self, channels=512, reduction=16, kernel_size=7):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(channels, reduction)
        self.spatial = LightSpatialAttn(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


if __name__ == '__main__':
    input=torch.randn(50, 512, 7, 7, 7)
    ker_size = input.shape[2]
    cbam = CBAMBlock(channel=512, reduction=16, kernel_size=ker_size)
    output = cbam(input)
    print(output.shape)
