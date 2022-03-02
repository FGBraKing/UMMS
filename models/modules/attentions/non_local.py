""" Bilinear-Attention-Transform and Non-Local Attention
Paper: `Non-Local Neural Networks With Grouped Bilinear Attentional Transforms`
    - https://openaccess.thecvf.com/content_CVPR_2020/html/Chi_Non-Local_Neural_Networks_With_Grouped_Bilinear_Attentional_Transforms_CVPR_2020_paper.html
Adapted from original code: https://github.com/BA-Transform/BAT-Image-Classification
"""
import torch
from torch import nn
from torch.nn import functional as F


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class NonLocalAttn(nn.Module):
    """Spatial NL block for image classification.
    This was adapted from https://github.com/BA-Transform/BAT-Image-Classification
    Their NonLocal impl inspired by https://github.com/facebookresearch/video-nonlocal-net.
    """

    def __init__(self, in_channels, use_scale=True,  rd_ratio=1/8, rd_channels=None, rd_divisor=8, **kwargs):
        super(NonLocalAttn, self).__init__()
        if rd_channels is None:
            rd_channels = make_divisible(in_channels * rd_ratio, divisor=rd_divisor)
        self.scale = in_channels ** -0.5 if use_scale else 1.0
        self.t = nn.Conv3d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.p = nn.Conv3d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.g = nn.Conv3d(in_channels, rd_channels, kernel_size=1, stride=1, bias=True)
        self.z = nn.Conv3d(rd_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.norm = nn.BatchNorm3d(in_channels)
        self.reset_parameters()

    def forward(self, x):
        shortcut = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        B, C, D, H, W = t.size()
        t = t.view(B, C, -1).permute(0, 2, 1)
        p = p.view(B, C, -1)
        g = g.view(B, C, -1).permute(0, 2, 1)

        att = torch.bmm(t, p) * self.scale
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1).reshape(B, C, D, H, W)
        x = self.z(x)
        x = self.norm(x) + shortcut

        return x

    def reset_parameters(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if len(list(m.parameters())) > 1:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    input=torch.randn(50,512,7,7,7)
    attn = NonLocalAttn(in_channels=512)
    output=attn(input)

    print(output.shape)
