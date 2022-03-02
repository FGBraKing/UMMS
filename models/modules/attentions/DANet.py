import torch
from torch import nn
from .SelfAttention import ScaledDotProductAttention
from .SimplifiedSelfAttention import SimplifiedScaledDotProductAttention


class PositionAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, D= 7, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv3d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)

        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def forward(self, x):
        bs, c, d, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1).permute(0, 2, 1)  # bs,d*h*w,c
        y = self.pa(y, y, y)  # bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, D=7, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv3d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2)

        self.pa = SimplifiedScaledDotProductAttention(D*H*W, h=1)

    def forward(self, x):
        bs, c, d, h, w = x.shape
        y = self.cnn(x)
        y = y.view(bs, c, -1)  # bs,c,d*h*w
        y = self.pa(y, y, y)  # bs,c,d*h*w
        return y


class DAModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, D=7, H=7, W=7):
        super().__init__()
        self.position_attention_module = PositionAttentionModule(d_model, kernel_size, D, H, W)
        self.channel_attention_module = ChannelAttentionModule(d_model, kernel_size, D, H, W)

    def forward(self, input):
        bs, c, d, h, w = input.shape
        p_out = self.position_attention_module(input)
        c_out = self.channel_attention_module(input)
        p_out = p_out.permute(0, 2, 1).view(bs, c, d, h, w)
        c_out = c_out.view(bs, c, d, h, w)
        return p_out+c_out


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7, 7)
    danet = DAModule(d_model=512, kernel_size=3, D=7, H=7, W=7)
    print(danet(input).shape)
