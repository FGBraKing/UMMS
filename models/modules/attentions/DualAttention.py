import torch
from torch.nn import Module, Conv3d, ReLU, Parameter, Sigmoid, Softmax
import torch.nn as nn

torch_ver = torch.__version__[:3]

__all__ = ['PAMModule', 'CAMModule']


class PAMModule(Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAMModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth, height, width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth, height, width)
        energy = torch.bmm(proj_query, proj_key)     # b,m,m
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, depth, height, width)            # b,c,m
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma*out + x
        return out


class CAMModule(Module):
    """ Channel attention module"""
    def __init__(self, in_dim=None):
        super(CAMModule, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, depth, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma*out + x
        return out


class DualAttention(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm3d, norm_kwargs=None, **kwargs):
        super(DualAttention, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = PAMModule(inter_channels)
        self.cam = CAMModule()
        self.conv_p2 = nn.Sequential(
            nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv3d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv3d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv3d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


class DualAttentionSimple(Module):
    def __init__(self, in_dim, out_dim):
        super(DualAttentionSimple, self).__init__()
        self.pos_attn = PAMModule(in_dim)
        self.cha_attn = CAMModule(in_dim)
        self.out_conv = Conv3d(2*in_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        pos_out = self.pos_attn(x)
        cha_out = self.cha_attn(x)
        out = self.out_conv(pos_out+cha_out)
        return out
