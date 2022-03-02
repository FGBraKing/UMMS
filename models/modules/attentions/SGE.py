import torch
from torch import nn
from torch.nn import init


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.weight=nn.Parameter(torch.zeros(1, groups,1,1,1))
        self.bias=nn.Parameter(torch.zeros(1, groups,1,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()

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
        b, c, d, h, w = x.shape
        x=x.view(b*self.groups,-1, d, h,w) #bs*g,dim//g,d,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,d,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,d,h,w
        t=xn.view(b*self.groups,-1) #bs*g,d*h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,d*h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,d,h,w) #bs,g,d*h*w

        t=t*self.weight+self.bias #bs,g,d*h*w
        t=t.view(b*self.groups,1,d,h,w) #bs*g,1,d*h*w
        x=x*self.sig(t)
        x=x.view(b,c,d,h,w)

        return x


if __name__ == '__main__':
    input=torch.randn(50,512,7,7,7)
    sge = SpatialGroupEnhance(groups=8)
    output=sge(input)
    print(output.shape)