import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from models.modules.blocks.blocks3d import create_conv_block, same_convlution
from models.auxiliary_funs import get_normalization3d, get_activation
from torch.nn import functional as F

from models.modules.MultimodalSegmentation.DualStream import DoubleConv as StdDoubleConv
from models.modules.MultimodalSegmentation.DomainSpecificNormalization import DoubleConv as DSBNDoubleConv

from models.modules.MultimodalSegmentation.DualStream import number_of_features_per_level
from models.modules.MultimodalSegmentation.DomainSpecificNormalization import OutConv, create_encoders, create_decoders, FeatureMapExtractor


class StdDoubleConvWithDomain(StdDoubleConv):
    def __init__(self, in_channels, out_channels, domains, in_encoder,
                 kernel_size=3, norm_type="batch", act_type="lrelu", **kwargs):
        super(StdDoubleConvWithDomain, self).__init__(in_channels, out_channels, in_encoder, kernel_size,
                                                      norm_type, act_type, **kwargs)

    def forward(self, x, domain=None):
        return super(StdDoubleConvWithDomain, self).forward(x)


class DsbnHalfUnet(nn.Module):
    def __init__(self, in_channels, out_channels, domains, down_basic_module=DSBNDoubleConv, up_basic_module=StdDoubleConvWithDomain,
                 f_maps=16, num_levels=5,
                 with_activation=True, final_sigmoid=True, norm_type="batch", act_type="lrelu",
                 conv_kernel_size=3, pool_kernel_size=2, interpolation=True, **kwargs):
        super(DsbnHalfUnet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        self.encoders = create_encoders(domains, in_channels, f_maps, down_basic_module, conv_kernel_size, pool_kernel_size, norm_type, act_type)
        self.decoders = create_decoders(domains, f_maps, up_basic_module, conv_kernel_size, interpolation, norm_type, act_type)
        self.outconv = OutConv(f_maps[0], out_channels, with_activation=with_activation, final_sigmoid=final_sigmoid)

    def forward(self, x, domain):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x, domain)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x, domain)

        x = self.outconv(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    from functools import partial
    from models.auxiliary_funs import print_model_parm_nums, print_model_parm_flops

    # from models.auxiliary_hookers import FeatureMapExtractor, FeatureGradientExtractor

    device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    net = DsbnHalfUnet(1, 1, ['target', 'source'], DSBNDoubleConv, StdDoubleConvWithDomain,
                       f_maps=16, num_levels=5, with_activation=False, final_sigmoid=True, interpolation=True,
                       norm_type="batch", act_type="lrelu").to(device)  # 4.1212M
    feature_extractor = FeatureMapExtractor(net, ['decoders.0.basic_module.conv2', 'decoders.1.basic_module.conv2'])
    # 'decoders.1.upsampling', 'decoders.1.basic_module.conv2', 'decoders.2.basic_module.conv2'
    print('--------------------------named_modules-------------------------------')
    for name, layer in net.named_modules():
        print(name, type(layer))
    print('------------------------named_children---------------------------------')
    for name, layer in net.named_children():
        print(name, type(layer))
    print('-------------------------named_parameters--------------------------------')
    for k, v in net.named_parameters():
        print(k, v.size())

        print(v.nelement())
    print('--------------------------named_buffers-------------------------------')
    for k, v in net.named_buffers():
        print(k, v.size())

    inputs = torch.rand((2, 1, 80, 112, 112), requires_grad=True).to(device)  # 64,96,96
    out = net(inputs, 'source')
    print(out.size())
    print('--------------------------single model-------------------------------')
    print_model_parm_nums(net)
    # print_model_parm_flops(net, inputs, need_idx=False, domain="source")  # 751.84G
    # print(net)

    feature_k, feature_l = feature_extractor.get_out_feature()
    print(feature_k.size(), feature_l.size())
    # feature = feature_extractor.get_out_feature()
    # for fea in feature:
    #     print(type(fea))
    #     print(fea.shape)





