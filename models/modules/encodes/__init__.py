import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------Encode ---------------------------------------
class DummyEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyEncoder, self).__init__()
        self.out_channels = (16, 32, 64, 128, 256)

    def forward(self, x):
        b, c, d, h, w = x.size()

        return (F.interpolate(x[:, :16, ...], scale_factor=16),
                F.interpolate(x[:, :32, ...], scale_factor=8),
                F.interpolate(x[:, :64, ...], scale_factor=4),
                F.interpolate(x[:, :128, ...], scale_factor=2),
                x)


def get_encoder(in_channels=1, depth=5, f_maps=16, **kwargs):
    kwargs.update(in_channels=in_channels)
    kwargs.update(depth=depth)
    kwargs.update(f_maps=f_maps)
    encoder = DummyEncoder(**kwargs)

    if 'output_stride' in kwargs.keys():
        output_stride = kwargs['output_stride']
    else:
        output_stride = 32

    if output_stride != 32 and hasattr(encoder, 'make_dilated'):
        encoder.make_dilated(output_stride)
    return encoder


