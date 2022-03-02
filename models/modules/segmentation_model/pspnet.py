import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union
from models.modules.encodes import get_encoder
from models.modules.blocks.heads import SegmentationHead, ClassificationHead
from models.modules.segmentation_model.basic_segmentation import SegmentationModel
from models.modules.blocks.blocks3d import ConvBnRelu


# -----------------------------Decode------------------
class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(pool_size),
            ConvBnRelu(in_channels, out_channels, 1, 1, 0, bias=not use_bathcnorm, use_bn=use_bathcnorm)
        )

    def forward(self, x):
        d, h, w = x.size(2), x.size(3), x.size(4)
        x = self.pool(x)
        x = F.interpolate(x, size=(d, h, w), mode='trilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            use_batchnorm=True,
            out_channels=512,
            dropout=0.2,
    ):
        super().__init__()

        self.psp = PSPModule(in_channels=encoder_channels[-1], sizes=(1, 2, 3, 6), use_bathcnorm=use_batchnorm,)

        self.conv = ConvBnRelu(encoder_channels[-1] * 2, out_channels, kernel_size=1, use_bn=use_batchnorm,)

        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x


# -----------------------------Model------------------

class PSPNet(SegmentationModel):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation. Consist of
    *encoder* and *Spatial Pyramid* (decoder). Spatial Pyramid build on top of encoder and does not
    use "fine-features" (features of high spatial resolution). PSPNet can be used for multiclass segmentation
    of high resolution images, however it is not good for detecting small objects and producing accurate, pixel-level mask.
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        psp_out_channels: A number of filters in Spatial Pyramid
        psp_use_batchnorm: If **True**, BatchNorm3d layer between Conv3d and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        psp_dropout: Spatial dropout rate in [0, 1) used in Spatial Pyramid
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **PSPNet**
    .. _PSPNet:
        https://arxiv.org/abs/1612.01105
    """

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: Optional[str] = "imagenet",
            encoder_depth: int = 3,
            psp_out_channels: int = 512,
            psp_use_batchnorm: bool = True,
            psp_dropout: float = 0.2,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "psp-{}".format(encoder_name)
        self.initialize()


if __name__ == '__main__':
    inputs = torch.randn(2,256,8,8,8)
    net = PSPNet(psp_out_channels=64, activation='sigmoid')
    out = net(inputs)
    print(out.size())








