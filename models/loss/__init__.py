import torch
from torch import nn

from .distribution_based.asymmetric_loss import AsymmetricLossOptimized, AsymmetricLossMultiLabel
from .distribution_based.cross_entropy import *
from .distribution_based.focal_loss import BinaryFocalLoss, FocalLoss
from .distribution_based.jsd import JsdCrossEntropy
from .distribution_based.others import *

from .region_based.dice_loss import BinaryDiceLoss, MutiClassDiceLoss
from .region_based.iou_loss import IOULoss
from .region_based.lovasz_loss import LovaszSoftmax
from .region_based.tverskyloss import BinaryTverskyLoss, MultiTverskyLoss, FocalTverskyLoss

from .generic_loss import *
from .combo_loss import *
from .custom_loss import *

SUPPORTED_LOSSES = ['bdc', 'dc', 'bce', 'ce', 'wce', 'pce', 'asymmetric', 'b_focal', 'focal', 'jsd', 'l1', 'l2', 'mse',
                    'lovasz', 'BinaryTversky', 'MultiTversky', 'tversky', 'combo', 'others',
                    'custom', 'custom_regular', 'custom_multimodal']


# --------------------------------------------------------CUSTOM------------------------------------------------
def get_loss_criterion(name, ignore_index=None, reduction='mean', **kwargs):
    """
    :param name:
    :param ignore_index:
    :param reduction:
    :param kwargs: eps, smooth, alpha, weight
    :return:
    """
    # name ignore_index reduction
    # eps smooth alpha beta gamma weight
    # (sample_weight)   (gamma_neg gamma_pos clip)  (num_splits)  (activate)  (bce_smooth)
    assert name in SUPPORTED_LOSSES, f'Invalid loss: {name}'
    if 'eps' in kwargs.keys():
        eps = kwargs['eps']
    else:
        eps = 1e-7
    if 'smooth' in kwargs.keys():
        smooth = kwargs['smooth']
    else:
        smooth = 1.0
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 1.0
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    else:
        beta = 2
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    else:
        gamma = 2
    if 'weight' in kwargs.keys():
        weight = kwargs['weight']
    else:
        weight = torch.tensor(1.0)

    if name == 'bce':
        if ignore_index is None:
            if 'sample_weight' in kwargs.keys():
                sample_weight = torch.Tensor(kwargs['sample_weight'])
            else:
                sample_weight = None
            return nn.BCEWithLogitsLoss(weight=sample_weight, reduction=reduction, pos_weight=torch.tensor(weight))
        else:
            return WBCEWithLogitLoss(weight=weight, ignore_index=ignore_index, reduction=reduction,
                                     eps=eps, smooth=smooth)  # gppd
            # return IgnoreIndexLossWrapper(nn.BCEWithLogitsLoss(), ignore_index=ignore_index)
    elif name == 'ce':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    elif name == 'wce':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        # return WeightedCrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        return MutiClassCrossEntropyLoss(class_weight=weight, ignore_index=ignore_index, reduction=reduction)
    # elif name == 'pce':
    #     return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'l1':
        return nn.L1Loss(reduction=reduction)
    elif name == 'l2' or name == 'mse':
        return nn.MSELoss(reduction=reduction)
    elif name == 'asymmetric':
        if 'gamma_neg' in kwargs.keys():
            gamma_neg = kwargs['gamma_neg']
        else:
            gamma_neg = 4
        if 'gamma_pos' in kwargs.keys():
            gamma_pos = kwargs['gamma_pos']
        else:
            gamma_pos = 1
        if 'clip' in kwargs.keys():
            clip = kwargs['clip']
        else:
            clip = 0.05
        return AsymmetricLossMultiLabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
                                        reduction=reduction, eps=eps)
    elif name == 'focal':
        return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
    elif name == 'b_focal':
        #  3 2
        return BinaryFocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index, reduction=reduction, smooth=smooth)
    elif name == 'jsd':
        if 'num_splits' in kwargs.keys():
            num_splits = kwargs['num_splits']
        else:
            num_splits = 4
        return JsdCrossEntropy(num_splits=num_splits, alpha=alpha, smoothing=smooth)
    elif name == 'iou':
        if 'activate' in kwargs.keys():
            activate = kwargs['activate']
        else:
            activate = 'softmax'
        return IOULoss(class_weight=weight, ignore_index=ignore_index, normalization=activate, reduction=reduction,
                       smooth=smooth, eps=eps)
    elif name == 'lovasz':
        return LovaszSoftmax(reduction=reduction)
    elif name == 'tversky':
        if 'activate' in kwargs.keys():
            activate = kwargs['activate']
        else:
            activate = 'sigmoid'
        return MultiTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index, reduction=reduction,
                                smooth=smooth, normalization=activate)
    elif name == 'BinaryTversky':
        return BinaryTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index, reduction=reduction,
                                 use_sigmoid=True, smooth=smooth, eps=eps)
    elif name == 'MultiTversky':
        return MultiTverskyLoss(alpha=alpha, beta=beta, weights=weight,
                                reduction=reduction, is_logit=True, ignore_index=ignore_index)
    elif name == 'combo':
        if 'bce_smooth' in kwargs.keys():
            bce_smooth = kwargs['bce_smooth']
        else:
            bce_smooth = 0.01
        return WBCE_DiceLoss(alpha=alpha, weight=weight, ignore_index=ignore_index, reduction=reduction,
                             bce_smooth=bce_smooth, bdc_smooth=smooth, eps=eps)
    elif name == 'bdc':
        if 'use_sigmoid' in kwargs.keys():
            use_sigmoid = kwargs['use_sigmoid']
        else:
            use_sigmoid = True

        return BinaryDiceLoss(ignore_index=ignore_index, reduction=reduction,
                              use_batch=True, use_sigmoid=use_sigmoid, smooth=smooth, eps=eps)

    elif name == 'custom':
        return CustomLoss()

    elif name == 'custom_regular':
        return RegularLoss()
    elif name == "custom_multimodal":
        return CustomMultiModalLoss()
    else:
        if 'activate' in kwargs.keys():
            activate = kwargs['activate']
        else:
            activate = 'softmax'
        return MutiClassDiceLoss(class_weight=weight, ignore_index=ignore_index, normalization=activate,
                                 reduction=reduction, smooth=smooth, eps=eps)
