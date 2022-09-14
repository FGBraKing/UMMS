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
                    'custom', 'custom_regular', 'custom_multimodal',
                    'prior', 'prior_asymmetric', 'prior_norm', 'prior_feature']


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
    eps = kwargs.get('eps', 1e-7)
    smooth = kwargs.get('smooth', 1.0)
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 2)
    gamma = kwargs.get('gamma', 2)
    weight = kwargs.get('weight', torch.tensor(1.0))

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
        gamma_neg = kwargs.get('gamma_neg', 4)
        gamma_pos = kwargs.get('gamma_pos', 1)
        clip = kwargs.get('clip', 0.05)
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
        activate = kwargs.get('activate', 'sigmoid')
        return MultiTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index, reduction=reduction,
                                smooth=smooth, normalization=activate)
    elif name == 'BinaryTversky':
        return BinaryTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index, reduction=reduction,
                                 use_sigmoid=True, smooth=smooth, eps=eps)
    elif name == 'MultiTversky':
        return MultiTverskyLoss(alpha=alpha, beta=beta, weights=weight,
                                reduction=reduction, is_logit=True, ignore_index=ignore_index)
    elif name == 'combo':
        bce_smooth = kwargs.get('bce_smooth', 0.01)
        return WBCE_DiceLoss(alpha=alpha, weight=weight, ignore_index=ignore_index, reduction=reduction,
                             bce_smooth=bce_smooth, bdc_smooth=smooth, eps=eps)
    elif name == 'bdc':
        use_sigmoid = kwargs.get('use_sigmoid', True)
        return BinaryDiceLoss(ignore_index=ignore_index, reduction=reduction,
                              use_batch=True, use_sigmoid=use_sigmoid, smooth=smooth, eps=eps)
    elif name == 'custom':
        return CustomLoss(**kwargs)
    elif name == 'custom_regular':
        return RegularLoss(**kwargs)
    elif name == "custom_multimodal":
        return CustomMultiModalLoss(**kwargs)
    elif name == 'prior':
        use_sigmoid = kwargs.get('use_sigmoid', True)
        prior_threshold = kwargs.get('prior_threshold', 0.10)
        return SizeConstrainedLoss(use_sigmoid=use_sigmoid, threshold=prior_threshold, reduction=reduction)
    elif name == 'prior_asymmetric':
        use_sigmoid = kwargs.get('use_sigmoid', True)
        prior_threshold = kwargs.get('prior_threshold', 0.10)
        return SizeConstrainedAsymmetricLoss(use_sigmoid=use_sigmoid, threshold=prior_threshold, reduction=reduction)
    elif name == 'prior_norm':
        use_sigmoid = kwargs.get('use_sigmoid', True)
        prior_threshold = kwargs.get('prior_threshold', 0.01)
        return SizeConstrainedNormLoss(use_sigmoid=use_sigmoid, threshold=prior_threshold, reduction=reduction)
    elif name == 'prior_feature':
        temperature = kwargs.get('temperature', 2.0)
        return FeatureConstraine(temperature=temperature, eps=eps, **kwargs)
    else:
        activate = kwargs.get('activate', 'softmax')
        return MutiClassDiceLoss(class_weight=weight, ignore_index=ignore_index, normalization=activate,
                                 reduction=reduction, smooth=smooth, eps=eps)
