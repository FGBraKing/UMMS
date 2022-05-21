from torch import nn as nn

from .distribution_based import WBCEWithLogitLoss, BinaryFocalLoss
from .region_based import BinaryDiceLoss, BinaryTverskyLoss
from .losses import l2_regularization


# [N, *]
class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs):

        super(CustomLoss, self).__init__()
        reduction = 'mean'
        eps = 1e-7
        distribution_smooth = 0.1
        region_smooth = 10

        self.w_region = 1.0       #
        self.w_distribution = 1.0

        self.pos_weight = 2.0   # Few samples

        self.gamma = 2          # hard sample

        self.alpha_fp = 1       # precision
        self.beta_fn = 1        # recall

        self.wbce = WBCEWithLogitLoss(weight=self.pos_weight,
                                      ignore_index=None, smooth=distribution_smooth, reduction=reduction, eps=eps)
        self.focal = BinaryFocalLoss(alpha=self.pos_weight, gamma=self.gamma,
                                     ignore_index=None, smooth=distribution_smooth, reduction=reduction, eps=eps)

        self.diceloss = BinaryDiceLoss(smooth=region_smooth, use_sigmoid=True, eps=eps, reduction=reduction)
        self.tverskyloss = BinaryTverskyLoss(self.alpha_fp, self.beta_fn,
                                             smooth=region_smooth, use_sigmoid=True, eps=eps, reduction=reduction)

    def forward(self, output, target):
        tversky_loss = self.tverskyloss(output, target)
        dice_loss = self.diceloss(output, target)
        # dice_loss = -torch.log(1 - dice_loss)

        wbce_loss = self.wbce(output, target)
        focal_loss = self.focal(output, target)

        loss = self.w_region * dice_loss + self.w_distribution * wbce_loss

        return {'dice': dice_loss,
                'tversky': tversky_loss,
                'wbce': wbce_loss,
                'focal': focal_loss}, loss


class RegularLoss(nn.Module):
    def __init__(self):
        super(RegularLoss, self).__init__()
        self.criterionL2 = l2_regularization

    def forward(self, paras):
        l2_regular = self.criterionL2(paras)
        return l2_regular
