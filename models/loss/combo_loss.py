from torch import nn as nn

from models.loss.distribution_based.cross_entropy import WBCEWithLogitLoss
from models.loss.region_based.dice_loss import BinaryDiceLoss


# from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/dice_loss.py
# [N, *]
class WBCE_DiceLoss(nn.Module):
    def __init__(self, alpha=1.0, weight=1.0, ignore_index=None, reduction='mean',
                 bce_smooth=0.01, bdc_smooth=1.0, eps=1e-6):
        """
        combination of Weight Binary Cross Entropy and Binary Dice Loss
        Args:
            @param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            @param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            @param alpha: weight between WBCE('Weight Binary Cross Entropy') and binary dice, apply on WBCE
        Shapes:
            output: A tensor of shape [N, *] without sigmoid activation function applied
            target: A tensor of shape same with output
        """
        super(WBCE_DiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        # assert 0 <= alpha <= 1, '`alpha` should in [0,1]'
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dice = BinaryDiceLoss(use_batch=True, use_sigmoid=True,
                                   ignore_index=ignore_index, reduction=reduction,
                                   smooth=bdc_smooth, eps=eps)
        self.wbce = WBCEWithLogitLoss(weight=weight,
                                      ignore_index=ignore_index, reduction=reduction,
                                      smooth=bce_smooth, eps=eps)
        self.dice_loss = None
        self.wbce_loss = None

    def forward(self, output, target):
        self.dice_loss = self.dice(output, target)
        # self.dice_loss = -torch.log(1 - self.dice_loss)
        self.wbce_loss = self.wbce(output, target)
        loss = self.alpha * self.wbce_loss + self.dice_loss
        return loss


