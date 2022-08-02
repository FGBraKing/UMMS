import torch
import torch.nn.functional as F
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

        self.pos_weight = 1.0   # Few samples

        self.gamma = 2          # hard sample

        self.alpha_fp = 2       # precision
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


class CustomMultiModalLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CustomMultiModalLoss, self).__init__()
        reduction = 'mean'
        eps = 1e-7
        distribution_smooth = 0.1
        region_smooth = 10

        self.source_weight = 1.
        self.target_weight = 1.

        self.w_region = 1.0       #
        self.w_distribution = 1.0

        self.pos_weight = 1.0   # Few samples

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

    def base_forward(self, output, target, prefix=''):
        tversky_loss = self.tverskyloss(output, target)
        dice_loss = self.diceloss(output, target)
        # dice_loss = -torch.log(1 - dice_loss)

        wbce_loss = self.wbce(output, target)
        focal_loss = self.focal(output, target)

        loss = self.w_region * dice_loss + self.w_distribution * wbce_loss

        return {prefix+'dice': dice_loss,
                prefix+'tversky': tversky_loss,
                prefix+'wbce': wbce_loss,
                prefix+'focal': focal_loss,
                prefix+'combo': loss}, loss

    def forward(self, s_out, s_aim, t_out, t_aim):
        source_dict, source_loss = self.base_forward(s_out, s_aim, 'source')
        target_dict, target_loss = self.base_forward(t_out, t_aim, 'target')
        total_loss = self.source_weight * source_loss + self.target_weight * target_loss
        return {**source_dict, **target_dict}, total_loss


class SizeConstrainedLoss(nn.Module):
    def __init__(self, use_sigmoid=False, threshold=0.15, reduction="mean", eps=1e-7):
        super(SizeConstrainedLoss, self).__init__()
        self.bound = threshold
        self.reduction = reduction
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, s_pre, t_pre):
        assert s_pre.size(0) == t_pre.size(0), "output & target batch size don't match"
        if self.use_sigmoid:
            s_pre = F.sigmoid(s_pre)
            t_pre = F.sigmoid(t_pre)

        # s_pre = (s_pre > 0.5).float()
        # t_pre = (t_pre > 0.5).float()

        s_sum = s_pre.view(s_pre.size(0), -1).sum(-1)
        t_sum = t_pre.view(t_pre.size(0), -1).sum(-1)

        print("s_sum: ", s_sum.detach().tolist(), "t_sum:", t_sum.detach().tolist())
        ratio = ((torch.abs((s_sum+t_sum)*(s_sum-t_sum)/(s_sum*t_sum+self.eps))) / 2).float()
        # ratio = ((torch.abs(s_sum/(t_sum+self.eps)-t_sum/(s_sum+self.eps))) / 2).float()
        print('ratio: ', ratio.detach().tolist())
        ratio[ratio < self.bound] = self.bound
        loss = (ratio - self.bound).pow(2)
        # loss = ratio - self.bound
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        # return F.mse_loss(ratio, self.bound, reduction=self.reduction)


class SizeConstrainedAsymmetricLoss(nn.Module):
    def __init__(self, use_sigmoid=False, threshold=0.15, reduction="mean", eps=1e-7):
        super(SizeConstrainedAsymmetricLoss, self).__init__()
        self.bound = threshold
        self.reduction = reduction
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, s_pre, t_pre):
        assert s_pre.size(0) == t_pre.size(0), "output & target batch size don't match"
        if self.use_sigmoid:
            s_pre = F.sigmoid(s_pre)
            t_pre = F.sigmoid(t_pre)

        # s_pre = (s_pre > 0.5).float()
        # t_pre = (t_pre > 0.5).float()

        s_sum = s_pre.view(s_pre.size(0), -1).sum(-1)
        t_sum = t_pre.view(t_pre.size(0), -1).sum(-1)

        print("s_sum: ", s_sum.detach().tolist(), "t_sum:", t_sum.detach().tolist())
        ratio = torch.abs((s_sum-t_sum)/(s_sum+self.eps))
        print('ratio: ', ratio.detach().tolist())
        ratio[ratio < self.bound] = self.bound
        loss = (ratio - self.bound).pow(2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        # return F.mse_loss(ratio, self.bound, reduction=self.reduction)


class SizeConstrainedNormLoss(nn.Module):
    def __init__(self, use_sigmoid=False, threshold=0.15, reduction="mean", eps=1e-7):
        super(SizeConstrainedNormLoss, self).__init__()
        self.bound = threshold
        self.reduction = reduction
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, s_pre, t_pre):
        assert s_pre.size(0) == t_pre.size(0), "output & target batch size don't match"
        if self.use_sigmoid:
            s_pre = F.sigmoid(s_pre)
            t_pre = F.sigmoid(t_pre)

        # s_pre = (s_pre > 0.5).float()
        # t_pre = (t_pre > 0.5).float()

        s_sum = s_pre.view(s_pre.size(0), -1).sum(-1)
        t_sum = t_pre.view(t_pre.size(0), -1).sum(-1)

        volume_size = s_pre.nelement()

        print("s_sum: ", s_sum.detach().tolist(), "t_sum:", t_sum.detach().tolist())
        ratio = torch.abs((s_sum-t_sum)/volume_size)
        print('ratio: ', ratio.detach().tolist())
        ratio[ratio < self.bound] = 0
        loss = ratio.pow(2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        # return F.mse_loss(ratio, self.bound, reduction=self.reduction)


