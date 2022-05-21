import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# [N, *], # 二进制FL
class BinaryFocalLoss(nn.Module):
    # from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/seg_loss/focal_loss.py
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """
    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', smooth=0.1, eps=1e-7):
        super(BinaryFocalLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.smooth = smooth  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        pos_mask = (target == 1).float()  # q
        neg_mask = (target == 0).float()  # 1-q
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.eps, 1.0 - self.eps)        # avoid `nan` loss
        target = torch.clamp(target.float(), min=self.smooth, max=1.0 - self.smooth)  # soft label

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * target * torch.log(prob)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -neg_weight * (1.0 - target) * torch.log(1-prob)   # sigmoid(x) -1 是奇函数，sigmoid(-x)+sigmoid(x)=1
        loss = self.alpha * pos_loss + neg_loss     # (N, *)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


# [N C *] [N *]   多分类FL，不支持smooth_target
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean',
                 smooth=0, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')
        self.smooth = smooth
        self.eps = eps

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target.long()).view(-1) + self.eps  # avoid nan, pt，q*p,  [N, *, 1]
        logpt = torch.log(prob)

        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)

        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()
        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss


# [N C *] [N *], # 多分类FL，不支持ignore_index
class FocalLossV1(nn.Module):
    # taken from https://github.com/JunMa11/SegLoss/blob/master/test/nnUNetV2/loss_functions/focal_loss.py
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV1, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# #  from https://github.com/clcarwin/focal_loss_pytorch
# # N C *  N * , # 多分类FL，不支持smooth_target, 不支持ignore_index
# class FocalLoss(nn.Module):
#     '''
#     Binary classification
#     https://arxiv.org/pdf/1708.02002.pdf
#     '''
#     def __init__(self, gamma=2, alpha=None, reduction='none'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1-alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)    # N*H*W,1
#
#         logpt = F.log_softmax(input)  # , dim=1           log_p
#         logpt = logpt.gather(1, target.long()).view(-1)    # p*log(q), log_pt  N*H*W
#
#         pt = logpt.exp()
#         # 或者也可以通过把target变成独热编码，再与通过softmax的输出相乘，得到pt
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#                 self.alpha.to(input.device)
#             at = self.alpha.gather(0, target.data.view(-1))
#             # at = self.alpha[target.data.view(-1)]
#             logpt = logpt * torch.Tensor(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise Exception('Unexpected reduction {}'.format(self.reduction))

