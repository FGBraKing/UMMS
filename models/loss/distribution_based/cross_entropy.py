import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.auxiliary_funs import make_one_hot
# 常见需要实现的内容，ignore_index，类权重、标签平滑、


def ce(output, target):
    return -target * torch.log(output)


# [N, 1, *] or [N, *]  target:N *,   加权的二进制交叉熵， 带有smooth项和eps项， useful
class WBCEWithLogitLoss(nn.Module):
    """
    Weighted Binary Cross Entropy.
    `WBCE(p,t)=-β*t*log(p)-(1-t)*log(1-p)`
    To decrease the number of false negatives, set β>1.
    To decrease the number of false positives, set β<1.
    Args:
            @param weight: positive sample weight
        Shapes：
            output: A tensor of shape [N, 1,(d,), h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
    """

    def __init__(self, weight=1.0, ignore_index=None, reduction='mean', eps=1e-7, smooth=0.1):
        super(WBCEWithLogitLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.ignore_index = ignore_index
        self.weight = float(weight)
        self.reduction = reduction
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target, sample_weights=None):
        # 这里的target是(0,1)
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"

        batch_size = output.size(0)
        output = output.contiguous().view(batch_size, -1).float()
        target = target.contiguous().view(batch_size, -1).float()

        # 忽略某类转化为忽略某类的样本
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
        else:
            valid_mask = torch.ones_like(target, dtype=torch.float32)

        if sample_weights is not None:
            sample_weights = torch.tensor(sample_weights)
            sample_weights = sample_weights.contiguous().view(batch_size, -1).float()
            valid_mask = sample_weights * valid_mask

        output = torch.sigmoid(output)
        output = torch.clamp(output, min=self.eps, max=1.0 - self.eps)              # avoid `nan` loss
        target = torch.clamp(target, min=self.smooth, max=1.0 - self.smooth)        # soft label

        pos_bce = -target.mul(torch.log(output))
        neg_bce = -(1.0 - target).mul(torch.log(1.0 - output))

        loss = (self.weight * pos_bce + neg_bce) * valid_mask       # (N, *)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


# [N,C,*], [N, C, *] custom, useful, 带有像素点权重的加权交叉熵
class MutiClassCrossEntropyLoss(nn.Module):
    '''
    变成one-hot编码的形式后，只计算每一类的前景。
    '''
    def __init__(self, class_weight=None, ignore_index=None, normalization='softmax', reduction='mean',
                 eps=1e-7, smooth=0.1):
        super(MutiClassCrossEntropyLoss, self).__init__()
        self.class_weight = class_weight

        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = None

        self.reduction = reduction
        self.eps = eps
        self.smooth = smooth

    def forward(self, output, target, weights=None):
        # output: [N, C, ...]; target: [N, C, ...]; weights: [N, 1, ...]
        assert output.size() == target.size(),  'output & target shape do not match'
        if self.class_weight is not None:
            assert self.class_weight.shape[0] == target.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.class_weight.shape[0])

        if self.normalization:
            output = self.normalization(output)

        N, C = output.shape[:2]
        # output = output.contiguous().view(N, C, -1).float()
        # target = target.contiguous().view(N, C, -1).float()

        # avoid `nan` loss
        output = torch.clamp(output.float(), min=self.eps, max=1.0 - self.eps)
        # soft label
        target = torch.clamp(target.float(), min=self.smooth/(C-1), max=1.0 - self.smooth)

        mask = torch.ones_like(output, dtype=torch.float32)      # (N, C, ...)
        if self.class_weight is not None:
            assert len(self.class_weight) == C
            for cls in range(C):
                mask[:, cls] = self.class_weight[cls]

        # 忽略某类转化为忽略某类的样本
        if self.ignore_index is not None:
            for ig_ind in self.ignore_index:
                mask[:, ig_ind] = 0

        if weights is not None:
            weights = torch.Tensor(weights)
            weights = weights.expand_as(output)                  # （N, C, ...）
            mask = mask.mul(weights)

        loss = torch.sum(-target * torch.log(output) * mask, dim=1)   # N, *

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    @staticmethod
    def _class_weights(inputs):
        # normalize the input first
        inputs = F.softmax(inputs, _stacklevel=5)

        dims_sum = (0,) + tuple(range(2, inputs.dim()))

        nominator = (1. - inputs).sum(dim=dims_sum)  # , keepdim=True
        denominator = inputs.sum(dim=dims_sum)
        class_weights = torch.Tensor(nominator / denominator, requires_grad=False)  # C
        return class_weights


# #  x: N C *  target: N C *   weights: N *,   带有像素点权重的加权交叉熵
# class PixelWiseCrossEntropyLoss(nn.Module):
#     def __init__(self, class_weights=None, ignore_index=None):
#         super(PixelWiseCrossEntropyLoss, self).__init__()
#         self.register_buffer('class_weights', class_weights)
#         self.ignore_index = ignore_index
#         self.log_softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, output, target, weights=None):
#         # normalize the input
#         log_p = self.log_softmax(output)
#
#         # mask ignore_index if present
#         if self.ignore_index is not None:
#             mask = target.ne(self.ignore_index).float()
#             log_p = log_p * mask
#             target = target.float() * mask
#
#         # apply class weights
#         if self.class_weights is None:
#             class_weights = torch.ones(output.size(1)).float()
#         else:
#             class_weights = torch.Tensor(self.class_weights)
#
#         new_shape = (1, output.size(1)) + (1,) * (output.dim()-2)
#         class_weights = class_weights.view(*new_shape)
#
#         if weights is not None:
#             # expand weights
#             weights = weights.unsqueeze(0)
#             weights = weights.expand_as(output)     # C -> NC*
#             # add class_weights to each channel
#         else:
#             weights = torch.zeros_like(output)
#
#         # compute the losses
#         result = -(weights + class_weights) * target * log_p
#         # average the losses
#         return result.mean()


# # x: NC *   target: N *,   带有类权重的交叉熵函数
# class WeightedCrossEntropyLoss(nn.Module):
#     """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
#     """
#
#     def __init__(self, weight=None, ignore_index=-1, reduction='mean'):
#         super(WeightedCrossEntropyLoss, self).__init__()
#         self.register_buffer('weight', weight)
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#
#     def forward(self, inputs, target):
#         class_weights = self._class_weights(inputs)
#         if self.weight is not None:
#             weight = torch.Tensor(self.weight, requires_grad=False)
#             class_weights = class_weights * weight
#         return F.cross_entropy(inputs, target, weight=class_weights,
#                                ignore_index=self.ignore_index, reduction=self.reduction)
#
#     @staticmethod
#     def _class_weights(inputs):
#         # normalize the input first
#         inputs = F.softmax(inputs, _stacklevel=5)
#
#         dims_sum = (0,)+tuple(range(2, inputs.dim()))
#
#         nominator = (1. - inputs).sum(dim=dims_sum)         # , keepdim=True
#         denominator = inputs.sum(dim=dims_sum)
#         class_weights = torch.Tensor(nominator / denominator, requires_grad=False)  # C
#         return class_weights


# # from https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py
# #  x: NC *  target: N C *, C>2
# # 支持smooth label，不支持类权重和focal项
# class SoftTargetCrossEntropy(nn.Module):
#
#     def __init__(self):
#         super(SoftTargetCrossEntropy, self).__init__()
#
#     def forward(self, x, target):
#         loss = torch.sum(-target * F.log_softmax(x, dim=1), dim=1)      # N
#         return loss.mean()


# from https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py
# x: NC *   target: N *, 带有smooth_loss的交叉熵，有点像滑动平均。 不支持smooth label
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=1)      # log_p, NC
        nll_loss = -logprobs.gather(dim=1, index=target.long().unsqueeze(1))    # -q*log_p， N,1,*
        nll_loss = nll_loss.squeeze(1)      # N,*
        smooth_loss = -logprobs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# # from https://github.com/MontaEllis/Pytorch-Medical-Segmentation/blob/master/loss_function.py
# # input: NC *   target: N *, 不支持smooth label
# def cross_entropy_3D(input, target, weight=None, size_average=True):
#
#     n, c, h, w, s = input.size()
#     log_p = F.log_softmax(input, dim=1)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)  # nhws,c
#     target = target.view(target.numel())    # nhws, 1
#     loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
#     if size_average:
#         loss /= float(target.numel())
#     return loss


