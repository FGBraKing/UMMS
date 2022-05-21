import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable


# [N, 1,(d,) h, w]  or [N, *]
class BinaryTverskyLoss(nn.Module):
    # from https://github.com/Hsuxu/Loss_ToolBox-PyTorch
    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean',
                 use_batch=True, use_sigmoid=True, smooth=10., eps=1e-6):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

        self.ignore_index = ignore_index        # 后期没有使用
        self.reduction = reduction
        self.use_batch = use_batch
        self.use_sigmoid = use_sigmoid
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        # get the logit
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        if self.use_batch:
            dim0 = output.shape[0]
        else:
            dim0 = 1

        output = output.contiguous().view(dim0, -1).float()
        target = target.contiguous().view(dim0, -1).float()

        P_G = torch.sum(output * target, 1)          # TP, dim0
        P_NG = torch.sum(output * (1 - target), 1)   # FP
        NP_G = torch.sum((1 - output) * target, 1)   # FN

        tversky_index = (P_G + self.smooth) / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth + self.eps)

        loss = 1. - tversky_index    # [dim0]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# [N C *]
class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """
    def __init__(self, alpha=0.5, beta=0.5, weights=None, ignore_index=None, is_logit=True,
                 reduction='mean', smooth=0., eps=1e-6):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_weight = weights

        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

        self.is_logit = is_logit
        self.reduction = reduction
        self.smooth = smooth
        self.eps = eps
        self.bin_tcersky = BinaryTverskyLoss(alpha=self.alpha, beta=self.beta, reduction=self.reduction,
                                             use_sigmoid=False, smooth=0)

    @staticmethod
    def basic_forward(output, target):
        N, C = output.shape[:2]
        output = output.contiguous().view(N, C, -1).float()
        target = target.contiguous().view(N, C, -1).float()

        TP = (output * target).sum(-1)  # N C
        FP = (output * (1-target)).sum(-1)
        FN = ((1-output) * target).sum(-1)
        return TP, FP, FN

    def std_forward(self, output, target):
        TP, FP, FN = self.basic_forward(output, target)

        numerator = TP.sum()
        denominator = TP.sum() + self.alpha*FP.sum() + self.beta*FN.sum()
        tversky_index = (numerator + self.smooth) / (denominator + self.smooth + self.eps)
        return 1 - tversky_index

    def generalized_forward(self, output, target):
        N, C = output.shape[:2]
        axis_order = (1, 0) + tuple(range(2, output.dim()))
        output = output.permute(axis_order).contiguous().view(C, -1)
        target = target.permute(axis_order).contiguous().view(C, -1)

        class_weight = 1. / (target.sum(-1) * target.sum(-1) + self.eps)    # C

        tp = (output*target).sum(-1)
        fp = (output * (1-target)).sum(-1)
        fn = ((1-output) * target).sum(-1)

        numerator = tp * class_weight
        denominator = (tp+self.alpha*fp+self.beta*fn) * class_weight    # C

        tversky_index = numerator.sum() / denominator.sum()
        return 1 - tversky_index

    def all_forward(self, output, target):
        TP, FP, FN = self.basic_forward(output, target)

        numerator = TP
        denominator = TP + self.alpha * FP + self.beta * FN
        tversky_index = (numerator + self.smooth) / (denominator + self.smooth + self.eps)
        tversky_loss = 1 - tversky_index
        if self.class_weight:
            tversky_loss = tversky_loss * self.class_weight
        return tversky_loss

    def class_forward(self, output, target):
        TP, FP, FN = self.basic_forward(output, target)

        numerator = TP.sum(0)
        denominator = TP.sum(0) + self.alpha * FP.sum(0) + self.beta * FN.sum(0)
        tversky_index = (numerator + self.smooth) / (denominator + self.smooth + self.eps)
        tversky_loss = 1 - tversky_index
        if self.class_weight:
            tversky_loss = tversky_loss * self.class_weight  # C * C
        return tversky_loss

    def batch_forward(self, output, target):
        num_class = output.size(1)
        tversky_loss = self.all_forward(output, target)
        return tversky_loss.sum(1)/(num_class-len(self.ignore_index))

    def forward(self, inputs, targets, forward_type='all'):

        if self.class_weight is not None:
            assert len(self.weights) == inputs.size(1), 'number of classes should be equal to length of weights '
        assert inputs.size() == targets.size(),  'inputs & target shape do not match'

        if self.is_logit:
            inputs = torch.softmax(inputs, dim=1)   # NC*

        if self.ignore_index is not None:
            for ig_ind in self.ignore_index:
                inputs[:, ig_ind] = 0
                targets[:, ig_ind] = 0

        used_classes = inputs.size(1) - len(self.ignore_index)
        if forward_type == 'class':
            loss = self.class_forward(inputs, targets)
        elif forward_type == 'batch':
            loss = self.batch_forward(inputs, targets)
        elif forward_type == 'std':
            loss = self.std_forward(inputs, targets)
        elif forward_type == 'all':
            loss = self.all_forward(inputs, targets)
        elif forward_type == 'generalized':
            loss = self.generalized_forward(inputs, targets)
        else:
            raise Exception('Unexpected forward_type {}'.format(forward_type))

        if self.reduction == 'mean':
            if forward_type == 'all' or forward_type == 'class':
                loss = loss.sum(1) / used_classes
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, epsilon=1e-6, reduction='mean', smooth=10,
                 weights=None, ignore_index=None, is_logit=True, is_binary=True):
        super(FocalTverskyLoss, self).__init__()
        assert 1 <= gamma <= 3, "gamma should be 1<=gamma <=3"
        self.bin_tversky = BinaryTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index,
                                             reduction=reduction, use_sigmoid=True, smooth=smooth, eps=epsilon)
        self.muti_tversky = MultiTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index,
                                             reduction=reduction, is_logit=is_logit, weights=weights,
                                             smooth=0., eps=epsilon)

        self.gamma = gamma
        self.epsilon = epsilon
        self.is_binary = is_binary

    def forward(self, output, target):
        if self.is_binary:
            tv_loss = self.bin_tversky(output, target)
        else:
            tv_loss = self.muti_tversky(output, target)
        # ft_loss = torch.pow((1-tv_loss), 1/self.gamma)
        ft_loss = torch.pow(tv_loss, 1/self.gamma)
        return ft_loss


# class TverskyLoss(nn.Module):
#     '''
#     Notes:
#         alpha = beta = 0.5 => dice coeff
#         alpha = beta = 1 => tanimoto coeff
#         alpha + beta = 1 => F beta coeff
#     '''
#     def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean', smooth=1., normalization='sigmoid'):
#         super(TverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.ignore_index = ignore_index
#         self.smooth = smooth
#         self.reduction = reduction
#         s = self.beta + self.alpha
#         if s != 1:
#             self.beta = self.beta / s
#             self.alpha = self.alpha / s
#         if normalization == 'sigmoid':
#             self.normal = nn.Sigmoid()
#         elif normalization == 'softmax':
#             self.normal = nn.Softmax(dim=1)
#         elif normalization == 'None':
#             self.normal = None
#         else:
#             raise ValueError('normalization have to be one of [sigmoid softmax None]')
#
#     def forward(self, output, target):
#         assert output.shape[0] == target.shape[0], "output & target batch size don't match"
#         if self.ignore_index is not None:
#             valid_mask = (target != self.ignore_index).float()
#             output = output.float().mul(valid_mask)  # can not use inplace for bp
#             target = target.float().mul(valid_mask)
#         if self.normal is not None:
#             output = self.normal(output)
#         if output.dim() > 2:
#             output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W => N,C,H*W
#             output = output.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             output = output.contiguous().view(-1, output.size(2))   # N,H*W,C => N*H*W,C
#         target = torch.zeros(output.size()).scatter_(1, target.view(-1, 1), 1)
#
#         TP = output * target
#         FN = (1-output) * target
#         FP = output * (1-target)
#
#         TL = 1. - (TP.sum(1))/((TP.sum(1))+self.alpha*FN.sum(1)+self.beta*FP.sum(1)+self.smooth)
#         if self.reduction == 'none':
#             loss = TL
#         elif self.reduction == 'sum':
#             loss = torch.sum(TL)
#         else:
#             loss = torch.mean(TL)
#         return loss
