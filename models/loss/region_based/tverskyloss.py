import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable


# from https://github.com/Hsuxu/Loss_ToolBox-PyTorch

# [N, 1,(d,) h, w]  or [N, *]
class BinaryTverskyLoss(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean',
                 use_sigmoid=False, smooth=10., eps=1e-6):
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
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

        self.eps = eps

        self.sigmoid = use_sigmoid

    def forward(self, output, target):

        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        # get the logit
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        batch_size = output.size(0)
        bg_target = 1 - target
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = output.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1)
        bg_target = bg_target.contiguous().view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * bg_target, 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth + self.eps)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


# [N C *]
class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.5, beta=0.5, weights=None,
                 reduction='mean', is_logit=True, ignore_index=None):
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
        self.weights = weights

        self.reduction = reduction
        self.is_logit = is_logit
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

        self.bin_tcersky = BinaryTverskyLoss(alpha=self.alpha, beta=self.beta, reduction=self.reduction,
                                             use_sigmoid=False, smooth=0)

    def forward(self, inputs, targets):

        num_class = inputs.size(1)
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
        assert inputs.size() == targets.size(),  'inputs & target shape do not match'

        if self.is_logit:
            inputs = torch.softmax(inputs, dim=1)   # NC*

        weight_losses = 0.0
        for idx in range(num_class):
            if idx in self.ignore_index:
                continue
            loss_idx = self.bin_tcersky(inputs[:, idx], targets[:, idx])    # N or 1
            if self.weights is not None:
                loss_idx *= self.weights[idx]
            weight_losses += loss_idx

        return weight_losses/(num_class - len(self.ignore_index))


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, epsilon=1e-6, reduction='mean', smooth=10,
                 weights=None, ignore_index=None, is_logit=True, is_binary=True):
        super(FocalTverskyLoss, self).__init__()
        self.bin_tversky = BinaryTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index,
                                             reduction=reduction, use_sigmoid=False, smooth=smooth)
        self.muti_tversky = MultiTverskyLoss(alpha=alpha, beta=beta, ignore_index=ignore_index,
                                             reduction=reduction, is_logit=is_logit, weights=weights)

        self.gamma = gamma
        self.epsilon = epsilon
        self.is_binary = is_binary

    def forward(self, output, target):
        if self.is_binary:
            tv_loss = self.bin_tversky(output, target)
        else:
            tv_loss = self.muti_tversky(output, target)
        ft_loss = torch.pow((1-tv_loss), 1/self.gamma)
        return ft_loss


class TverskyLoss(nn.Module):
    '''
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    '''
    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean', smooth=1., normalization='sigmoid'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s
        if normalization == 'sigmoid':
            self.normal = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normal = nn.Softmax(dim=1)
        elif normalization == 'None':
            self.normal = None
        else:
            raise ValueError('normalization have to be one of [sigmoid softmax None]')

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)
        if self.normal is not None:
            output = self.normal(output)
        if output.dim() > 2:
            output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W => N,C,H*W
            output = output.transpose(1, 2)    # N,C,H*W => N,H*W,C
            output = output.contiguous().view(-1, output.size(2))   # N,H*W,C => N*H*W,C
        target = torch.zeros(output.size()).scatter_(1, target.view(-1, 1), 1)

        TP = output * target
        FN = (1-output) * target
        FP = output * (1-target)

        TL = 1. - (TP.sum(1))/((TP.sum(1))+self.alpha*FN.sum(1)+self.beta*FP.sum(1)+self.smooth)
        if self.reduction == 'none':
            loss = TL
        elif self.reduction == 'sum':
            loss = torch.sum(TL)
        else:
            loss = torch.mean(TL)
        return loss
