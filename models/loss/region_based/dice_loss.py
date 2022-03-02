import torch
import torch.nn as nn
import torch.nn.functional as F

from models.auxiliary_funs import make_one_hot
from utils.others.metrics import SoftMetrics


# [ *]
def dice_loss(input, target, smooth=1., eps=1e-6):
    '''计算某特定类别的DICE系数
    :param input: 每个点的值为该点为某类的概率, [*], [D,H,W]
    :param target: 样本的真实概率分布，每个点的值为该点为某类的真实概率, [*], [D,H,W]
    :param smooth:
    :return:
    '''
    TP, FN, TN, FP = SoftMetrics.get_basic_metrics(input, target)

    intersection = TP
    denominator = 2*TP + FP + FN

    return 1 - ((2. * intersection + smooth) / (denominator + smooth + eps))


# [ *]
class DiceLoss(nn.Module):
    '''
    the input must be the feature from sigmoid
    '''
    def __init__(self, smooth=1.0, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.register_buffer('smooth', torch.tensor(smooth))
        self.loss = dice_loss
        self.eps = eps

    def forward(self, input, target):
        loss = self.loss(input, target, self.smooth, self.eps)
        return loss


# # from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/tree/master/DiceLoss
# [N, *]
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean',
                 use_batch=True, use_sigmoid=False, smooth=1., eps=1e-6):
        super(BinaryDiceLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_batch = use_batch     # treat a large map when True
        self.use_sigmoid = use_sigmoid
        self.eps = eps
        self.smooth = smooth

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        # get the logit
        if self.use_sigmoid:
            output = torch.sigmoid(output)

        # 把=ignore_index的类的地方全部清零
        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.float().mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        if self.use_batch:
            dim0 = output.shape[0]      # N
        else:
            dim0 = 1

        output = output.contiguous().view(dim0, -1).float()
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth + self.eps

        loss = 1 - (num / den)  # [dim0]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# [N,C,*]
class MutiClassDiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, class_weight=None, ignore_index=None, normalization=None,
                 reduction='mean', smooth=1., eps=1e-6):
        super(MutiClassDiceLoss, self).__init__()
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

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
        self.smooth = smooth
        self.eps = eps

        # self.kwargs = kwargs
        #
        # self.bin_dice = BinaryDiceLoss(use_sigmoid=False, use_batch=True,
        #                                smooth=self.smoooth, reduction=self.reduction, **self.kwargs)
        # self.origin_dice = DiceLoss(smooth=self.smooth)

    def basic_forward(self, output, target):
        N, C = output.shape[:2]
        output = output.contiguous().view(N, C, -1).float()
        target = target.contiguous().view(N, C, -1).float()
        intersect = (output * target).sum(-1)       # N C
        denominator = (output.abs() + target.abs()).sum(-1)
        if self.class_weight:
            intersect = intersect * self.class_weight  # NC * C
        return intersect, denominator

    def std_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        std_dice = (2. * intersect.sum() + self.smooth) / (denominator.sum() + self.smooth + self.eps)
        return 1 - std_dice     # 1

    def generalized_forward(self, output, target):
        N, C = output.shape[:2]
        axis_order = (1, 0) + tuple(range(2, output.dim()))
        output = output.permute(axis_order).contiguous().view(C, -1)
        target = target.permute(axis_order).contiguous().view(C, -1)

        class_weight = 1. / (target.sum(-1) * target.sum(-1) + self.eps)    # C

        numerator = 2. * ((output*target).sum(-1) * class_weight)
        denominator = ((output.abs() + target.abs()).sum(-1)) * class_weight    # C

        if self.class_weight:
            numerator = numerator * self.class_weight

        dice = numerator.sum() / denominator.sum()
        return 1 - dice

    def all_forward(self, output, target):
        # version2
        intersect, denominator = self.basic_forward(output, target)
        all_dice = (2. * intersect + self.smooth) / (denominator + self.smooth + self.eps)  # NC

        # version1
        # total_loss = 0
        # for i in range(target.shape[1]):
        #     if i not in self.ignore_index:
        #         dice_l = self.bin_dice(output[:, i], target[:, i])      # N*1 or 1
        #         if self.weight is not None:
        #             dice_l = dice_l * self.weights[i]
        #         total_loss = total_loss + dice_l
        # loss = total_loss / (target.size(1) - len(self.ignore_index))
        return 1 - all_dice

    def class_forward(self, output, target):
        # version4
        intersect, denominator = self.basic_forward(output, target)
        class_dice = (2. * intersect.sum(dim=0) + self.smooth) / (denominator.sum(dim=0) + self.smooth + self.eps)
        class_loss = 1 - class_dice     # C

        # version 3
        # b_c_loss = self.all_forward(output, target)     # N C
        # class_loss = torch.mean(b_c_loss, dim=0)        # C

        # version2
        # N, C = output.shape[:2]
        # axis_order = (1, 0) + tuple(range(2, output.dim()))
        # output = output.permute(axis_order).contiguous().view(C, -1)
        # target = target.permute(axis_order).contiguous().view(C, -1)
        # intersect = (output * target).sum(-1)       # C
        # denominator = (output.abs() + target.abs()).sum(-1)
        # if self.weight:
        #     intersect = intersect * self.weight
        # class_loss = 1 - (2. * intersect + self.smooth) / (denominator + self.smooth + self.eps)

        # version1
        # class_loss = []
        # for i in range(target.shape[1]):
        #     if i not in self.ignore_index:
        #         dice_loss_c = self.bin_dice(output[:, i], target[:, i])
        #         if self.reduction == 'none':
        #             dice_loss_c = dice_loss_c.mean()
        #         if self.weight is not None:
        #             dice_loss_c *= self.weights[i]
        #         class_loss.append(dice_loss_c)
        #     else:
        #         class_loss.append(-1)
        return class_loss

    def batch_forward(self, output, target):
        # version4
        intersect, denominator = self.basic_forward(output, target)
        batch_dice = (2. * intersect.sum(dim=1) + self.smooth) / (denominator.sum(dim=1) + self.smooth + self.eps)
        batch_loss = 1 - batch_dice     # N

        # version3
        # b_c_loss = self.all_forward(output, target)
        # batch_loss = b_c_loss.sum(dim=-1) / (output.size(1) - len(self.ignore_index))   # N

        # version2
        # N, C = output.shape[:2]
        # output = output.contiguous().view(N, C, -1)
        # target = target.contiguous().view(N, C, -1).float()
        # intersect = (output * target).sum(-1)       # N C
        # denominator = (output.abs() + target.abs()).sum(-1)
        # if self.weight:
        #     intersect = intersect * self.weight
        # b_c_loss = 1 - (2. * intersect + self.smooth) / (denominator + self.smooth + self.eps)
        # batch_loss = b_c_loss.sum(-1) / (output.size(1) - len(self.ignore_index))

        # version1
        # batch_loss = []
        # for n in range(target.shape[0]):
        #     total_loss = 0
        #     for c in range(target.shape[1]):
        #         if c not in self.ignore_index:
        #             dice_l = self.origin_dice(output[n, c], target[n, c])
        #             if self.weight is not None:
        #                 dice_l = dice_l * self.weight[c]
        #             total_loss += dice_l
        #     total_loss = total_loss / (target.size(1) - len(self.ignore_index))
        #     batch_loss.append(total_loss)
        return batch_loss

    def forward(self, output, target, forward_type='std'):
        if self.class_weight is not None:
            assert self.class_weight.shape[0] == target.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
        assert output.size() == target.size(),  'output & target shape do not match'

        if self.normalization:
            # output = F.softmax(output, dim=1)
            output = self.normalization(output)

        if self.ignore_index is not None:
            for ig_ind in self.ignore_index:
                output[:, ig_ind] = 0
                target[:, ig_ind] = 0
                # valid_mask = target.ne(ig_ind).float()
                # output = torch.mul(output, valid_mask)  # can not use inplace for bp
                # target = torch.mul(target.float(), valid_mask)

        if forward_type == 'class':
            loss = self.class_forward(output, target)
        elif forward_type == 'batch':
            loss = self.batch_forward(output, target)
        elif forward_type == 'std':
            loss = self.std_forward(output, target)
        elif forward_type == 'all':
            loss = self.all_forward(output, target)
        elif forward_type == 'generalized':
            loss = self.generalized_forward(output, target)
        else:
            raise Exception('Unexpected forward_type {}'.format(forward_type))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def test():
    input = torch.rand((3, 1, 32, 32, 32))
    model = nn.Conv3d(1, 4, 3, padding=1)
    target = torch.randint(0, 4, (3, 1, 32, 32, 32)).float()
    target = make_one_hot(target, num_classes=4, with_channel=True)
    criterion = MutiClassDiceLoss(ignore_index=[2, 3], reduction='mean')
    loss = criterion(model(input), target)
    loss.backward()
    print(loss.item())

    # input = torch.zeros((1, 2, 32, 32, 32))
    # input[:, 0, ...] = 1
    # target = torch.ones((1, 1, 32, 32, 32)).long()
    # target_one_hot = make_one_hot(target, num_classes=2)
    # # print(target_one_hot.size())
    # criterion = DiceLoss()
    # loss = criterion(input, target_one_hot)
    # print(loss.item())


if __name__ == '__main__':
    test()
