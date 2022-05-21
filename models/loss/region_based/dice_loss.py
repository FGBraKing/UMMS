import torch
import torch.nn as nn

from models.auxiliary_funs import make_one_hot
from utils.others.metrics import SoftMetrics


# [ *]
def diceloss(input, target, smooth=0., eps=1e-6):
    '''计算某特定类别的DICE系数
    :param input: 每个点的值为该点为某类的概率, [*], [D,H,W]
    :param target: 样本的真实概率分布，每个点的值为该点为某类的真实概率, [*], [D,H,W]
    :param smooth:
    :param eps
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
        self.loss = diceloss
        self.eps = eps

    def forward(self, input, target):
        loss = self.loss(input, target, self.smooth, self.eps)
        return loss


# [N, *],二分类的时候其实只需要计算一个类的dice，
class BinaryDiceLoss(nn.Module):
    # from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/tree/master/DiceLoss
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

    def __init__(self, use_sigmoid=False, smooth=1.0, eps=1e-7,
                 use_batch=True, ignore_index=None, reduction='mean'):
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
        # # 把=ignore_index的类的地方全部清零, 我认为这里没什么用，本来就没有计算背景类的dice
        # if self.ignore_index is not None:
        #     validmask = (target != self.ignore_index).float()
        #     output = output.float().mul(validmask)  # can not use inplace for bp
        #     target = target.float().mul(validmask)

        if self.use_batch:
            dim0 = output.shape[0]      # 每个data独自算dice， [N]
        else:
            dim0 = 1                    # 多个data拼在一起算dice

        output = output.contiguous().view(dim0, -1).float()
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth + self.eps

        loss = 1. - (num / den)  # [dim0]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# [N,C,*],    return [N, C]
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
                 reduction='mean', smooth=0., eps=1e-6):
        super(MutiClassDiceLoss, self).__init__()
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        else:
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
        self.smooth = smooth
        self.eps = eps

    # 返回intersect, denominator，[N,C]
    @staticmethod
    def basic_forward(output, target):
        N, C = output.shape[:2]
        output = output.contiguous().view(N, C, -1).float()
        target = target.contiguous().view(N, C, -1).float()
        intersect = (output * target).sum(-1)       # N C
        denominator = (output.abs() + target.abs()).sum(-1)
        return intersect, denominator

    def std_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        dice = (2 * intersect.sum() + self.smooth) / (denominator.sum() + self.smooth + self.eps)
        return 1 - dice

    # 按类加权累加，得到一个dice
    def generalized_forward(self, output, target):
        N, C = output.shape[:2]
        axis_order = (1, 0) + tuple(range(2, output.dim()))
        output = output.permute(axis_order).contiguous().view(C, -1)
        target = target.permute(axis_order).contiguous().view(C, -1)

        class_weight = 1. / (target.sum(-1) * target.sum(-1) + self.eps)    # C

        numerator = 2. * ((output*target).sum(-1) * class_weight)
        denominator = ((output.abs() + target.abs()).sum(-1)) * class_weight    # C

        dice = numerator.sum() / denominator.sum()
        return 1 - dice

    # 返回 [N,C]个dice_loss
    def all_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        dice = (2. * intersect + self.smooth) / (denominator + self.smooth + self.eps)  # NC

        dice_loss = 1 - dice
        if self.class_weight:
            dice_loss = dice_loss * self.class_weight  # NC * C
        return dice_loss

    # 返回 C 个dice
    def class_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        dice = (2. * intersect.sum(dim=0) + self.smooth) / (denominator.sum(dim=0) + self.smooth + self.eps)

        dice_loss = 1 - dice
        if self.class_weight:
            dice_loss = dice_loss * self.class_weight  # C * C
        return dice_loss

    # 返回 N 个dice
    def batch_forward(self, output, target):
        num_class = output.size(1)
        all_dice_loss = self.all_forward(output, target)
        return all_dice_loss.sum(1)/(num_class-len(self.ignore_index))

    def forward(self, output, target, forward_type='all'):
        if self.class_weight is not None:
            assert self.class_weight.shape[0] == target.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
        assert output.size() == target.size(),  'output & target shape do not match'    # [N,C,*]

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

        used_classes = output.size(1) - len(self.ignore_index)
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
            if forward_type == 'all' or forward_type == 'class':
                loss = loss.sum(1)/used_classes
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
