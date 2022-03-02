import torch
import torch.nn as nn
import torch.nn.functional as F

from models.auxiliary_funs import make_one_hot
from utils.others.metrics import SoftMetrics
from torch.nn import CTCLoss


# [N,C,*]
class IOULoss(nn.Module):
    def __init__(self, class_weight=None, ignore_index=None, normalization=None, reduction='mean',
                 smooth=1., eps=1e-6):
        super(IOULoss, self).__init__()
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

    def basic_forward(self, output, target):
        N, C = output.shape[:2]
        output = output.contiguous().view(N, C, -1).float()
        target = target.contiguous().view(N, C, -1).float()
        intersect = (output * target).sum(-1)       # N C
        denominator = output.abs().sum(-1) + target.abs().sum(-1) - intersect
        if self.class_weight:
            intersect = intersect * self.class_weight  # NC * C
        return intersect, denominator

    def std_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        std_iou = (intersect.sum() + self.smooth) / (denominator.sum() + self.smooth + self.eps)
        return 1 - std_iou     # 1

    def generalized_forward(self, output, target):
        N, C = output.shape[:2]
        axis_order = (1, 0) + tuple(range(2, output.dim()))
        output = output.permute(axis_order).contiguous().view(C, -1)
        target = target.permute(axis_order).contiguous().view(C, -1)

        class_weight = 1. / (target.sum(-1) * target.sum(-1) + self.eps)    # C

        numerator = ((output*target).sum(-1) * class_weight)
        denominator = ((output.abs() + target.abs() - output*target).sum(-1)) * class_weight    # C

        if self.class_weight:
            numerator = numerator * self.class_weight

        iou = numerator.sum() / denominator.sum()
        return 1 - iou

    def all_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        all_iou = (intersect + self.smooth) / (denominator + self.smooth + self.eps)  # NC

        return 1 - all_iou

    def class_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        class_dice = (intersect.sum(dim=0) + self.smooth) / (denominator.sum(dim=0) + self.smooth + self.eps)
        class_loss = 1 - class_dice     # C

        return class_loss

    def batch_forward(self, output, target):
        intersect, denominator = self.basic_forward(output, target)
        batch_iou = (intersect.sum(dim=1) + self.smooth) / (denominator.sum(dim=1) + self.smooth + self.eps)
        batch_loss = 1 - batch_iou     # N

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
    input = torch.rand((5, 1, 32, 32, 32))
    model = nn.Conv3d(1, 4, 3, padding=1)
    target = torch.randint(0, 4, (5, 1, 32, 32, 32)).float()
    target = make_one_hot(target, num_classes=4, with_channel=True)
    criterion = IOULoss(ignore_index=[2, 3], reduction='mean')
    loss = criterion(model(input), target)
    loss.backward()
    print(loss.item())


if __name__ == '__main__':
    test()
