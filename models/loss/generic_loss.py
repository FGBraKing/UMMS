import torch
from torch import nn as nn
from models.auxiliary_funs import make_one_hot


class WeightedMixupLoss(nn.Module):
    def __init__(self, weights, criterions: nn.ModuleList, **kwargs):
        super(WeightedMixupLoss, self).__init__()
        self.weights = weights
        self.criterions = criterions

    def forward(self, output, target):
        assert len(self.weight) == len(self.criterions)
        loss = 0.
        for weight, criterions in zip(self.weights, self.criterions):
            loss += weight * criterions(output, target)
        return loss


class WeightedLoss:
    def __init__(self, weights, loss_func, weighted_dimension=1):
        self.weights = weights
        self.loss_func = loss_func
        self.weighted_dimension = weighted_dimension

    def __call__(self, input_data, target):
        weighted_length = input_data.shape[self.weighted_dimension]
        losses = torch.zeros(weighted_length)
        for index in range(weighted_length):
            x = input_data.select(dim=self.weighted_dimension, index=index)
            y = target.select(dim=self.weighted_dimension, index=index)
            losses[index] = self.loss_func(x, y)
        return torch.mean(self.weights * losses)


class WeightedLogLoss:
    def __init__(self, weights, gammas):
        self.weights = weights
        self.gammas = gammas

    def __call__(self, *losses, **kwargs):
        self.weights = torch.tensor(self.weights).to(losses[0].device)
        self.gammas = torch.tensor(self.gammas).to(losses[0].device)
        weighted_loss = torch.zeros(1).to(losses[0].device)
        for inx, loss in enumerate(losses):
            weighted_loss = weighted_loss + (torch.pow(-1 * loss.log(), self.gammas[inx])) * self.weights[inx]
        return weighted_loss


class IgnoreIndexLossWrapper:
    """
    Wrapper around loss functions which do not support 'ignore_index', e.g. BCELoss.
    Throws exception if the wrapped loss supports the 'ignore_index' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index

    def __call__(self, output, target):
        if target.dim() == output.dim()-1:
            target = make_one_hot(target, num_classes=output.size(1), ignore_index=self.ignore_index, with_channel=False)

        assert output.size() == target.size()

        valid_mask = target.clone().ne_(self.ignore_index)
        valid_mask.requires_grad = False

        masked_input = output * valid_mask
        masked_target = target * valid_mask
        return self.loss_criterion(masked_input, masked_target)


