import torch.nn.functional as F

from torch.nn import CTCLoss, NLLLoss, SmoothL1Loss
from torch.nn import L1Loss, MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, KLDivLoss


# nn.L1Loss
def l1(input, target):
    # return torch.abs(input-target)
    return F.l1_loss(input, target, reduction='none')


# nn.MSELoss
def l2(input, target):
    # return torch.pow((input-target), 2)
    return F.mse_loss(input, target, reduction='none')
