import torch
from torch import nn
from torch.nn import functional as F
from models.auxiliary_funs import Activation


# from https://doi.org/10.1007/978-3-030-59719-1_19
def calculate_csa_matrix(features_l, features_k, mask=None, s_c=1):
    '''
    :param features_l: b,m,d0,h0,w0
    :param features_k: b,n,d1,h1,w1
    :param mask: b,1,d,h,w  the ground truth mask of size (h,w) for class c, (reshape to the size of feature map if necessary)
    :return:
    '''
    b0,m,d0,h0,w0 = features_l.size()
    b1,n,d1,h1,w1 = features_k.size()
    assert b0==b1 and d0==d1 and h0==h1 and w0==w1
    if mask is not None:
        assert mask.size(1)==1
        s_c = torch.sum(mask, dim=(1,2,3,4)).reshape(-1,1,1) # b,1,1
        features_l = features_l * F.interpolate(mask, (d0,h0,w0), mode='nearest', align_corners=True)
        features_k = features_k * F.interpolate(mask, (d1,h1,w1), mode='nearest', align_corners=True)

    features_l_norm = features_l.view(b0,m,-1) / torch.norm(features_l.view(b0,m,-1), dim=-1, keepdim=True)  # b,m,d*h*w
    features_k_norm = features_k.view(b1,n,-1) / torch.norm(features_k.view(b1,n,-1), dim=-1, keepdim=True)  # b,n,d*h*w
    csa = torch.bmm(features_l_norm, features_k_norm.transpose(1,2))    # b,m,n
    return csa / s_c
    # csa = F.cosine_similarity(features_l.view(b0,m,-1), features_k.view(b1,n,-1), dim=-1, eps=1e-8)


def get_csa_matrix(features_l, features_k, mask):
    bs = features_k.size(0)
    m = features_l.size(1)
    n = features_k.size(1)
    classes = mask.size(1)
    all_csa = torch.zeros(bs, classes, m, n)
    for c in range(classes):
        all_csa[:, c, ...] = calculate_csa_matrix(features_l, features_k, mask[:, c:c+1, ...])   # b,m,n
    return all_csa



