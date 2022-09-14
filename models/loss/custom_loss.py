import torch
import torch.nn.functional as F
from torch import nn as nn

from .distribution_based import WBCEWithLogitLoss, BinaryFocalLoss
from .region_based import BinaryDiceLoss, BinaryTverskyLoss
from .losses import l2_regularization


# [N, *]
class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs):

        super(CustomLoss, self).__init__()
        reduction = 'mean'
        # eps = 1e-7
        distribution_smooth = 0.1
        region_smooth = 10

        use_mixed_precision = kwargs.get('use_mixed_precision', False)
        eps = 5e-4 if use_mixed_precision else 1e-7

        self.w_region = 1.0       #
        self.w_distribution = 1.0

        self.pos_weight = 1.0   # Few samples

        self.gamma = 2          # hard sample

        self.alpha_fp = 2       # precision
        self.beta_fn = 1        # recall

        self.wbce = WBCEWithLogitLoss(weight=self.pos_weight,
                                      ignore_index=None, smooth=distribution_smooth, reduction=reduction, eps=eps)
        self.focal = BinaryFocalLoss(alpha=self.pos_weight, gamma=self.gamma,
                                     ignore_index=None, smooth=distribution_smooth, reduction=reduction, eps=eps)

        self.diceloss = BinaryDiceLoss(smooth=region_smooth, use_sigmoid=True, eps=eps, reduction=reduction)
        self.tverskyloss = BinaryTverskyLoss(self.alpha_fp, self.beta_fn,
                                             smooth=region_smooth, use_sigmoid=True, eps=eps, reduction=reduction)

    def forward(self, output, target):
        tversky_loss = self.tverskyloss(output, target)
        dice_loss = self.diceloss(output, target)
        # dice_loss = -torch.log(1 - dice_loss)

        wbce_loss = self.wbce(output, target)
        focal_loss = self.focal(output, target)

        loss = self.w_region * dice_loss + self.w_distribution * wbce_loss

        return {'dice': dice_loss,
                # 'tversky': tversky_loss,
                'wbce': wbce_loss,
                'focal': focal_loss}, loss


class RegularLoss(nn.Module):
    def __init__(self):
        super(RegularLoss, self).__init__()
        self.criterionL2 = l2_regularization

    def forward(self, paras):
        l2_regular = self.criterionL2(paras)
        return l2_regular


class CustomMultiModalLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        # 半精度最大的小于1的数是0.99951，因此精度设置为5e-4
        super(CustomMultiModalLoss, self).__init__()
        use_mixed_precision = kwargs.get('use_mixed_precision', False)
        reduction = 'mean'
        # eps = 6e-8 / 5e-4
        # region_eps = 5e-4 if use_mixed_precision else 1e-7
        # distribution_eps = 5e-4 if use_mixed_precision else 1e-7
        eps = 5e-4 if use_mixed_precision else 1e-7

        distribution_smooth = 0.1
        region_smooth = 10

        self.source_weight = 1.5
        self.target_weight = 1.
        self.prior_weight = 0.

        self.w_region = 1.0       #
        self.w_distribution = 1.5

        self.pos_weight = 1.0   # Few samples

        self.gamma = 2          # hard sample

        self.alpha_fp = 1       # precision
        self.beta_fn = 1        # recall
        # print('distribution_eps: ', distribution_eps, 'region_eps: ', region_eps)

        self.wbce = WBCEWithLogitLoss(weight=self.pos_weight,
                                      ignore_index=None, smooth=distribution_smooth, reduction=reduction, eps=eps)
        self.focal = BinaryFocalLoss(alpha=self.pos_weight, gamma=self.gamma,
                                     ignore_index=None, smooth=distribution_smooth, reduction=reduction, eps=eps)

        self.diceloss = BinaryDiceLoss(smooth=region_smooth, use_sigmoid=True, eps=eps, reduction=reduction)
        self.tverskyloss = BinaryTverskyLoss(self.alpha_fp, self.beta_fn,
                                             smooth=region_smooth, use_sigmoid=True, eps=eps, reduction=reduction)

        self.sizeloss = SizeConstrainedNormLoss(True, 0.02, eps=eps)

        self.sizelossV1 = SizeConstrainedNormLoss(True, 0.15, eps=eps)

    def base_forward(self, output, target, prefix='', sample_weights=None):
        tversky_loss = self.tverskyloss(output, target)
        dice_loss = self.diceloss(output, target)
        # dice_loss = -torch.log(1 - dice_loss)

        wbce_loss = self.wbce(output, target, sample_weights)
        focal_loss = self.focal(output, target)

        loss = self.w_region * dice_loss + self.w_distribution * wbce_loss

        return {prefix+'dice': dice_loss,
                # prefix+'tversky': tversky_loss,
                prefix+'wbce': wbce_loss,
                # prefix+'focal': focal_loss,
                prefix+'combo': loss}, loss

    def forward(self, s_out, s_aim, t_out, t_aim, source_sample_weights=None, target_sample_weights=None):
        source_dict, source_loss = self.base_forward(s_out, s_aim, 'source', source_sample_weights)
        target_dict, target_loss = self.base_forward(t_out, t_aim, 'target', target_sample_weights)
        size_loss = self.sizeloss(s_out, t_out)
        # size_loss = self.sizelossV1(s_out, t_out)
        total_loss = self.source_weight * source_loss + self.target_weight * target_loss + self.prior_weight * size_loss
        return {**source_dict, **target_dict, 'prior_loss:': 10000*size_loss}, total_loss


class SizeConstrainedLoss(nn.Module):
    def __init__(self, use_sigmoid=False, threshold=0.15, reduction="mean", eps=1e-7):
        super(SizeConstrainedLoss, self).__init__()
        self.bound = threshold
        self.reduction = reduction
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, s_pre, t_pre):
        assert s_pre.size(0) == t_pre.size(0), "output & target batch size don't match"
        if self.use_sigmoid:
            s_pre = F.sigmoid(s_pre)
            t_pre = F.sigmoid(t_pre)

        # s_pre = (s_pre > 0.5).float()
        # t_pre = (t_pre > 0.5).float()

        s_sum = s_pre.view(s_pre.size(0), -1).sum(-1)
        t_sum = t_pre.view(t_pre.size(0), -1).sum(-1)

        # print("s_sum: ", s_sum.detach().tolist(), "t_sum:", t_sum.detach().tolist())
        # ratio = ((torch.abs((s_sum+t_sum)*(s_sum-t_sum)/(s_sum*t_sum+self.eps))) / 2).float()
        ratio = ((torch.abs(s_sum/(t_sum+self.eps)-t_sum/(s_sum+self.eps))) / 2).float()
        # print('ratio: ', ratio.detach().tolist())
        # ratio[ratio < self.bound] = self.bound
        # loss = (ratio - self.bound).pow(2)
        ratio[ratio < self.bound] = 0
        loss = ratio.pow(2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        # return F.mse_loss(ratio, self.bound, reduction=self.reduction)


class SizeConstrainedAsymmetricLoss(nn.Module):
    def __init__(self, use_sigmoid=False, threshold=0.15, reduction="mean", eps=1e-7):
        super(SizeConstrainedAsymmetricLoss, self).__init__()
        self.bound = threshold
        self.reduction = reduction
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, s_pre, t_pre, uselabel=False):
        assert s_pre.size(0) == t_pre.size(0), "output & target batch size don't match"
        if uselabel:
            s_sum = s_pre
            t_pre = F.sigmoid(t_pre) if self.use_sigmoid else t_pre
            t_sum = t_pre.view(t_pre.size(0), -1).sum(-1)
        else:
            if self.use_sigmoid:
                s_pre = F.sigmoid(s_pre)
                t_pre = F.sigmoid(t_pre)

            # s_pre = (s_pre > 0.5).float()
            # t_pre = (t_pre > 0.5).float()

            s_sum = s_pre.view(s_pre.size(0), -1).sum(-1)
            t_sum = t_pre.view(t_pre.size(0), -1).sum(-1)

        # print("s_sum: ", s_sum.detach().tolist(), "t_sum:", t_sum.detach().tolist())
        ratio = torch.abs((s_sum-t_sum)/(s_sum+self.eps))
        # print('ratio: ', ratio.detach().tolist())
        # ratio[ratio < self.bound] = self.bound
        # loss = (ratio - self.bound).pow(2)
        ratio[ratio < self.bound] = 0
        loss = ratio.pow(2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        # return F.mse_loss(ratio, self.bound, reduction=self.reduction)


class SizeConstrainedNormLoss(nn.Module):
    def __init__(self, use_sigmoid=False, threshold=0.02, reduction="mean", eps=1e-7):
        super(SizeConstrainedNormLoss, self).__init__()
        self.bound = threshold
        self.reduction = reduction
        self.eps = eps
        self.use_sigmoid = use_sigmoid

    def forward(self, s_pre, t_pre):
        assert s_pre.size(0) == t_pre.size(0), "output & target batch size don't match"
        n,c,d,h,w = s_pre.size()
        if self.use_sigmoid:
            s_pre = F.sigmoid(s_pre)
            t_pre = F.sigmoid(t_pre)

        # s_pre = (s_pre > 0.5).float()
        # t_pre = (t_pre > 0.5).float()

        s_mean = s_pre.view(s_pre.size(0), -1).mean(-1)
        t_mean = t_pre.view(t_pre.size(0), -1).mean(-1)

        # F.l1_loss(s_mean, t_mean, reduction=self.reduction)
        # F.mse_loss(s_mean, t_mean, reduction=self.reduction)

        ratio = torch.abs(s_mean-t_mean)
        # print('ratio: ', (ratio.detach()*100).tolist(), 'bound: ', self.bound)

        ratio[ratio < self.bound] = 0
        loss = ratio.pow(2)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        # return F.mse_loss(ratio, self.bound, reduction=self.reduction)


class FeatureConstraine(nn.Module):
    def __init__(self, temperature=2.0,  eps=1e-7, **kwargs):
        super(FeatureConstraine, self).__init__()
        use_mixed_precision = kwargs.get('use_mixed_precision', False)

        self.temperature = temperature
        self.eps = 5e-4 if use_mixed_precision else eps

    def cal_pair_featureloss(self, source_feature, target_feature):
        assert source_feature.size() == target_feature.size()
        source_feature = F.relu(source_feature)
        target_feature = F.relu(target_feature)

        #   ## old try
        # source_prob = F.softmax(source_feature.sum([0, 2, 3, 4])/self.temperature, dim=0)
        # target_prob = F.softmax(target_feature.sum([0, 2, 3, 4])/self.temperature, dim=0)
        #
        # # KL divergence loss
        # loss = (torch.sum(source_prob * torch.log(source_prob / (target_prob + self.eps))) +
        #         torch.sum(target_prob * torch.log(target_prob / (source_prob + self.eps)))) / 2.0

        # ## new try
        source_vect_norm = source_feature.sum([2, 3, 4]) / (source_feature.sum([2, 3, 4]).norm(dim=-1, keepdim=True) + self.eps)
        target_vect_norm = target_feature.sum([2, 3, 4]) / (target_feature.sum([2, 3, 4]).norm(dim=-1, keepdim=True) + self.eps)
        sim = torch.sum(source_vect_norm*target_vect_norm, dim=-1).mean()
        return 1-sim

    def forward(self, *paired_features):
        cst_loss = 0
        num_pairs = len(paired_features)
        for pair_features in paired_features:
            loss = self.cal_pair_featureloss(*pair_features)
            cst_loss += loss

        cst_loss /= num_pairs
        return cst_loss


class EdgeFilter(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(EdgeFilter, self).__init__()
        self.use_sigmoid = use_sigmoid
        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]])
        sobel_filter_x = sobel_filter.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        sobel_filter_y = sobel_filter_x.transpose(3, 4)
        sobel_filter_z = sobel_filter_x.transpose(2, 4)

        self.register_buffer('sobel_filter_x', sobel_filter_x)
        self.register_buffer('sobel_filter_y', sobel_filter_y)
        self.register_buffer('sobel_filter_z', sobel_filter_z)

    def forward(self, x):
        if self.use_sigmoid:
            x = F.sigmoid(x)
        filters_x = F.conv3d(x, self.sobel_filter_x, stride=1, padding=(0, 1, 1))
        filters_y = F.conv3d(x, self.sobel_filter_y, stride=1, padding=(0, 1, 1))
        filters_z = F.conv3d(x, self.sobel_filter_z, stride=1, padding=(1, 1, 0))
        edge = torch.norm(torch.stack((filters_x, filters_y, filters_z), dim=0), dim=0)
        edge = torch.clamp(edge, 0, 1)
        return edge


class EdgeLoss(nn.Module):
    def __init__(self, use_sigmoid=True, reduction='mean'):
        super(EdgeLoss, self).__init__()
        kernel = torch.tensor([[[0,0,0],[0,1,0],[0,0,0]],
                               [[0,1,0],[1,0,1],[0,1,0]],
                               [[0,0,0],[0,1,0],[0,0,0]]])

        self.weight = kernel.view(1, 1, 3, 3, 3)
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self, predict, distance_map):
        if self.use_sigmoid:
            predict = F.sigmoid(predict)
        bs = predict.size(1)
        edge = F.conv3d(predict, self.weight, stride=1, padding=1)
        edge[edge>3]=0
        edge[edge>0]=1

        out_edge = edge * predict

        loss = (out_edge.view(bs, -1) * distance_map.view(bs, -1)).sum(-1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class EdgeLossV1(nn.Module):
    def __init__(self, use_sigmoid=True, reduction='mean'):
        super(EdgeLossV1, self).__init__()
        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]])

        self.sobel_filter_x = sobel_filter.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        self.sobel_filter_y = self.sobel_filter_x.transpose(3, 4)
        self.sobel_filter_z = self.sobel_filter_x.transpose(2, 4)
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def get_edge(self, volume):
        self.sobel_filter_x = self.sobel_filter_x.float().to(volume.device)
        self.sobel_filter_y = self.sobel_filter_y.float().to(volume.device)
        self.sobel_filter_z = self.sobel_filter_z.float().to(volume.device)
        filters_x = F.conv3d(volume, self.sobel_filter_x, stride=1, padding=(0, 1, 1))
        filters_y = F.conv3d(volume, self.sobel_filter_y, stride=1, padding=(0, 1, 1))
        filters_z = F.conv3d(volume, self.sobel_filter_z, stride=1, padding=(1, 1, 0))
        edge = torch.norm(torch.stack((filters_x, filters_y, filters_z), dim=0), dim=0)
        edge = torch.clamp(edge, 0, 1)
        return edge

    def forward(self, predict, distance_map):
        if self.use_sigmoid:
            predict = F.sigmoid(predict)
        bs = predict.size(1)
        edge = self.get_edge(predict)

        loss = (edge.view(bs, -1) * distance_map.view(bs, -1)).sum(-1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class EdgeLossV2(nn.Module):
    def __init__(self, use_sigmoid=True, reduction='mean'):
        super(EdgeLossV2, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.get_edge = EdgeFilter(use_sigmoid)

    def forward(self, predict, distance_map):
        bs = predict.size(1)
        edge = self.get_edge(predict)
        loss = (edge.view(bs, -1) * distance_map.view(bs, -1)).sum(-1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss


class EdgeLossV3(nn.Module):
    def __init__(self, reduction='mean'):
        super(EdgeLossV3, self).__init__()
        self.reduction = reduction

    def forward(self, edge, distance_map):
        bs = edge.size(1)
        loss = (edge.view(bs, -1) * distance_map.view(bs, -1)).mean(-1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError
        return loss
