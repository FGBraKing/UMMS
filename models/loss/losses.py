import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import l1_loss, mse_loss

from .region_based.dice_loss import diceloss


# -------------------------------------------------------Function--------------------------------------------------
# I don't know from where
def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class KDLoss(nn.Module):
    def __init__(self, n_class, temperature=2.0,  eps=1e-6):
        super(KDLoss, self).__init__()
        self.n_class = n_class
        self.temperature = temperature
        self.eps = eps

    def _cal_soft_prob(self, logits, mask):
        if mask.ndim == logits.ndim:
            p_mask = mask.repeat([1, self.n_class, 1, 1, 1])
        elif mask.ndim == logits.ndim - 1:
            p_mask = mask.unsqueeze(1).repeat([1, self.n_class, 1, 1, 1])
        else:
            raise ValueError
        logits_mask_out = logits * p_mask
        logits_avg = torch.sum(logits_mask_out, [0, 2, 3, 4]) / (torch.sum(mask) + self.eps)  # C*1 (A 交 B/A)
        if self.n_class >= 2:
            soft_prob = F.softmax(torch.tensor(logits_avg/self.temperature),  dim=1)
        else:
            logit_avg_neg = - logits_avg
            soft_prob_pos = F.sigmoid(torch.tensor(logits_avg/self.temperature))
            soft_prob_neg = F.sigmoid(torch.tensor(logits_avg/self.temperature))
            soft_prob = torch.stact([soft_prob_pos, soft_prob_neg])
        return soft_prob

    def forward(self, source_logits, source_gt, target_logits, target_gt):
        # source_logits source_gt target_logits target_gt : n,C,h,w
        kd_loss = 0.0

        if self.n_class == 1:
            s_soft_prob = self._cal_soft_prob(source_logits, source_gt[:, 0, :, :, :])
            t_soft_prob = self._cal_soft_prob(target_logits, target_gt[:, 0, :, :, :])
            s_soft_prob_neg = self._cal_soft_prob(-source_logits, 1-source_gt[:, 0, :, :, :])
            t_soft_prob_neg = self._cal_soft_prob(-target_logits, 1-target_gt[:, 0, :, :, :])
            pos_loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob/t_soft_prob)) +
                        torch.sum(t_soft_prob * torch.log(t_soft_prob/s_soft_prob))) / 2.0
            neg_loss = (torch.sum(s_soft_prob_neg * torch.log(s_soft_prob_neg/t_soft_prob_neg)) +
                        torch.sum(t_soft_prob_neg * torch.log(t_soft_prob_neg/s_soft_prob_neg))) / 2.0
            kd_loss = (pos_loss + neg_loss) / 2

        for i in range(self.n_class):
            s_soft_prob = self._cal_soft_prob(source_logits, source_gt[:, i:i+1, :, :, :])
            t_soft_prob = self._cal_soft_prob(target_logits, target_gt[:, i:i+1, :, :, :])

            # ## KL divergence loss
            loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob/t_soft_prob)) +
                    torch.sum(t_soft_prob * torch.log(t_soft_prob/s_soft_prob))) / 2.0

            kd_loss += loss

        kd_loss = kd_loss / self.n_class

        return kd_loss


class CSALoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CSALoss, self).__init__()
        self.reduction = reduction

    def _cal_csa_matrix(self, features_l, features_k, mask=None, s_c=1):
        '''
        :param features_l: b,m,d0,h0,w0
        :param features_k: b,n,d1,h1,w1
        :param mask: b,1,d,h,w  the ground truth mask of size (h,w) for class c, (reshape to the size of feature map if necessary)
        :return:
        '''
        b0,m,d0,h0,w0 = features_l.size()
        b1,n,d1,h1,w1 = features_k.size()
        assert b0 == b1 and d0 == d1 and h0 == h1 and w0 == w1
        if mask is not None:
            assert mask.size(1) == 1
            s_c = torch.sum(mask, dim=(1, 2, 3, 4)).reshape(-1, 1, 1)  # b,1,1
            features_l = features_l * F.interpolate(mask, (d0, h0, w0), mode='nearest', align_corners=True)
            features_k = features_k * F.interpolate(mask, (d1, h1, w1), mode='nearest', align_corners=True)
        torch.linalg.norm()
        features_l_norm = features_l.view(b0, m, -1) / torch.norm(features_l.view(b0, m, -1), dim=-1, keepdim=True)  # b,m,d*h*w
        features_k_norm = features_k.view(b1, n, -1) / torch.norm(features_k.view(b1, n, -1), dim=-1, keepdim=True)  # b,n,d*h*w
        # torch.linalg.norm==torch.norm
        # 文章中似乎用的是cos求相似度，所以前面先求了单位向量
        csa = torch.bmm(features_l_norm, features_k_norm.transpose(1, 2))    # b,m,n
        return csa / s_c

    def forward(self,
                source_features_l, source_features_k, source_mask,
                target_features_l, target_features_k, target_mask):
        classes = source_mask.size(1)
        csa_loss = 0.0
        for i in range(classes):
            source_csa = self._cal_csa_matrix(source_features_l, source_features_k, source_mask[:, i:i+1, ...])
            target_csa = self._cal_csa_matrix(target_features_l, target_features_k, target_mask[:, i:i+1, ...])
            loss = F.mse_loss(source_csa, target_csa, reduction=self.reduction)
            csa_loss += loss
        csa_loss = csa_loss / classes
        return csa_loss


def l2_regularization(parameters):
    l2_norm = torch.sum(torch.Tensor([torch.sum(torch.pow(parameter, 2))/2
                                      for parameter in parameters if parameter.requires_grad]))
    return l2_norm


# maybe from MUNIT
def kl_loss(mu, logvar, N):
    return (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)


def vae_loss(predicted_x, mu, logvar, x, recon_loss=mse_loss, divergence_loss=kl_loss, recon_weight=1, kl_weight=1):
    loss_recon = recon_loss(predicted_x, x)
    loss_kl = divergence_loss(mu, logvar, x.numel()/x.shape[0])
    return recon_weight * loss_recon + kl_weight * loss_kl


def vae_dice_loss(predicted, mu, logvar, target, loss=diceloss, divergence_loss=kl_loss, weight=1, kl_weight=1):
    return vae_loss(predicted_x=predicted, mu=mu, logvar=logvar, x=target, recon_loss=loss,
                    divergence_loss=divergence_loss, recon_weight=weight, kl_weight=kl_weight)


def vae_l1_loss(predicted, mu, logvar, target, loss=l1_loss, divergence_loss=kl_loss, weight=1, kl_weight=1):
    return vae_loss(predicted_x=predicted, mu=mu, logvar=logvar, x=target, recon_loss=loss,
                    divergence_loss=divergence_loss, recon_weight=weight, kl_weight=kl_weight)


def regularized_loss(predicted_y, predicted_x, x, y, pred_loss=l1_loss, decoder_loss=mse_loss, decoder_weight=0.1):
    return pred_loss(predicted_y, y) + decoder_weight * decoder_loss(predicted_x, x)


def variational_regularized_loss(predicted, vae_x, mu, logvar, x, y, pred_loss=l1_loss, decoder_loss=mse_loss,
                                 vae_weight=0.1, kl_weight=0.1):
    loss_pred = pred_loss(predicted, y)
    loss_vae = decoder_loss(vae_x, x)
    N = x.numel()/x.shape[0]
    loss_kl = (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return loss_pred + (vae_weight * loss_vae) + (kl_weight * loss_kl)


# -------------------------------------------------------Class-------------------------------------------------
# copy from CycleGAN
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            loss = None
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        # torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# # Clip weights of discriminator
# for p in discriminator.parameters():
#     p.data.clamp_(-opt.clip_value, opt.clip_value)
