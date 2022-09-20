import os
import torch
import logging
import itertools
import warnings
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp
import torch.nn.functional as F
from functools import partial
from types import SimpleNamespace
from collections import OrderedDict, defaultdict
from models.modules.MultimodalSegmentation.AdversarialDA import MCMDA, FeatureMapExtractor, FeatureDiscriminator, MaskDiscriminator, GANLoss
from models.modules.MultimodalSegmentation.AdversarialDA import cal_gradient_penalty, cal_gradient_penalty_feature
from models.modules.MultimodalSegmentation.DsbnHalf import DsbnHalfUnet
from models.networks.base_model import BaseModel
from models.loss import losses, get_loss_criterion
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from models.auxiliary_funs import get_init_func, get_activation
from utils.others.metrics import BinaryMetrics, SoftMetrics
from utils.others.distributed_utils import reduce_mean
from utils.others.utils import print_numpy
from configs.excess_config import ex_config


ddp_logger = logging.getLogger('ddp_logger')


def define_model(opt, device, domains=None):
    assert not(opt.DDP and opt.DP)

    net = MCMDA(in_channels=opt.input_nc,
                out_channels=opt.output_nc,
                domains=domains,
                f_maps=opt.init_channel_number,
                num_levels=5,
                with_activation=False,
                final_sigmoid=True,
                interpolation=opt.up_interpolate,
                norm_type="batch",
                act_type="lrelu").to(device)

    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    net.apply(init_func)

    if opt.SyncBatchNorm and opt.DDP:
        ddp_logger.warning('using torch.nn.SyncBatchNorm.convert_sync_batchnorm')
        # only single gpu per process is currently supported
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    else:
        net = net.to(device)

    if opt.DDP:
        ddp_logger.warning('using nn.parallel.DistributedDataParallel')
        # 使用DDP前，模型一定要进行初始化
        assert(torch.distributed.is_available())
        net = nn.parallel.DistributedDataParallel(module=net,
                                                  device_ids=[opt.local_gpu],  # 猜测填多个的时候每个进程都相当于DP
                                                  output_device=opt.local_gpu)
    elif opt.DP:
        ddp_logger.warning('using nn.parallel.DataParallel')
        # 必须先to(device)，再用DP封装
        assert(torch.cuda.is_available())
        net = nn.parallel.DataParallel(module=net,
                                       device_ids=opt.gpu_ids,
                                       output_device=opt.gpu_ids[0])    # 默认都用0号
        ddp_logger.warning('ending to use nn.parallel.DataParallel')
    else:
        ddp_logger.warning('It seems do not use the parallel mode')
    return net


def define_mask_discriminator(opt, device):
    assert not (opt.DDP and opt.DP)

    net = MaskDiscriminator(input_nc=1,
                            ndf=16,
                            n_layers=4,
                            norm_type="batch",
                            act_type="lrelu").to(device)

    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    net.apply(init_func)

    if opt.SyncBatchNorm and opt.DDP:
        ddp_logger.warning('using torch.nn.SyncBatchNorm.convert_sync_batchnorm')
        # only single gpu per process is currently supported
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    else:
        net = net.to(device)

    if opt.DDP:
        ddp_logger.warning('using nn.parallel.DistributedDataParallel')
        # 使用DDP前，模型一定要进行初始化
        assert (torch.distributed.is_available())
        net = nn.parallel.DistributedDataParallel(module=net,
                                                  device_ids=[opt.local_gpu],  # 猜测填多个的时候每个进程都相当于DP
                                                  output_device=opt.local_gpu)
    elif opt.DP:
        ddp_logger.warning('using nn.parallel.DataParallel')
        # 必须先to(device)，再用DP封装
        assert (torch.cuda.is_available())
        net = nn.parallel.DataParallel(module=net,
                                       device_ids=opt.gpu_ids,
                                       output_device=opt.gpu_ids[0])  # 默认都用0号
        ddp_logger.warning('ending to use nn.parallel.DataParallel')
    else:
        ddp_logger.warning('It seems do not use the parallel mode')
    return net


def define_feature_discriminator(opt, device):
    assert not (opt.DDP and opt.DP)

    net = FeatureDiscriminator(fmap=16,
                               mid_ch=16,
                               ndf=16,
                               n_layers=4,
                               norm_type="batch",
                               act_type="lrelu").to(device)

    init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
    net.apply(init_func)

    if opt.SyncBatchNorm and opt.DDP:
        ddp_logger.warning('using torch.nn.SyncBatchNorm.convert_sync_batchnorm')
        # only single gpu per process is currently supported
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    else:
        net = net.to(device)

    if opt.DDP:
        ddp_logger.warning('using nn.parallel.DistributedDataParallel')
        # 使用DDP前，模型一定要进行初始化
        assert (torch.distributed.is_available())
        net = nn.parallel.DistributedDataParallel(module=net,
                                                  device_ids=[opt.local_gpu],  # 猜测填多个的时候每个进程都相当于DP
                                                  output_device=opt.local_gpu)
    elif opt.DP:
        ddp_logger.warning('using nn.parallel.DataParallel')
        # 必须先to(device)，再用DP封装
        assert (torch.cuda.is_available())
        net = nn.parallel.DataParallel(module=net,
                                       device_ids=opt.gpu_ids,
                                       output_device=opt.gpu_ids[0])  # 默认都用0号
        ddp_logger.warning('ending to use nn.parallel.DataParallel')
    else:
        ddp_logger.warning('It seems do not use the parallel mode')
    return net


class AdversarialDAModel(BaseModel):
    def __init__(self, opt):
        super(AdversarialDAModel, self).__init__(opt)
        self.domains = ["source", "target"]
        self.opt.gan_mode = 'lsgan'
        # lsgan vanilla wgangp

        self.model_names = ['umms', 'df', 'dm']

        self.net_umms = define_model(opt, self.device, self.domains)
        self.net_df = define_feature_discriminator(opt, self.device)
        self.net_dm = define_mask_discriminator(opt, self.device)
        self.finally_activate = get_activation('sigmoid').to(self.device)

        self.set_requires_grad(self.net_umms, False)
        # self.net_umms.set_requires_grad_source_encoders(False)
        # self.net_umms.set_requires_grad_decoders(False)
        # self.net_umms.set_requires_grad_target_encoders(False)
        # self.net_umms.set_requires_grad_outconv(False)
        self.feature_extractor = FeatureMapExtractor(self.net_umms, ['source_encoders.4',
                                                                     'target_encoders.4',
                                                                     'decoders.0',
                                                                     'decoders.1',
                                                                     'decoders.2',
                                                                     'decoders.3'])
        self.logit_extractor = FeatureMapExtractor(self.net_umms, ['outconv.conv3d'])

        self.loss_names = ['regular', 'prior', 'gf', 'gm', 'umms', 'df', 'dm']   #
        # , 'dfpenalty', 'dmpenalty'

        if self.isTrain:
            self.criterionRoutine = get_loss_criterion(name='custom')
            self.criterionRegular = get_loss_criterion(name='custom_regular')
            # self.criterionPrior = get_loss_criterion(name='prior_norm', prior_threshold=opt.prior_threshold)
            self.criterionPrior = get_loss_criterion(name='prior_asymmetric', prior_threshold=opt.prior_threshold)
            self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)  # define GAN loss.
            # self.criterionDm = GANLoss('vanilla').to(self.device)  # define GAN loss.

            d_opt_name = 'rmsprop' if self.opt.gan_mode == 'wgangp' else 'adam'  # rmsprop
            optim_d_params = [{'params': self.net_df.parameters(), 'lr': 5e-5}, {'params': self.net_dm.parameters(), 'lr': 5e-5}]
            # itertools.chain(self.net_df.parameters(), self.net_dm.parameters())
            self.optimizer_D = create_optimizer_v2(optim_d_params,
                                                   opt=d_opt_name, lr=5e-5, weight_decay=0, momentum=0.9, eps=5e-4)
            self.optimizers.append(self.optimizer_D)

            optimizer_kwargs = {'eps': 5e-4, 'betas': (opt.optim_beta, 0.999)}
            if 'sgd' in opt.optimizer_name.lower():
                optimizer_kwargs.pop('betas', None)
            self.optimizer_S = create_optimizer_v2(self.net_umms.get_target_encoders_parameters(),
                                                   opt=opt.optimizer_name,
                                                   lr=opt.lr,
                                                   weight_decay=opt.weight_decay,
                                                   momentum=opt.momentum,
                                                   **optimizer_kwargs)
            self.optimizers.append(self.optimizer_S)

            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]

        # specify the images you want to save/display.
        self.visual_names = ['volume', 'predict', 'label']
        self.metric_names = ['DC', 'recall', 'precision', 'ravd']
        # 'hd' 'hd95' 'assd' 'asd'  'specificity', 'accuracy', 'roisize'

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-6)

        self.source_volume = None
        self.target_volume = None
        self.source_label = None
        self.target_label = None
        self.source_predict = None
        self.target_predict = None
        self.spacing = None
        self.label_ratio = None

        self.source_features = []
        self.target_features = []
        self.source_logits = []
        self.target_logits = []

        self.loss_df = 0
        self.loss_dm = 0
        self.loss_umms = 0
        self.loss_regular = 0
        self.loss_prior = 0
        self.loss_gf = 0
        self.loss_gm = 0
        self.loss_dfpenalty = 0
        self.loss_dmpenalty = 0

        self.metric_dict_source = None
        self.metric_dict_target = None

        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_umms.no_sync if opt.DDP else contextlib.nullcontext
        self.scaler_D = torch.cuda.amp.GradScaler()
        self.scaler_S = torch.cuda.amp.GradScaler()

        self.is_activated = False

    def setup(self, opt):
        """Load and print networks
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if not self.isTrain or opt.continue_train:
            self.load_networks(self.opt.weight_path)
        else:
            self.load_umms_weights(self.opt.weight_path)
        self.set_requires_grad(self.net_umms, False)
        self.net_umms.set_requires_grad_target_encoders(True)
        self.print_networks(opt.verbose)

    def load_umms_weights(self, load_path):
        if not os.path.exists(load_path):
            raise IOError(f"Checkpoint '{load_path}' does not exist")
        ddp_logger.info('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        net = self.net_umms
        if isinstance(net, torch.nn.parallel.DataParallel) or isinstance(net, torch.nn.parallel.DistributedDataParallel):
            ddp_logger.warning('loading model of type torch.nn.parallel')
            net = net.module
        net.load_state_dict(state_dict, strict=True)

    def set_input(self, inputs):
        self.source_volume = inputs['us_volume'].to(self.device)   # bs C D H W, C=1
        self.source_label = inputs['us_label'].to(self.device)     # bs C D H W, C=1
        self.target_volume = inputs['mr_volume'].to(self.device)   # bs C D H W, C=1
        self.target_label = inputs['mr_label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = {'source': inputs['us_volume_path'], 'target': inputs['mr_volume_path']}
        self.label_path = {'source': inputs['us_label_path'],  'target': inputs['mr_label_path']}
        self.spacing = {'source': inputs['us_spacing'].mean(0).tolist(), 'target': inputs['mr_spacing'].mean(0).tolist()}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.source_predict = self.net_umms(self.source_volume, 'source')
            self.source_features = self.feature_extractor.get_out_feature()
            self.source_logits = self.logit_extractor.get_out_feature()
            self.target_predict = self.net_umms(self.target_volume, 'target')
            self.target_features = self.feature_extractor.get_out_feature()
            self.target_logits = self.logit_extractor.get_out_feature()
        self.is_activated = False

    @staticmethod
    def get_tensors_detach(tensors):
        out_list = []
        for tensor in tensors:
            out_list.append(tensor.detach())
        return out_list

    def backward_df(self):
        with self.autocast_context():
            pred_real = self.net_df(*self.get_tensors_detach(self.source_features))
            loss_real = self.criterionGAN(pred_real, True)

            pred_fake = self.net_df(*self.get_tensors_detach(self.target_features))
            loss_fake = self.criterionGAN(pred_fake, False)
            loss_df = (loss_real+loss_fake)/0.5

            if self.opt.gan_mode == 'wgangp':
                self.loss_dfpenalty, _ = cal_gradient_penalty_feature(self.net_df,
                                                                      self.source_features, self.target_features,
                                                                      self.device, lambda_gp=0.2)
                self.loss_df = self.loss_dfpenalty + loss_df
            else:
                self.loss_df = loss_df

        if self.opt.use_mixed_precision:
            self.scaler_D.scale(self.loss_df).backward()
        else:
            self.loss_df.backward()

    def backward_dm(self):
        with self.autocast_context():
            pred_real = self.net_dm(*self.get_tensors_detach(self.source_logits))
            loss_real = self.criterionGAN(pred_real, True)

            pred_fake = self.net_dm(*self.get_tensors_detach(self.target_logits))
            loss_fake = self.criterionGAN(pred_fake, False)
            loss_dm = (loss_real + loss_fake) / 0.5

            if self.opt.gan_mode == 'wgangp':
                self.loss_dmpenalty, _ = cal_gradient_penalty(self.net_dm,
                                                              *self.source_logits, *self.target_logits,
                                                              self.device, lambda_gp=0.2)
                self.loss_dm = self.loss_dmpenalty + loss_dm
            else:
                self.loss_dm = loss_dm

        if self.opt.use_mixed_precision:
            self.scaler_D.scale(self.loss_dm).backward()
        else:
            self.loss_dm.backward()

    def backward_umms(self):
        with self.autocast_context():
            self.loss_gf = self.criterionGAN(self.net_df(*self.target_features), True)
            self.loss_gm = self.criterionGAN(self.net_dm(*self.target_logits), True)
            self.loss_regular = self.criterionRegular(self.net_umms.get_target_encoders_parameters())  # 1e-4
            # self.loss_prior = 1e4*self.criterionPrior(self.source_predict.detach(), self.target_predict)   # 10
            self.loss_prior = 1e4*self.criterionPrior(self.source_label.sum((1, 2, 3, 4)),
                                                      self.target_predict, uselabel=True)   # 10
            self.loss_umms = 1e-4*self.loss_regular + 1e-4*self.loss_prior + self.loss_gm + self.loss_gf

        if self.opt.use_mixed_precision:
            self.scaler_S.scale(self.loss_umms).backward()
        else:
            self.loss_umms.backward()

    def record_loss_item(self, domain):
        errors_ret = OrderedDict()
        for key, item in self.loss_item_dict.items():
            errors_ret[domain + key] = item
        return errors_ret

    def optimizer_step(self, optimizer, scaler=None):
        if self.opt.use_mixed_precision:
            scaler.step(optimizer)  # maybe apply to all optimizers
            scaler.update()
        else:
            optimizer.step()

    def optimize_parameters(self, update=True):
        if update:
            self.forward()
            self.set_requires_grad([self.net_dm, self.net_df], False)
            self.optimizer_S.zero_grad()
            self.backward_umms()
            self.optimizer_step(self.optimizer_S, self.scaler_S)

            self.set_requires_grad([self.net_dm, self.net_df], True)
            self.optimizer_D.zero_grad()
            self.backward_df()
            self.backward_dm()
            self.optimizer_step(self.optimizer_D, self.scaler_D)

        else:
            self.forward()
            self.set_requires_grad([self.net_dm, self.net_df], True)
            self.optimizer_D.zero_grad()
            self.backward_df()
            self.backward_dm()
            self.optimizer_step(self.optimizer_D, self.scaler_D)

    def compute_visuals(self):
        if not self.is_activated:
            self.source_predict = self.finally_activate(self.source_predict)
            self.target_predict = self.finally_activate(self.target_predict)
            self.is_activated = True

    def compute_metrics(self, *args, **kwargs):
        if not self.is_activated:
            self.source_predict = self.finally_activate(self.source_predict)
            self.target_predict = self.finally_activate(self.target_predict)
            self.is_activated = True

        self.metric_dict_source = self.compute_metrics_base(self.source_predict.clone().detach(),
                                                            self.source_label.clone().detach(), 'source')
        self.metric_dict_target = self.compute_metrics_base(self.target_predict.clone().detach(),
                                                            self.target_label.clone().detach(), 'target')

    def compute_metrics_base(self, predict, label, domain, *args, **kwargs):
        keys = tuple(self.metric_names) + args

        predict = (predict > 0.5).float()
        label = (label > 0.5).float()
        metrics = self.get_metrics_soft(predict, label, *self.metric_names,
                                        *args, **kwargs, voxelspacing=self.spacing[domain])

        if self.opt.DDP:
            for i in range(len(metrics)):
                if isinstance(metrics[i], torch.Tensor):
                    metrics[i] = reduce_mean(metrics[i], torch.distributed.get_world_size())

        metric_dict = dict(zip(keys, metrics))

        return metric_dict

    def get_current_metrics(self):
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_ret['source'+name] = self.metric_dict_source[name]
                metrics_ret['target'+name] = self.metric_dict_target[name]
        metrics_ret['mravd'] = 1 - self.target_predict.detach().mean() / self.source_predict.detach().mean()
        metrics_ret['mlravd'] = 1 - self.target_predict.detach().mean() / self.source_label.detach().mean()
        return metrics_ret

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret['source_'+name] = getattr(self, 'source_'+name)
                visual_ret['target_'+name] = getattr(self, 'target_'+name)
        return visual_ret


if __name__ == '__main__':
    pass

