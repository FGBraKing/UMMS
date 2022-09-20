import torch
import logging
import copy
import warnings
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp
import torch.nn.functional as F
from functools import partial
from types import SimpleNamespace
from collections import OrderedDict, defaultdict
from models.modules.MultimodalSegmentation.DSBNWithAuxTask import ChilopodUnetWithRegression, ChilopodUnetWithRegressionFinetuneV1, ChilopodUnetWithRegressionFinetuneV2
from models.modules.MultimodalSegmentation.DSBNWithFuse import ChilopodUnetWithFuse
from models.networks.base_model import BaseModel
from models.loss import losses, get_loss_criterion
from models.loss.custom_loss import EdgeFilter, EdgeLossV3
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

    if opt.network_type == 'finetunev1':
        net = ChilopodUnetWithRegressionFinetuneV1(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            f_maps=opt.init_channel_number,
            num_levels=5,
            with_activation=False,
            final_sigmoid=True,
            interpolation=opt.up_interpolate,
            norm_type="batch",
            act_type="lrelu",
        ).to(device)
    elif opt.network_type == 'finetunev2':
        net = ChilopodUnetWithRegressionFinetuneV2(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            f_maps=opt.init_channel_number,
            num_levels=5,
            with_activation=False,
            final_sigmoid=True,
            interpolation=opt.up_interpolate,
            norm_type="batch",
            act_type="lrelu",
        ).to(device)
    else:
        net = ChilopodUnetWithRegression(
            in_channels=opt.input_nc,
            out_channels=opt.output_nc,
            f_maps=opt.init_channel_number,
            num_levels=5,
            with_activation=False,
            final_sigmoid=True,
            interpolation=opt.up_interpolate,
            norm_type="batch",
            act_type="lrelu",
        ).to(device)

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


class DsbnPlusWithEdgeModel(BaseModel):
    def __init__(self, opt):
        super(DsbnPlusWithEdgeModel, self).__init__(opt)
        self.domains = ["source", "target"]

        self.model_names = ['umms']
        self.net_umms = define_model(opt, self.device, self.domains)
        self.finally_activate = get_activation('sigmoid').to(self.device)

        self.get_edge = EdgeFilter(True).to(self.device)

        self.loss_names = ['regular', 'routine', 'regress', 'edge', 'total']  # 'kd',

        if self.isTrain:
            self.criterionRoutineMulti = get_loss_criterion(name='custom_multimodal', use_mixed_precision=opt.use_mixed_precision)
            self.criterionRegular = get_loss_criterion(name='custom_regular')
            self.criterionRegress = nn.MSELoss()
            # self.criterionRegress = nn.L1Loss()
            self.criterionEdge = EdgeLossV3()

            regress_paras = []
            routine_paras = []
            for k, v in self.net_umms.named_parameters():
                if k.find('regressors') == 0 or k.find('outregress') == 0:
                    regress_paras.append(v)
                else:
                    routine_paras.append(v)
            optimizer_paras = [{'params': regress_paras, 'lr': 1e-5}, {'params': routine_paras, 'lr': opt.lr}]

            optimizer_kwargs = {'eps': 5e-4 if opt.use_mixed_precision else 6e-8,
                                'betas': (opt.optim_beta, 0.999)
                                }
            if 'sgd' in opt.optimizer_name.lower():
                optimizer_kwargs.pop('betas', None)
            self.optimizer = create_optimizer_v2(optimizer_paras,
                                                 opt=opt.optimizer_name,
                                                 lr=opt.lr,
                                                 weight_decay=opt.weight_decay,
                                                 momentum=opt.momentum,
                                                 **optimizer_kwargs)

            self.optimizers.append(self.optimizer)
            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]

        # specify the images you want to save/display.
        self.visual_names = ['volume', 'predict', 'label']  # , 'feature_k', 'feature_l'
        self.metric_names = ['DC', 'recall', 'precision', 'ravd']
        # 'hd' 'hd95' 'assd' 'asd'  'specificity', 'accuracy', 'roisize'

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-6)

        self.source_volume = None
        self.target_volume = None
        self.source_sampleweight = None
        self.target_sampleweight = None
        self.source_dismap = None
        self.target_dismap = None
        self.source_pedge = None
        self.target_pedge = None
        self.source_label = None
        self.target_label = None
        self.source_predict = None
        self.target_predict = None
        self.spacing = None
        self.label_ratio = None
        self.aim_ratio = None
        self.ratio_predict = None
        self.source_aux_predict1 = None
        self.target_aux_predict1 = None

        self.loss_routine = None
        self.loss_regular = None
        self.loss_regress = None
        self.loss_edge = None
        self.loss_total = None

        self.prior_gamma_base = opt.prior_gamma

        self.metric_dict_source = None
        self.metric_dict_target = None
        self.radio_error = None

        self.scaler = torch.cuda.amp.GradScaler()
        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_umms.no_sync if opt.DDP else contextlib.nullcontext

        self.is_activated = False

    def set_input(self, inputs):
        self.source_volume = inputs['mr_volume'].to(self.device)   # bs C D H W, C=1
        self.source_label = inputs['mr_label'].to(self.device)     # bs C D H W, C=1
        self.target_volume = inputs['us_volume'].to(self.device)   # bs C D H W, C=1
        self.target_label = inputs['us_label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = {'source': inputs['mr_volume_path'], 'target': inputs['us_volume_path']}
        self.label_path = {'source': inputs['mr_label_path'],  'target': inputs['us_label_path']}
        self.spacing = {'source': inputs['mr_spacing'].mean(0).tolist(), 'target': inputs['us_spacing'].mean(0).tolist()}
        self.label_ratio = {'source': self.source_label.mean((1, 2, 3, 4)),
                            'target': self.target_label.mean((1, 2, 3, 4))}
        self.aim_ratio = (self.label_ratio['source']+self.label_ratio['target']) / 2.0

        if self.source_label.min() < 0 or self.target_label.min() < 0:
            print(self.source_label.min(), self.target_label.min())

        self.source_dismap = inputs['mr_dismap'].to(self.device)
        self.target_dismap = inputs['mr_dismap'].to(self.device)
        self.source_sampleweight = 1 - self.source_dismap
        self.target_sampleweight = 1 - self.target_dismap

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            if self.opt.network_type == 'finetunev1' or self.opt.network_type == 'finetunev2':
                self.source_aux_predict1, self.target_aux_predict1, self.ratio_predict, self.source_predict, self.target_predict = self.net_umms(self.source_volume, self.target_volume)
                self.source_pedge = self.get_edge(self.source_predict)
                self.target_pedge = self.get_edge(self.target_predict)
            else:
                self.source_predict, self.target_predict, self.ratio_predict = self.net_umms(self.source_volume, self.target_volume)
                self.source_pedge = self.get_edge(self.source_predict)
                self.target_pedge = self.get_edge(self.target_predict)
        self.is_activated = False
        # print((self.ratio_predict, self.aim_ratio))

    def backward(self):
        with self.autocast_context():
            self.loss_regular = self.criterionRegular(self.net_umms.parameters())
            if ex_config.current_epoch < 10:
                self.loss_regress = 0
            else:
                self.loss_regress = 1e4 * self.criterionRegress(self.ratio_predict, self.aim_ratio)
            # self.loss_regress = 1000. * self.criterionRegress(self.ratio_predict, self.aim_ratio)

            self.loss_edge = (self.criterionEdge(self.source_pedge, self.source_dismap) +
                              self.criterionEdge(self.target_pedge, self.target_dismap)) / 2. * 1e4
            if self.opt.network_type == 'finetunev1' or self.opt.network_type == 'finetunev2':
                _, aux_routine1 = self.criterionRoutineMulti(self.source_aux_predict1, self.source_label,
                                                             self.target_aux_predict1, self.target_label)
                self.loss_item_dict, self.loss_routine = self.criterionRoutineMulti(self.source_predict, self.source_label,
                                                                                    self.target_predict, self.target_label)
                # self.source_sampleweight, self.target_sampleweight
                self.loss_total = 1e-4*self.loss_regular + 1.0*self.loss_routine + 0.6*aux_routine1 +\
                                  0.0*self.loss_edge*1e-4 + 1.0*self.loss_regress*1e-4
            else:
                self.loss_item_dict, self.loss_routine = self.criterionRoutineMulti(self.source_predict, self.source_label,
                                                                                    self.target_predict, self.target_label)
                self.loss_total = 1e-4*self.loss_regular + 1.0*self.loss_routine + \
                                  0.0*self.loss_edge*1e-4 + 1.0*self.loss_regress*1e-4

            if torch.isnan(self.loss_total) or self.loss_total > 100:
                message_logger = logging.getLogger('train_message_log')
                message_logger.warning('epoch: ', ex_config.current_epoch,
                                       'losses: ', self.loss_routine, self.loss_regress)
                raise RuntimeError

        self.loss_total = self.loss_total / self.opt.gradient_accumulation_k_step

        if self.opt.use_mixed_precision:
            self.scaler.scale(self.loss_total).backward()
        else:
            self.loss_total.backward()

    def record_loss_item(self, domain):
        errors_ret = OrderedDict()
        for key, item in self.loss_item_dict.items():
            errors_ret[domain+key] = item
        return errors_ret

    def optimizer_step(self):
        if self.opt.use_mixed_precision:
            self.scaler.step(self.optimizer)  # maybe apply to all optimizers
            self.scaler.update()
        else:
            self.optimizer.step()

    def optimize_parameters(self, update=True):
        if update:
            self.forward()
            self.backward()
            self.optimizer_step()
            self.optimizer.zero_grad()
        else:
            with self.no_sync_context():
                self.forward()
                self.backward()

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
        self.radio_error = (self.ratio_predict.mean() - self.aim_ratio.mean()) / self.aim_ratio.mean()

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
                metrics_ret['source' + name] = self.metric_dict_source[name]
                metrics_ret['target' + name] = self.metric_dict_target[name]
        metrics_ret['radio'] = self.radio_error
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

