import torch
import logging
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp
import torch.nn.functional as F

from functools import partial
from types import SimpleNamespace
from .base_model import BaseModel
from models.loss import losses, get_loss_criterion
from models.auxiliary_funs import get_init_func, get_activation
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from utils.others.metrics import BinaryMetrics, SoftMetrics
from utils.others.distributed_utils import reduce_mean
from utils.others.utils import print_numpy
from collections import OrderedDict, defaultdict

from models.modules.ummkd3d import UnetWithNormSpecficity

ddp_logger = logging.getLogger('ddp_logger')


def define_ummkd(opt, device):
    assert not(opt.DDP and opt.DP)

    net = UnetWithNormSpecficity(['source', 'target'], 'batch', opt.input_nc, opt.output_nc,
                                 deptp=4, init_channel_number=opt.init_channel_number, final_sigmoid=False)

    # init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
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


class UmmkdModel(BaseModel):
    def __init__(self, opt):
        super(UmmkdModel, self).__init__(opt)

        self.model_names = ['umms']
        self.net_umms = define_ummkd(opt, self.device)
        self.finally_activate = get_activation('sigmoid').to(self.device)

        # 'combo_source', 'combo_target',
        self.loss_names = ['dice_source', 'dice_target', 'bce_source', 'bce_target', 'l2', 'total']

        if self.isTrain:
            other_loss_kwargs = {}
            self.criterionDice = get_loss_criterion(name='bdc',
                                                    ignore_index=opt.ignore_index,
                                                    reduction=opt.reduction,
                                                    use_sigmoid=True,
                                                    eps=1e-7,
                                                    smooth=1.0,
                                                    **other_loss_kwargs).to(self.device)
            self.criterionBCE = get_loss_criterion(name='bce',
                                                   ignore_index=opt.ignore_index,
                                                   reduction=opt.reduction,
                                                   weight=1,
                                                   smooth=0.01,
                                                   eps=1e-7,
                                                   )

            self.criterionL2 = getattr(losses, 'l2_regularization')

            optimizer_kwargs = {'eps': 1e-8,
                                'betas': (0.9, 0.999)
                                }
            if 'sgd' in opt.optimizer_name.lower():
                optimizer_kwargs.pop('betas', None)
            self.optimizer = create_optimizer_v2(self.net_umms.parameters(),
                                                 opt=opt.optimizer_name,
                                                 lr=opt.lr,
                                                 weight_decay=opt.weight_decay,
                                                 momentum=opt.momentum,
                                                 **optimizer_kwargs)

            self.optimizers.append(self.optimizer)
            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]
        # specify the images you want to save/display.
        self.visual_names = ['source_y', 'target_y', 'source_seg', 'target_seg']
        self.metric_names = ['DC', 'recall', 'precision', 'specificity', 'accuracy']

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-6)

        self.source = None
        self.target = None
        self.source_y = None
        self.target_y = None
        self.source_seg = None
        self.target_seg = None

        self.loss_dice_source = None
        self.loss_dice_target = None
        self.loss_bce_source = None
        self.loss_bce_target = None
        # self.loss_combo_source = None
        # self.loss_combo_target = None
        self.loss_l2 = None
        self.loss_total = None

        self.metric_dict_source = None
        self.metric_dict_target = None
        # setattr(self, opt.loss_name, None)

        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_umms.no_sync if opt.DDP else contextlib.nullcontext
        self.scaler = torch.cuda.amp.GradScaler()

        self.is_activated = False

    def set_input(self, input):
        self.source = input['mr_volume'].to(self.device)   # bs C D H W, C=1
        self.source_y = input['mr_label'].to(self.device)     # bs C D H W, C=1
        self.target = input['us_volume'].to(self.device)   # bs C D H W, C=1
        self.target_y = input['us_label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = [{'source_path': input['mr_volume_path'], 'target_path': input['us_volume_path']}]
        self.label_path = [{'source_y_path': input['mr_label_path'], 'target_y_path': input['us_label_path']}]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.source_seg = self.net_umms(self.source, 'source')
            self.target_seg = self.net_umms(self.target, 'target')
        self.is_activated = False

    def get_seg_loss(self, predict, target):
        dice = self.criterionDice(predict, target)
        bce = self.criterionBCE(predict, target)
        return dice+bce

    def backward(self):
        with self.autocast_context():
            self.loss_dice_source = self.criterionDice(self.source_seg, self.source_y)
            self.loss_dice_target = self.criterionDice(self.target_seg, self.target_y)
            self.loss_bce_source = self.criterionBCE(self.source_seg, self.source_y)
            self.loss_bce_target = self.criterionBCE(self.target_seg, self.target_y)
            # self.loss_combo_source = self.get_seg_loss(self.source_y, self.source_seg)
            # self.loss_combo_target = self.get_seg_loss(self.target_y, self.target_seg)
            self.loss_l2 = self.criterionL2(self.net_umms.parameters())

        self.loss_total = (1.5*self.loss_dice_source + 0.75*self.loss_dice_target
                           + 1.5*self.loss_bce_source + 0.75*self.loss_bce_target
                           + 1.5e-4*self.loss_l2)

        self.loss_total = self.loss_total / self.opt.gradient_accumulation_k_step

        if self.opt.use_mixed_precision:
            self.scaler.scale(self.loss_total).backward()
            self.scaler.step(self.optimizer)  # maybe apply to all optimizers
            self.scaler.update()
        else:
            self.loss_total.backward()

    def optimize_parameters(self, update=True):
        if update:
            self.forward()
            self.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            with self.no_sync_context():
                self.forward()
                self.backward()

    def compute_visuals(self):
        if not self.is_activated:
            self.source_seg = self.finally_activate(self.source_seg)
            self.target_seg = self.finally_activate(self.target_seg)
            self.is_activated = True

    def compute_metrics(self, *args, **kwargs):
        if not self.is_activated:
            self.source_seg = self.finally_activate(self.source_seg)
            self.target_seg = self.finally_activate(self.target_seg)
            self.is_activated = True

        self.metric_dict_source = self.compute_metrics_base(self.source_seg.clone().detach(),
                                                            self.source_y.clone().detach())
        self.metric_dict_target = self.compute_metrics_base(self.target_seg.clone().detach(),
                                                            self.target_y.clone().detach())

    def compute_metrics_base(self, predict, label, *args, **kwargs):
        keys = tuple(self.metric_names) + args

        predict = (predict > 0.5).float()
        label = (label > 0.5).float()
        metrics = self.get_metrics_soft(predict, label, *self.metric_names, *args, **kwargs)

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
        return metrics_ret


def main():
    from configs.options.dataset_network import ProjectOptions
    opt = ProjectOptions().parse(True)   # get training options
    model = UmmkdModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()

