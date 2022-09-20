import warnings
import copy
import torch
import logging
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp

from types import SimpleNamespace
from models.modules.MultimodalSegmentation import SingleUnet
from models.networks.base_model import BaseModel
from models.loss import losses, get_loss_criterion
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from models.auxiliary_funs import get_init_func, get_activation
from utils.others.metrics import BinaryMetrics, SoftMetrics
from utils.others.distributed_utils import reduce_mean
from utils.others.utils import print_numpy
from test import test_during_train
from collections import OrderedDict, defaultdict

ddp_logger = logging.getLogger('ddp_logger')
# --config_path=configs/defaults/mrusmr_unet_train.yaml --use_config


def define_model(opt, device):
    assert not(opt.DDP and opt.DP)

    net = SingleUnet(opt.input_nc, opt.output_nc,
                     f_maps=opt.init_channel_number,
                     num_levels=5,
                     with_activation=False,
                     final_sigmoid=True,
                     interpolation=True,
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


class SingleModel(BaseModel):
    def __init__(self, opt):
        super(SingleModel, self).__init__(opt)

        self.model_names = ['segment']
        self.net_segment = define_model(opt, self.device)
        self.finally_activate = get_activation('sigmoid').to(self.device)

        self.loss_names = ['regular', 'combo', 'total']
        if self.isTrain:
            self.criterionCombo = get_loss_criterion(name='custom')
            self.criterionRegular = get_loss_criterion(name='custom_regular')

            optimizer_kwargs = {'eps': 1e-8,
                                'betas': (opt.optim_beta, 0.999)}
            if 'sgd' in opt.optimizer_name.lower():
                optimizer_kwargs.pop('betas', None)
            self.optimizer = create_optimizer_v2(self.net_segment.parameters(),
                                                 opt=opt.optimizer_name,
                                                 lr=opt.lr,
                                                 weight_decay=opt.weight_decay,
                                                 momentum=opt.momentum,
                                                 **optimizer_kwargs)

            self.optimizers.append(self.optimizer)
            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]

        self.visual_names = ['predict', 'label', 'volume']
        self.metric_names = ['DC', 'ravd', 'recall', 'precision', 'accuracy', 'roisize']
        self.test_metric_names = ['DC', 'recall', 'precision', 'specificity', 'accuracy', 'hd', 'hd95', 'assd', 'asd', 'ravd', 'roisize']

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-6)

        self.volume = None
        self.label = None
        self.predict = None
        self.spacing = None

        self.loss_combo = None
        self.loss_regular = None
        self.loss_total = None
        self.metrics = None
        # setattr(self, opt.loss_name, None)

        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_segment.no_sync if opt.DDP else contextlib.nullcontext
        self.scaler = torch.cuda.amp.GradScaler()

        self.is_activated = False

    def set_input(self, inputs):
        self.volume = inputs['volume'].to(self.device)   # bs C D H W, C=1
        self.label = inputs['label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = inputs['volume_path']
        self.label_path = inputs['label_path']
        self.spacing = inputs['spacing'].mean(0).tolist()
        if self.opt.DEBUG:
            print('proportion: {:.2%}'.format(inputs['label'].sum()/inputs['label'].numpy().size))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.predict = self.net_segment(self.volume)
        self.is_activated = False

    def backward(self):
        with self.autocast_context():
            # self.loss_dice = self.criterionDice(self.predict, self.label)
            # self.loss_bce = self.criterionBCE(self.predict, self.label)
            self.loss_item_dict, self.loss_combo = self.criterionCombo(self.predict, self.label)
            self.loss_regular = self.criterionRegular(self.net_segment.parameters())
            self.loss_total = self.loss_combo + 1e-4*self.loss_regular
        self.loss_total = self.loss_total / self.opt.gradient_accumulation_k_step

        if self.opt.use_mixed_precision:
            self.scaler.scale(self.loss_total).backward()
        else:
            self.loss_total.backward()

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
            self.predict = self.finally_activate(self.predict)
            self.is_activated = True

    def compute_metrics(self, *args, **kwargs):
        if not self.is_activated:
            self.predict = self.finally_activate(self.predict)
            self.is_activated = True

        if self.net_segment.training:
            metric_names = tuple(self.metric_names)
        else:
            metric_names = tuple(self.test_metric_names)
        keys = metric_names + args

        predict = self.predict.clone().detach()
        label = self.label.clone().detach()
        predict = (predict > 0.5).float()
        label = (label > 0.5).float()
        self.metrics = self.get_metrics_soft(predict, label, *metric_names,
                                             *args, **kwargs, voxelspacing=self.spacing)

        if self.opt.DDP:
            for i in range(len(self.metrics)):
                if isinstance(self.metrics[i], torch.Tensor):
                    self.metrics[i] = reduce_mean(self.metrics[i], torch.distributed.get_world_size())

        self.metric_dict = dict(zip(keys, self.metrics))

    # 当前版本尚不支持DDP模式下进行滑动窗口测试
    def slide_test(self, one_patient):
        if self.opt.DDP:
            raise AttributeError('当前版本尚不支持DDP模式下进行滑动窗口测试')
        kwargs = {
            'batch_size': self.opt.slide_test_batchsize,
            'num_threads': self.opt.num_threads,
            'crop_size': self.opt.crop_size,
            'stride': (16, 16, 8),      # 3*3*3
            'no_augment': True,
            'visual_names': ('segment', 'label', 'origin_volume'),
            'metric_names': tuple(self.test_metric_names)
        }
        # ('DC', 'recall', 'precision', 'specificity', 'accuracy', 'hd', 'hd95', 'assd', 'asd', 'ravd')
        self.metric_dict, visuals = test_during_train(one_patient, self.net_segment,
                                                      SimpleNamespace(**kwargs), self.device)

        for name in self.visual_names:
            if name == 'predict':
                self.predict = visuals['segment']
            elif name == 'label':
                self.label = visuals['label']
            elif name == 'volume':
                self.volume = visuals['origin_volume']
            else:
                warnings.warn('在滑窗测试中使用了错误的visual name:{}'.format(name))
                ddp_logger.warning('在滑窗测试中使用了错误的visual name')

    def get_current_metrics(self):
        if self.net_segment.training:
            metric_names = tuple(self.metric_names)
        else:
            metric_names = tuple(self.test_metric_names)
        metrics_ret = OrderedDict()
        for name in metric_names:
            if isinstance(name, str) and name in self.metric_dict.keys():
                metrics_ret[name] = self.metric_dict[name]
        return metrics_ret


def main():
    from configs.options.dataset_network import ProjectOptions
    opt = ProjectOptions().parse(True)   # get training options
    model = SingleModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()
