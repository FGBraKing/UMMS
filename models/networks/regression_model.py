import torch
import logging
import warnings
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp
import torch.nn.functional as F
from functools import partial
from types import SimpleNamespace
from collections import OrderedDict, defaultdict
from models.modules.classification.resnet3d import ResNet3D, generate_model
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

    net = generate_model(50, n_input_channels=opt.input_nc, n_classes=opt.output_nc, norm_type='batch', act_type="lrelu").to(device)

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


class RegressionModel(BaseModel):
    def __init__(self, opt):
        super(RegressionModel, self).__init__(opt)

        self.model_names = ['regression']
        self.net_regression = define_model(opt, self.device)

        self.loss_names = ['regular', 'mse', 'total']
        if self.isTrain:
            self.criterionMse = nn.MSELoss()
            self.criterionRegular = get_loss_criterion(name='custom_regular')

            optimizer_kwargs = {'eps': 1e-8,
                                'betas': (opt.optim_beta, 0.999)
                                }
            if 'sgd' in opt.optimizer_name.lower():
                optimizer_kwargs.pop('betas', None)
            self.optimizer = create_optimizer_v2(self.net_regression.parameters(),
                                                 opt=opt.optimizer_name,
                                                 lr=opt.lr,
                                                 weight_decay=opt.weight_decay,
                                                 momentum=opt.momentum,
                                                 **optimizer_kwargs)

            self.optimizers.append(self.optimizer)
            self.schedulers = [create_scheduler(opt, optimizer)[0] for optimizer in self.optimizers]

        self.visual_names = []  # 'predict', 'label', 'volume'
        self.metric_names = ['DC', 'mae', 'roisize']
        # 'hd' 'hd95' 'assd' 'asd'  'ravd' 'DC', 'ravd', 'recall', 'precision', 'specificity', 'accuracy'

        self.volume = None
        self.label = None
        self.label_ratio = None
        self.predict = None
        self.spacing = None

        self.loss_mse = None
        self.loss_regular = None
        self.loss_total = None
        self.metrics = None

        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_regression.no_sync if opt.DDP else contextlib.nullcontext
        self.scaler = torch.cuda.amp.GradScaler()

    def set_input(self, inputs):
        self.volume = inputs['volume'].to(self.device)   # bs C D H W, C=1
        self.label = inputs['label'].to(self.device)     # bs C D H W, C=1
        self.label_ratio = self.label.sum() / self.label.numel()
        self.volume_path = inputs['volume_path']
        self.label_path = inputs['label_path']
        self.spacing = inputs['spacing'].mean(0).tolist()
        if self.opt.DEBUG:
            print('proportion: {:.2%}'.format(inputs['label'].sum()/inputs['label'].numpy().size))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.predict = self.net_regression(self.volume)

    def backward(self):
        with self.autocast_context():
            self.loss_regular = self.criterionRegular(self.net_regression.parameters())
            self.loss_mse = self.criterionMse(self.predict, self.label_ratio)

            self.loss_total = 100*self.loss_mse
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

    def compute_metrics(self, *args, **kwargs):
        keys = list(self.metric_names)
        values = []
        for key in keys:
            if key == 'mae':
                values.append((self.predict.detach() - self.label_ratio.detach()).mean())
            elif key == 'roisize':
                values.append(self.label_ratio.detach().mean())
            elif key == "DC":
                values.append(1 - (self.predict.detach() - self.label_ratio.detach()).pow(2).mean())
            else:
                keys.remove(key)

        if self.opt.DDP:
            for i in range(len(values)):
                if isinstance(values[i], torch.Tensor):
                    values[i] = reduce_mean(values[i], torch.distributed.get_world_size())

        self.metric_dict = dict(zip(keys, values))


def main():
    from configs.options.dataset_network import ProjectOptions
    opt = ProjectOptions().parse(True)   # get training options
    model = RegressionModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()
