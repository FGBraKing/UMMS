import torch
import logging
import contextlib
import torch.distributed
import torch.nn as nn
import torch.cuda.amp

from types import SimpleNamespace
from models.networks.base_model import BaseModel
from models.modules import VNet
from models.loss import losses, get_loss_criterion
from models.auxiliary_funs import get_init_func, get_activation
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from utils.others.metrics import BinaryMetrics, SoftMetrics
from utils.others.distributed_utils import reduce_mean
from utils.others.utils import print_numpy

ddp_logger = logging.getLogger('ddp_logger')


def define_model(opt, device):
    assert not(opt.DDP and opt.DP)

    net = VNet(in_channels=opt.input_nc, classes=opt.output_nc)

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


class Vnet3dModel(BaseModel):
    def __init__(self, opt):
        super(Vnet3dModel, self).__init__(opt)

        self.model_names = ['segment']
        self.net_segment = define_model(opt, self.device)
        self.finally_activate = get_activation('sigmoid').to(self.device)

        self.loss_names = ['seg']
        if self.isTrain:
            other_loss_kwargs = {}
            # (sample_weight)   (gamma_neg gamma_pos clip)  (num_splits)  (activate)  (bce_smooth)
            self.criterion = get_loss_criterion(name=opt.loss_name,
                                                ignore_index=opt.ignore_index, reduction=opt.reduction,
                                                eps=opt.loss_eps, smooth=opt.loss_smooth,
                                                alpha=opt.loss_alpha, beta=opt.loss_beta,
                                                gamma=opt.loss_gamma, weight=opt.loss_weight,
                                                **other_loss_kwargs).to(self.device)
            optimizer_kwargs = {'eps': 1e-8,
                                'betas': (opt.optim_beta, 0.999)
                                }
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

        # specify the images you want to save/display.
        self.visual_names = ['predict', 'label', 'volume']
        self.metric_names = ['DC', 'recall', 'precision', 'specificity', 'accuracy']

        self.get_metrics = BinaryMetrics()
        self.get_metrics_soft = SoftMetrics(smooth=0., eps=1e-6)

        self.volume = None
        self.label = None
        self.predict = None
        self.loss_seg = None
        self.metrics = None
        # setattr(self, opt.loss_name, None)

        self.autocast_context = torch.cuda.amp.autocast if opt.use_mixed_precision else contextlib.nullcontext
        self.no_sync_context = self.net_segment.no_sync if opt.DDP else contextlib.nullcontext
        self.scaler = torch.cuda.amp.GradScaler()

        self.is_activated = False

    def set_input(self, input):

        self.volume = input['volume'].to(self.device)   # bs C D H W, C=1
        self.label = input['label'].to(self.device)     # bs C D H W, C=1
        self.volume_path = input['volume_path']
        self.label_path = input['label_path']
        if self.opt.DEBUG:
            print('proportion: {:.2%}'.format(input['label'].sum()/input['label'].numpy().size))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.predict = self.net_segment(self.volume)
        self.is_activated = False

    def backward(self):
        with self.autocast_context():
            self.loss_seg = self.criterion(self.predict, self.label)
        self.loss_seg = self.loss_seg / self.opt.gradient_accumulation_k_step

        if self.opt.use_mixed_precision:
            self.scaler.scale(self.loss_seg).backward()
        else:
            self.loss_seg.backward()

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

        keys = tuple(self.metric_names) + args

        predict = self.predict.clone().detach()
        label = self.label.clone().detach()
        predict = (predict > 0.5).float()
        label = (label > 0.5).float()
        self.metrics = self.get_metrics_soft(predict, label, *self.metric_names, *args, **kwargs)

        if self.opt.DDP:
            for i in range(len(self.metrics)):
                if isinstance(self.metrics[i], torch.Tensor):
                    self.metrics[i] = reduce_mean(self.metrics[i], torch.distributed.get_world_size())

        self.metric_dict = dict(zip(keys, self.metrics))


def main():
    from configs.options.dataset_network import ProjectOptions
    opt = ProjectOptions().parse(True)   # get training options
    model = Vnet3dModel(opt)
    opt.continue_train = True
    model.setup(opt)


if __name__ == '__main__':
    main()
