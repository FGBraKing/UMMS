import torch
import logging
import torch.distributed
import torch.nn as nn
import torch.cuda.amp
from models.modules.MultimodalSegmentation.SingleUnetWithFuse import SingleUnetWithFuse
from models.networks.waittorepair.standardwithprior_model import StandardWithPriorModel

from models.auxiliary_funs import get_init_func

ddp_logger = logging.getLogger('ddp_logger')


def define_model(opt, device, domains=None):
    assert not(opt.DDP and opt.DP)
    net = SingleUnetWithFuse(in_channels=opt.input_nc,
                             out_channels=opt.output_nc,
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


class SingleFuseModel(StandardWithPriorModel):
    def __init__(self, opt):
        self.device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')
        self.net_umms = define_model(opt, self.device)
        super(SingleFuseModel, self).__init__(opt, self.net_umms)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast_context():
            self.source_predict, self.target_predict = self.net_umms(self.source_volume, self.target_volume)
        self.is_activated = False

    def backward(self):
        self.backward_both()

