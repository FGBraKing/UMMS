# -*- coding: utf-8 -*-

import os
import re
# import sys
# import time
# import yaml
import torch
# import logging
# import imageio
# import argparse
import numpy as np
# import torch.nn as nn
# import nibabel as nib
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
import h5py
# import torch.optim
# import torch.distributed
# import torch.utils.data
# # import horovod.torch as hvd
# import torch.distributed as dist
# from data import CustomDatasetDataLoader
# from torchvision import transforms
# from torchvision.datasets.folder import default_loader
# # from configs.options.promise_3dunet import TrainOptions
# from data import create_dataset
# from models import create_model
# from utils.forLogs import Visualizer, get_logger
# from utils.forLogs.visualizer import Visualizer
# from utils.others.utils import Timer, convert_str_to_list
# from torch.nn.parallel import DistributedDataParallel as DDP
# from utils.others.metrics import BinaryMetrics
from pprint import pprint
from configs.utils_config import get_config

# from data.dataloads.trus_dataset import TrusDataset
# from data.utils_data import nii_loader
# from utils.others.utils import init_seed, init_torch
# from models.modules.segmentation.three_d.unet3d_gn import UNet3D
# from models.loss.region_based import BinaryDiceLoss
# from configs.simple_options import get_opt
# from configs.utils_config import pretty_print_opt
# from models.auxiliary_funs import get_init_func, get_activation
# from models.loss import get_loss_criterion
# from models.optim import create_optimizer, create_optimizer_v2
# from models.scheduler import create_scheduler
# # from horovod.runner.launch import run_commandline
# # from utils.others.distributed_utils_horovod import reduce_mean, metric_average
# import torch.distributed.launch
# from utils.others.img_io import show_volume_label_predict
# # import multiprocessing
from multiprocessing import Process
from glob import glob
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict
from data.utils_data import h5_loader


def print_visible(obj):
    pprint([a for a in dir(obj) if not a.startswith('_') and not a.endswith('_')])


def test_config():
    '''
['DEPRECATED_KEYS',
 'IMMUTABLE',
 'NEW_ALLOWED',
 'RENAMED_KEYS',
 'clear',
 'clone',
 'copy',
 'defrost',
 'dump',
 'freeze',
 'fromkeys',
 'get',
 'is_frozen',
 'is_new_allowed',
 'items',
 'key_is_deprecated',
 'key_is_renamed',
 'keys',
 'load_cfg',
 'merge_from_file',
 'merge_from_list',
 'merge_from_other_cfg',
 'pop',
 'popitem',
 'raise_key_rename_error',
 'register_deprecated_key',
 'register_renamed_key',
 'set_new_allowed',
 'setdefault',
 'update',
 'values']
'''
    from configs.default_config import _C as cfg

    default_dir = '/raid/lf/PROJECT/DLForPytorch/configs/defaults/'
    config_path = os.path.join(default_dir, 'trus_unet3d.yaml')
    print(cfg)  # yacs.config.CfgNode, dict
    print(len(cfg))     # 89
    print_visible(cfg)
    config_yaml = get_config(config_path)   # dict

    cfg.merge_from_file(config_path)

    print('config_yaml\n', config_yaml)
    print('yaml len:', len(config_yaml))        # 72
    # for k, v in config_yaml.items():
    #     print(k, v)
    # print('after merge:\n', cfg)


# def debug():
#     opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d.yaml', '--use_config'])
#     opt.horovod = True
#     pretty_print_opt(opt)
#
#     init_torch(gpu_id=opt.visible_gpu, deterministic=True)
#
#     # 1.Run hvd.init().
#     hvd.init()
#     # 2. Pin each GPU to a single process.
#     if torch.cuda.is_available():
#         torch.cuda.set_device(hvd.local_rank())
#
#     opt.gradient_predivide_factor = 1
#
#     # 3. Define dataset and dataloader
#     dataset = TrusDataset(opt, loader=nii_loader)
#     print('dataset created!')
#
#     sampler = torch.utils.data.distributed.DistributedSampler(dataset,
#                                                               num_replicas=hvd.size(),
#                                                               rank=hvd.rank(),
#                                                               shuffle=not opt.serial_batches,
#                                                               seed=0,
#                                                               drop_last=False) if opt.horovod else None
#     print('size:{},rank:{}, local_rank:{}'.format(hvd.size(), hvd.rank(), hvd.local_rank()))
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=opt.batch_size,
#         shuffle=(sampler is None) and (not opt.serial_batches),
#         sampler=sampler,        #
#         batch_sampler=None,     #
#         num_workers=int(opt.num_threads),
#         collate_fn=None,        #
#         pin_memory=True,
#         drop_last=False,        #
#         prefetch_factor=2       #
#     )
#
#     # 4. define model and optimizer
#     model = UNet3D(in_channels=opt.input_nc, out_channels=opt.output_nc, final_sigmoid=False,
#                    conv_layer_order=opt.conv_order, init_channel_number=opt.init_channel_number)
#     init_func = get_init_func(init_type=opt.init_type, init_gain=opt.init_gain)
#     model.apply(init_func)
#     model = model.cuda()
#     criterion = get_loss_criterion(name='bdc', ignore_index=None, reducetion='mean',
#                                    use_batch=True, use_sigmoid=True, smooth=0.).cuda()
#
#     lr_scaler = hvd.size() if not opt.use_adasum else 1
#     opt.lr = lr_scaler * opt.lr
#     optimizer = create_optimizer_v2(model.parameters(), opt='adam', lr=opt.lr, betas=(opt.optim_beta, 0.999))
#
#     optimizer = hvd.DistributedOptimizer(optimizer,
#                                          named_parameters=model.named_parameters(),
#                                          compression=hvd.Compression.fp16 if opt.fp16_allreduce else hvd.Compression.none,
#                                          backward_passes_per_step=1,
#                                          op=hvd.Adasum if opt.use_adasum else hvd.Average,
#                                          gradient_predivide_factor=opt.gradient_predivide_factor)
#     schedulers = create_scheduler(opt, optimizer)[0]
#
#     # 5 broadcast the initial variable states from rank 0 to all other processes:
#     hvd.broadcast_parameters(model.state_dict(), root_rank=0)
#     hvd.broadcast_optimizer_state(optimizer, root_rank=0)
#
#     # 6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.
#     for epoch in range(3):
#         for batch_idx, data in enumerate(dataloader):
#             volume = data['volume'].cuda(non_blocking=True)   # bs C D H W, C=1
#             label = data['label'].cuda(non_blocking=True)     # bs C D H W, C=1
#             volume_path = data['volume_path']
#             label_path = data['label_path']
#             output = model(volume)
#
#             loss = criterion(output, label)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print('batch_idx:', batch_idx)
#         schedulers.step(epoch)
#         print('epoch:', epoch)


def test_generator():
    print('entering test_generator')
    for i in range(10):
        print('before yield', i)
        yield i
        print('after yield', i)


def test_get_item():
    pass


class TestGetItem:
    def __init__(self):
        self.num = 10
        self.test_list = list(range(self.num))

    def __getitem__(self, item):
        return self.test_list[item]


def test_generator_code():
    aa = test_generator()

    print_visible(aa)

    print(aa)
    print(next(aa))
    print('**'*50)
    print(next(aa))
    print('**'*50)


def test_val_dataset():
    import numpy as np
    from data.dataloads.base_dataset import TestOnePatientDataset
    # from data.dataloads.trus_dataset import TestTrusDataset
    from yacs.config import CfgNode as CN
    from torch.utils.data import DataLoader
    from utils.others.img_io import show_image, show_array_3d
    opt = CN(new_allowed=True)

    opt.dataroot = './traces/datasets/prostate_daf3d_pre'
    opt.phase = 'test'
    opt.crop_size = 96
    opt.stride = 96
    opt.no_augment = False

    test_dataset = TestTrusDataset(opt)
    #  {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}
    print('test_dataset:{}'.format(len(test_dataset)))
    for data in test_dataset:
        print(data['volume'].shape)     # (175, 224, 224)
        show_image(data['volume'][:, :, 100], title='origin image')

        one_patient_dataset = TestOnePatientDataset(data['volume'][:, :, 100], opt)
        print('one_patient_dataset:{}'.format(len(one_patient_dataset)))

        dataset_info = one_patient_dataset.get_info()   # 'crop_size' 'stride' 'origin_shape'  'pad_shape'
        dataset_volumes = one_patient_dataset.get_volume()   # 'origin_volume'  'pad_volume'
        row, column = one_patient_dataset.get_crop_num_list()

        test_dataloader = DataLoader(one_patient_dataset,
                                     batch_size=len(one_patient_dataset),
                                     shuffle=False,
                                     num_workers=8,
                                     drop_last=False)
        print('test_dataloader:{}'.format(len(test_dataloader)))
        for test_data in test_dataloader:
            print(test_data.shape)      # N C ...
            data_to_show = test_data[:, 0, ...].numpy()
            show_array_3d(data_to_show, row, column, title='crop_image')
            # 还原的时候，axis的顺序是由大到小，2D先1后0，3D是210。也就是从循环的最深层开始，逐层还原
            # concat_array = [np.concatenate(data_to_show[i*column:i*column+column], axis=1) for i in range(row)]
            # show_image(concat_array[1], title='partly concat image')
            # concat_array = np.concatenate(concat_array, 0)
            # show_image(concat_array, title='concat image')
            for kk in range(test_data.shape[1]):
                data_to_show = test_data[kk].numpy()
                show_array_3d(data_to_show, 2, 2, title='crop_image')
                break

            pass
        break


def main():
    # test_val_dataset()
    data_path = r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/results/' \
                r'trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4/test/' \
                r'slide_test_pad_noaug/65_net_trus_unet3d_DDP_SynBN_crop128_bs3x4_ch32_dc_adam_1e-4id-3.h5'
    patient_id = re.match(r'^/(?:.+/)*((\d+).*)\.h5$', data_path).groups()[-1]
    fr = h5py.File(data_path, 'r')
    label = fr.get('label')[:]
    segment = fr.get('segment')[:]
    volume = fr.get('pad_volume')[:]
    fr.close()
    show_volume_label_predict(volume.transpose((2, 1, 0)),
                              label.transpose((2, 1, 0)),
                              segment.transpose((2, 1, 0)),
                              True,
                              row=3, col=2, title=f'test on patient: {patient_id} ')

    pass


def try_multiprocess():
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()


def f(name):
    info('function f')
    print('hello', name)


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def isVaildsStr(S, L):
    assert isinstance(S, str)
    assert isinstance(L, str)
    if len(S) > len(L):
        return False
    vaild_char = []
    S_char = []
    for c in S:
        S_char.append(c)
    for i in range(len(L)):
        if S_char:
            print(S_char)
            if S_char[0] == L[i]:
                vaild_char.append(i)
                S_char.pop(0)
        else:
            print('vaild_char:', vaild_char)
            return True
    if S_char:
        return False
    else:
        print('vaild_char:', vaild_char)
        return True


def combineWord(start, total, *args):
    out_str = ''
    args_len = []
    args_list = []
    for arg in args:
        if isinstance(arg, str):
            args_len.append(len(arg))
            args_list.append(arg)
    args_len = args_len[:total]
    args_list = args_list[:total]
    out_str += args_list[start]
    args_len.pop(start)
    args_list.pop(start)
    for arg_l, arg in zip(args_len, args_list):
        if arg[0] != out_str[0]:
            args_len.remove(arg_l)
            args_list.remove(arg)
    max_len = max(args_len)
    for arg_l, arg in zip(args_len, args_list):
        if arg_l != max_len:
            args_len.remove(arg_l)
            args_list.remove(arg)
    min_str = args_list[0]
    for arg in args_list:
        if arg < min_str:
            min_str = arg
    out_str += min_str
    print(out_str)
    return out_str


def testfun(start, total, *args):
    print('args type:', type(args))
    ss='ace'
    ll='abcde'
    print(isVaildsStr(ss, ll))
    combineWord(4,6,'word','dd','da','dc','dword','d')


def get_gauusian_kernel(shape):
    if len(shape) == 2:
        X, Y = shape
        target = np.zeros(shape, dtype=np.float32)
        for x in range(X):
            x_i = x if x <= (X - 1)/2 else (X - 1) - x
            for y in range(Y):
                y_i = y if y <= (Y - 1)/2 else (Y - 1) - y
                target[x, y] = x_i + y_i
        # t_max = (X - 1)//2 + (Y - 1)//2
        # assert t_max == target.max()
        # target = target / t_max
        return target


def get_gauusian_kernel_v2(shape):
    dims = len(shape)
    target = np.zeros(shape, dtype=np.float32)
    # assert dims in (1, 2, 3, 4)
    for axis in range(dims):
        shp = shape[axis]

        target_tmp = target if axis == 0 else np.swapaxes(target, 0, axis)

        for i in range(shp):
            val = i if i <= (shp - 1)/2 else (shp - 1) - i
            target_tmp[i, ...] += val

        target = target_tmp if axis == 0 else np.swapaxes(target, 0, axis)
    return target


def check_predicted_results(result_dir=r'/raid/lf/PROJECT/DLForPytorch/traces/results/trus_unet3d_DDP_Sybn_crop128_bs2x4_ch32_kaiming_dc_adam_1e-4_step_0.2_warmup_10_5e-5/val/crop128_slide24_nopad_noaug/'):

    result = glob(result_dir+'*.h5')
    # <KeysViewHDF5 ['label', 'origin_volume', 'segment']>
    for data_path in result:
        data_name = os.path.basename(data_path).split('.')[0]
        volume, segment, label = h5_loader(data_path, 'volume', 'segment', 'label')
        show_volume_label_predict(volume, segment, label, add_line=True, row=5, col=5, title=data_name)


if __name__ == '__main__':
    # for tt in [(3,3,), (3,4), (3,5), (4, 4), (4,5), (5,4), (5,5), (3,3,3)]:
    #     print('get_gauusian_kernel:')
    #     print(get_gauusian_kernel(tt))
    #     print('get_gauusian_kernel_v2:')
    #     print(get_gauusian_kernel_v2(tt))
    result_dir = r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/results/promise12_unet3dV1_crop969632_bs6_ch32_kaiming_combo_1.0_1.5_adam_5e-4_step_0.3_100_warmup_10_5e-5/test/centercrop/'
    check_predicted_results(result_dir)
