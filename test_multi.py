import os
import sys
import time
import tqdm
import h5py
import torch
import logging
# import matplotlib
import numpy as np
import torch.distributed

from types import SimpleNamespace
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from glob import glob
from pprint import pprint
from matplotlib import pyplot as plt
from argparse import ArgumentParser, REMAINDER, ZERO_OR_MORE, OPTIONAL

from configs.excess_config import ex_config
from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt, get_pretty_opt, get_config
from data import create_dataset, create_test_dataset, create_predict_dataset
from data.connected_components import retain_the_largest_connected_component_binary
from data.dataloads.base_dataset import BaseDataset, TestOnePatientDataset
from data.utils_data import get_pad_image, get_unpad_image
from models import create_model, create_test_model
from utils.forLogs import Visualizer, get_logger
from utils.others.metrics import BinaryMetrics
from utils.others.utils import init_seed, init_torch, print_numpy, mkdirs, DataPool, get_device_name, Timer
from utils.others.distributed_utils import record_distribute_ddp, torch_distributed_zero_first
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict, show_image, show_paired_image
# matplotlib.use('TKAgg')

domain_map_modal = {'source': 'mr', 'target': 'us'}


def main_predict():
    parser = ArgumentParser(description="Project's useful tool to parse args")
    parser.add_argument('--config_name', type=str, default='adversarial', help='the name of config')
    parser.add_argument('--second', type=int, default=1, help='wait some second and then run')
    parser.add_argument('training_script_args', nargs=REMAINDER, help='training_script_args')
    args = parser.parse_args()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    opt_dict = get_config(f'configs/defaults/{args.config_name}_test.yaml')
    opt = SimpleNamespace(**opt_dict)
    time.sleep(args.second)

    opt.visible_gpu = str(opt.visible_gpu)
    print('torch.cuda.is_available:', torch.cuda.is_available(), opt.local_gpu, opt.phase, opt.visible_gpu)
    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)

    do_predict(opt)


def do_predict(opt):
    print('now is in do_predict')

    # device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

    results_dir = os.path.join(opt.results_dir, opt.name, opt.phase, opt.test_name)
    mkdirs(results_dir)
    opt_logger = get_logger(logname='opt_logger', is_save=not opt.DEBUG and opt.weight_on_dir,
                            save_name=os.path.join(results_dir, 'option.txt'), fmt='%(message)s')
    opt_logger.info(get_pretty_opt(opt))
    print('logger finished')

    if opt.weight_on_dir:
        weight_paths = glob(os.path.join(opt.checkpoints_dir, opt.name, '*.pth'))
    else:
        weight_paths = [os.path.join(opt.checkpoints_dir, opt.name, opt.weight_name)]
    if len(weight_paths) == 0:
        opt_logger.info(f'there is a empty weight_dir!')
        return
    print('weight_paths ready')

    predict_dataset_class = create_predict_dataset(opt.dataset_name)
    test_dataset = predict_dataset_class(opt)
    # {'volume', 'label', 'volume_path', 'label_path', 'origin_shape', 'now_shape', 'spacing'}
    print('test_dataset:{}'.format(len(test_dataset)))

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=opt.num_threads,
                                 pin_memory=False,
                                 drop_last=False)
    # 因为要求每个病人的结果，因此一般设batch_size=1
    print('test_dataloader:{}'.format(len(test_dataloader)))

    model = create_test_model(opt)
    print('network created', type(model))

    source_dice_pool = DataPool(2, 0.5)
    target_dice_pool = DataPool(2, 0.5)
    preprocess = opt.preprocess
    for weight_path in weight_paths:
        model.load_networks(weight_path)
        print('weights loaded', type(model))
        if opt.eval:
            model.eval()

        # checkpoint_name = os.path.basename(weight_path).split('.')[0]
        checkpoint_name = ".".join(os.path.basename(weight_path).split('.')[:-1])
        log_name = 'message_logger' + checkpoint_name
        log_path = os.path.join(results_dir, f"{checkpoint_name}_{preprocess}.txt")
        message_logger = get_logger(logname=log_name, is_save=not opt.DEBUG, save_name=log_path, fmt='%(message)s')
        print('visualizer created')

        all_source_result_metrics = []
        all_source_result_visuals = []
        all_target_result_metrics = []
        all_target_result_visuals = []

        for patient_id, data in enumerate(test_dataloader):
            volume_name = os.path.basename(data['mr_volume_path'][0]).split('.')[0]
            model.set_input(data)
            model.test()

            metrics_ret = model.get_current_metrics()
            # visual_ret = model.get_current_visuals()
            source_metrics_ret = model.get_current_metrics_by_domain('source')
            target_metrics_ret = model.get_current_metrics_by_domain('target')
            all_source_result_metrics.append(source_metrics_ret)
            all_target_result_metrics.append(target_metrics_ret)

            message = "number {} paitent {} metrics on experiment: {}\n".format(patient_id, volume_name, opt.name)
            for k, v in metrics_ret.items():
                try:
                    message += '%s: %.4f ' % (k, v)
                except TypeError as e:
                    message += '%s: %r ' % (k, list(v))
            message += '\n'
            message_logger.info(message)

            if opt.save_visuals:
                model.save_current_visuals(vis_name=checkpoint_name[:6]+volume_name)

        message_logger.info('source')
        source_total_metrics = combine_metrics(opt.name, metrics_list=all_source_result_metrics, message_logger=message_logger)
        message_logger.info('target')
        target_total_metrics = combine_metrics(opt.name, metrics_list=all_target_result_metrics, message_logger=message_logger)

        source_dice_pool.update(weight_path, source_total_metrics['dice'])
        target_dice_pool.update(weight_path, target_total_metrics['dice'])
    best_source_weight, best_source_dice = source_dice_pool.get_best_data()
    best_target_weight, best_target_dice = target_dice_pool.get_best_data()

    opt_logger.info(
        f'best_source_dice: {best_source_dice}\n'
        f'best_souece_weight: {best_source_weight}\n'
        f'best_target_dice: {best_target_dice}\n'
        f'best_target_weight: {best_target_weight}\n'
    )


def combine_metrics(name, metrics_list=None, **kwargs):
    message_logger = kwargs.get('message_logger', get_logger('message_logger', is_save=False, fmt='%(message)s'))
    # combine
    patient_num = len(metrics_list)
    metrics_names = metrics_list[0].keys()
    total_metrics = {}
    for key in metrics_names:
        value = 0
        try:
            for v in metrics_list:
                if v[key] > 0:
                    value += v[key]
                else:
                    value -= v[key]
        except TypeError as e:
            print('some worng of key :{} with value: {}'.format(key, metrics_list[0][key]))
            continue
        value /= patient_num
        total_metrics[key] = value

    # show metrics
    message = "total metrics on experiment: {}\n".format(name)
    for k, v in total_metrics.items():
        try:
            message += '%s: %.4f ' % (k, v)
        except TypeError as e:
            message += '%s: %r ' % (k, list(v))
    message_logger.info(message)
    return total_metrics


if __name__ == '__main__':
    main_predict()









