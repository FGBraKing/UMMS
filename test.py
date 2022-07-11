import os
import sys
import time
import tqdm
import h5py
import torch
import logging
import importlib
import numpy as np
import torch.distributed
from types import SimpleNamespace
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from glob import glob
from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt, get_pretty_opt, get_config
from data.connected_components import retain_the_largest_connected_component_binary
from data.dataloads.base_dataset import BaseDataset, TestOnePatientDataset
from data.utils_data import get_pad_image, get_unpad_image
# from models.modules.segmentation.three_d.unet3d_V0 import UNet3D
# from models.modules.segmentation.three_d.unet3d_V0 import UNet3D as UNetV0
# from models.modules.segmentation.three_d.unet3d_V1 import UNet3D as UNetV1
# from models.modules.segmentation.three_d.unet3d_V2 import UNet3D as UNetV2
# from models.modules.segmentation.three_d.unet3d_V3 import UNet3D as UNetV3
from models.modules.segmentation_model.unet_custom import UnetCustom as UNet
from utils.forLogs import get_logger
from utils.others.metrics import BinaryMetrics
from utils.others.utils import init_seed, init_torch, mkdirs, Timer, print_numpy
from utils.others.img_io import show_paired_image, show_array_3d, show_volume_label, show_volume_label_predict
from argparse import ArgumentParser, REMAINDER, ZERO_OR_MORE, OPTIONAL


def test():
    parser = ArgumentParser(description="Project's useful tool to parse args")
    parser.add_argument('--config_name', type=str, default='mrusmr_unet', help='the name of config')
    parser.add_argument('--second', type=int, default=1, help='wait some second and then run')
    parser.add_argument('training_script_args', nargs=REMAINDER, help='training_script_args')
    args = parser.parse_args()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d_test.yaml', '--use_config'], save_log=False)
    # opt_dict = get_config('configs/defaults/trus_unet3d_test.yaml')
    # opt_dict = get_config('configs/defaults/promise_unet3d_test.yaml')
    config_name = args.config_name
    opt_dict = get_config(f'configs/defaults/{config_name}_test.yaml')
    opt = SimpleNamespace(**opt_dict)
    time.sleep(args.second)

    opt.visible_gpu = str(opt.visible_gpu)
    print('torch.cuda.is_available:', torch.cuda.is_available(), opt.local_gpu, opt.phase, opt.visible_gpu)
    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)

    do_test(opt)


def do_test(opt):
    print('now is in do_test')
    # if opt.local_gpu >= 0 and torch.cuda.is_available():
    #     torch.cuda.set_device(opt.local_gpu)
    #     torch.cuda.empty_cache()
    # else:
    #     opt.local_gpu = -1
    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

    log_dir = os.path.join(opt.results_dir, opt.name, opt.phase, opt.test_name)
    mkdirs(log_dir)
    opt_logger = get_logger(logname='opt_logger', is_save=not opt.DEBUG and opt.weight_on_dir,
                            save_name=os.path.join(log_dir, 'option.txt'), fmt='%(message)s')
    opt_logger.info(get_pretty_opt(opt))
    print('logger finished')

    get_metrics = BinaryMetrics()
    print('BinaryMetrics finished')

    if opt.weight_on_dir:
        weight_paths = glob(os.path.join(opt.weight_dir, '*.pth'))
    else:
        # weight_paths = [opt.weight_path]
        weight_paths = [os.path.join(opt.weight_dir, opt.weight_name)]
    if len(weight_paths) == 0:
        opt_logger.info(f'there is a empty weight_dir!')
        return
    print('weight_paths ready')

    test_dataset_class = create_test_dataset(opt.dataset_name)
    test_dataset = test_dataset_class(opt)
    print('test_dataset:{}'.format(len(test_dataset)))

    test_network = define_net(opt, device)
    print('network created', type(test_network))

    best_dice = 0.0
    best_weight = weight_paths[0]
    for weight_path in weight_paths:
        test_network = load_weithts(test_network, weight_path, device, name='segment')
        print('weights loaded', type(test_network))
        if opt.eval:
            test_network.eval()

        checkpoint_name = os.path.basename(weight_path).split('.')[0]
        log_name = 'message_logger'+checkpoint_name
        log_path = os.path.join(log_dir, checkpoint_name+'.txt')
        message_logger = get_logger(logname=log_name, is_save=not opt.DEBUG, save_name=log_path, fmt='%(message)s')
        tensor_dir = os.path.join(log_dir, checkpoint_name+'tensorboard_log')
        # writer = SummaryWriter(logdir=tensor_dir, flush_secs=120, filename_suffix=opt.name, write_to_disk=True)
        print('visualizer created')

        all_result_metrics = []
        all_result_visuals = []
        for patient_id, data in enumerate(test_dataset):
            volume_name = os.path.basename(data['volume_path']).split('.')[0]
            #  {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}
            print(patient_id, ':', volume_name)
            one_patient_dataset = TestOnePatientDataset(data['volume'], opt)    # CDHW
            print('one_patient_dataset:{}'.format(len(one_patient_dataset)))
            dataset_info = one_patient_dataset.get_info()   # 'crop_size' 'stride' 'origin_shape'  'pad_shape'
            dataset_volumes = one_patient_dataset.get_volume()   # 'origin_volume'  'pad_volume'
            crop_num_list = one_patient_dataset.get_crop_num_list()
            axis_tuple = one_patient_dataset.get_axis()

            test_dataloader = DataLoader(one_patient_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=False,
                                         num_workers=opt.num_threads,
                                         pin_memory=True,
                                         drop_last=False)
            print('test_dataloader:{}'.format(len(test_dataloader)))

            with torch.no_grad():
                sub_segments = test_by_patient(test_dataloader, test_network, device)    # tensor NCDHW
                sub_segments = torch.sigmoid(sub_segments)

                kwargs = {'do_connected_component': opt.do_connected_component,
                          'minimum_valid_object_size': opt.minimum_valid_object_size,
                          'use_gauusion_kernel': opt.use_gauusion_kernel}
                segment_regain_dict = compute_segment_by_patient(sub_segments,  dataset_info, crop_num_list, axis_tuple,
                                                                 SimpleNamespace(**kwargs), data['spacing'])

                segment = segment_regain_dict['padded'] if opt.compute_pad else segment_regain_dict['origin']
                label = get_pad_image(data['label'],
                                      crop_size=dataset_info['crop_size'],
                                      stride=dataset_info['stride'],
                                      common_order=False,
                                      mode='minimum') if opt.compute_pad else data['label']

                # show_volume_label(segment, label, 5, 5, title='padded segment label')

                metrics = compute_metrics_by_patient(segment, label, *opt.metric_names,
                                                     get_metrics=get_metrics, need_key=True,
                                                     voxelspacing=data['spacing'])
                visual = compute_visual_by_patient(segment, label, *opt.visual_names, **dataset_volumes)
                show_result_by_patient(opt, patient_id, metrics=metrics, visuals=visual, message_logger=message_logger,
                                       vis_name=checkpoint_name[:6]+volume_name)
                all_result_metrics.append(metrics)
                all_result_visuals.append(visual)

        total_metrics = combine_and_show_result_all(opt,
                                                    metrics_list=all_result_metrics,
                                                    visuals_list=all_result_visuals,
                                                    message_logger=message_logger)
        if total_metrics['dice'] > best_dice:
            best_dice = total_metrics['dice']
            best_weight = weight_path

    opt_logger.info(f'best_dice: {best_dice}\n'
                    f'best_weight: {best_weight}')


def test_during_train(one_patient, model, opt, device):
    # crop_size: WHD
    # stride: WHD
    # no_augment： 扩增
    # metric_names
    # visual_names
    one_patient_dataset = TestOnePatientDataset(one_patient['volume'], opt)  # CDHW
    dataset_info = one_patient_dataset.get_info()  # 'crop_size' 'stride' 'origin_shape'  'pad_shape'
    dataset_volumes = one_patient_dataset.get_volume()  # 'origin_volume'  'pad_volume'
    crop_num_list = one_patient_dataset.get_crop_num_list()
    axis_tuple = one_patient_dataset.get_axis()

    test_dataloader = DataLoader(one_patient_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_threads,
                                 pin_memory=True,
                                 drop_last=False)
    with torch.no_grad():
        result_list = []
        for data in test_dataloader:  # NCDHW
            data = data.to(device)
            channel_result_list = [model(data[:, channel:channel + 1, ...]) for channel in range(data.shape[1])]
            result_list.append(torch.cat(channel_result_list, dim=1))
        sub_segments = torch.cat(result_list, dim=0)        # tensor NCDHW
        sub_segments = torch.sigmoid(sub_segments)

        kwargs = {'do_connected_component': False, 'minimum_valid_object_size': 1000, 'use_gauusion_kernel': False}
        segment_regain_dict = compute_segment_by_patient(sub_segments, dataset_info, crop_num_list, axis_tuple,
                                                         SimpleNamespace(**kwargs),  one_patient['spacing'])
        segment = segment_regain_dict['origin']
        label = one_patient['label']
        metrics = compute_metrics_by_patient(segment, label, *opt.metric_names,
                                             get_metrics=BinaryMetrics(), need_key=True,
                                             voxelspacing=one_patient['spacing'])
        visual = compute_visual_by_patient(segment, label, *opt.visual_names, **dataset_volumes)
    return metrics, visual


def create_test_dataset(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data.dataloads." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = 'test' + dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    return dataset


def define_net(opt, device=torch.device('cpu')):
    # net = UNet3D(in_channels=opt.input_nc,
    #              out_channels=opt.output_nc,
    #              final_sigmoid=True,
    #              conv_layer_order=opt.conv_order,
    #              init_channel_number=opt.init_channel_number)
    net = UNet(norm_type='batch', in_channels=opt.input_nc, n_class=opt.output_nc, deptp=4,
               init_channel_number=opt.init_channel_number, final_sigmoid=False)   # cbr
    # net = net.cuda()
    net = net.to(device)

    if opt.verbose:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net, '\n[Network %s] Total number of parameters : %.3f M' % (opt.model_name, num_params / 1e6))

    return net


def load_weithts(net, weight_path, device=torch.device('cpu'), name='segment'):
    print('loading the model from %s' % weight_path)
    state_dict = torch.load(weight_path, map_location=device)

    if name in state_dict.keys():
        net_state_dict = state_dict.get(name)
    else:
        net_state_dict = state_dict

    net.load_state_dict(net_state_dict)
    return net


def test_by_patient(dataloader, model, device=torch.device('cpu')):
    result_list = []
    for data in dataloader:     # NCDHW
        data = data.to(device)
        channel_result_list = [model(data[:, channel:channel+1, ...]) for channel in range(data.shape[1])]
        result_list.append(torch.cat(channel_result_list, dim=1))

        # show_volume_label(data.cpu().numpy()[0, :, 10, :, :], result_list[-1].cpu().numpy()[0, :, 10, :, :], 3, 3,
        #                   title='CHW view')
    return torch.cat(result_list, dim=0)


def compute_segment_by_patient(sub_volumes, dataset_info, crop_num_list, axises, opt, spacing=1):
    '''
    :param sub_volumes:tensor,NCDHW
    :param dataset_info: {'crop_size': self.crop_size, 'stride': self.stride,
                        'origin_shape': self.origin_size, 'pad_shape': self.padded_size}
    :param crop_num_list:  [int,int,int]
    :param axises: (tuple,tuple)
    :param opt: option for do_connected_component,
    :param spacing :
    :return: {'origin': origin_segment, 'padded': padded_segment}
    '''
    assert sub_volumes.shape[0] == np.prod(crop_num_list)
    assert sub_volumes.shape[1] == len(axises) + 1
    sub_volumes_mask = torch.where(sub_volumes > 0.5, 1, 0)     # NCDHW
    # ============================================combine filp=======================================
    sub_volumes_mask = sub_volumes_mask.int()
    for ind, axis in enumerate(axises):
        shift_axis = tuple([a+1 for a in axis])
        sub_volumes_mask[:, ind+1, ...] = torch.flip(sub_volumes_mask[:, ind+1, ...], dims=shift_axis)
    # print(sub_volumes_mask.size())
    try:
        sub_volumes_mask = torch.sum(sub_volumes_mask, dim=1)         # NDHW  , keepdim=True
    except RuntimeError:
        sub_volumes_mask = torch.sum(sub_volumes_mask.cpu(), dim=1)         # NDHW  , keepdim=True
    sub_volumes_mask = torch.where(sub_volumes_mask > sub_volumes.shape[1]/2, 1, 0)

    # ========================================combine slide=========================================
    crop_size = dataset_info['crop_size']
    stride = dataset_info['stride']
    pad_shape = dataset_info['pad_shape']

    seg_mask = torch.zeros(size=pad_shape, dtype=torch.int32, device=sub_volumes_mask.device)
    threshold = torch.zeros(size=pad_shape, dtype=torch.float32, device=sub_volumes_mask.device)
    std_weights = torch.ones_like(sub_volumes_mask[0], dtype=torch.int32, device=sub_volumes_mask.device)
    if opt.use_gauusion_kernel:
        std_weights = get_gauusian_kernel_v2(sub_volumes_mask[0].shape, dtype=torch.int32, device=sub_volumes_mask.device)

    ndim = len(crop_num_list)
    assert ndim == len(crop_size) == len(stride) == len(crop_num_list)
    assert tuple(crop_size) == tuple(sub_volumes_mask.shape[1:])
    if ndim == 2:
        for i in range(crop_num_list[0]):
            for j in range(crop_num_list[1]):
                seg_mask[i*stride[0]:i*stride[0]+crop_size[0],
                         j*stride[1]:j*stride[1]+crop_size[1]] += sub_volumes_mask[i*crop_num_list[1]+j] * std_weights
                threshold[i*stride[0]:i*stride[0]+crop_size[0],
                          j*stride[1]:j*stride[1]+crop_size[1]] += std_weights
    elif ndim == 3:
        for i in range(crop_num_list[0]):
            for j in range(crop_num_list[1]):
                for k in range(crop_num_list[2]):
                    seg_mask[i*stride[0]:i*stride[0]+crop_size[0],
                             j*stride[1]:j*stride[1]+crop_size[1],
                             k*stride[2]:k*stride[2]+crop_size[2]] += \
                        sub_volumes_mask[i*crop_num_list[1]*crop_num_list[2]+j*crop_num_list[2]+k] * std_weights

                    threshold[i*stride[0]:i*stride[0]+crop_size[0],
                              j*stride[1]:j*stride[1]+crop_size[1],
                              k*stride[2]:k*stride[2]+crop_size[2]] += std_weights
    else:
        for c in range(crop_num_list[0]):
            for i in range(crop_num_list[1]):
                for j in range(crop_num_list[2]):
                    for k in range(crop_num_list[3]):
                        seg_mask[
                            c*stride[0]:c*stride[0]+crop_size[0],
                            i*stride[1]:i*stride[1]+crop_size[1],
                            j*stride[2]:j*stride[2]+crop_size[2],
                            k*stride[3]:k*stride[3]+crop_size[3]
                        ] += sub_volumes_mask[c*crop_num_list[1]*crop_num_list[2]*crop_num_list[3] +
                                              i*crop_num_list[2]*crop_num_list[3] + j*crop_num_list[3] + k] * std_weights
                        threshold[
                            c*stride[0]:c*stride[0]+crop_size[0],
                            i*stride[1]:i*stride[1]+crop_size[1],
                            j*stride[2]:j*stride[2]+crop_size[2],
                            k*stride[3]:k*stride[3]+crop_size[3]] += std_weights

    seg_mask = torch.where(seg_mask > threshold/2, 1, 0)
    padded_segment = seg_mask.cpu().numpy()
    if opt.do_connected_component:
        volume_per_voxel = float(np.prod(spacing, dtype=np.float64))
        padded_segment, _, _ = retain_the_largest_connected_component_binary(padded_segment, volume_per_voxel,
                                                                             opt.minimum_valid_object_size)
    origin_segment = get_unpad_image(pad_shape, dataset_info['origin_shape'], padded_segment)

    return {'origin': origin_segment, 'padded': padded_segment}


def compute_metrics_by_patient(predict, target, *metric_names, get_metrics=None, need_key=True, **kwargs):
    if get_metrics is None:
        get_metrics = BinaryMetrics()
    # 加上求hd95、assd、ravd等参数。voxelspacing=xxx、connectivity=xxx
    metrics = get_metrics(predict, target, *metric_names, **kwargs)
    keys = tuple(metric_names)
    metrics_dict = dict(zip(keys, metrics))
    if need_key:
        return metrics_dict
    else:
        return metrics


def compute_visual_by_patient(predict, target, *names, need_key=True, **kwargs):
    keys = tuple(names)
    visuals = []
    for name in names:
        if name == 'segment':
            visuals.append(predict)
        elif name == 'label':
            visuals.append(target)
        else:
            try:
                visuals.append(kwargs[name])
            except RuntimeError:
                print('the {} is not on your keys{}'.format(name, kwargs.keys()))
                raise KeyError
    if need_key:
        return dict(zip(keys, visuals))
    else:
        return visuals


# TODO: 通过一定的参数设置，保存visual的结果到磁盘
def show_result_by_patient(opt, *args, metrics=None, visuals=None, **kwargs):
    if 'tensor_writer' in kwargs.keys():
        writer = kwargs['tensor_writer']
    else:
        # writer = SummaryWriter(write_to_disk=False)
        writer = None
    if 'message_logger' in kwargs.keys():
        message_logger = kwargs['message_logger']
    else:
        message_logger = get_logger('message_logger', is_save=False, fmt='%(message)s')

    if len(args) > 0:
        paitent_id = args[0]
    else:
        paitent_id = 0

    message = "number {} paitent metrics on experiment: {}\n".format(paitent_id, opt.name)
    for k, v in metrics.items():
        try:
            message += '%s: %.4f ' % (k, v)
        except TypeError as e:
            message += '%s: %r ' % (k, list(v))

    message_logger.info(message)

    if writer is not None:
        for name, image in visuals.items():
            if image.ndim == 2:
                image = torch.unsqueeze(image, dim=0)
            writer.add_image(tag=name, img_tensor=image, global_step=paitent_id)

    # other visualization
    if opt.save_visuals and not opt.DEBUG:
        assert 'vis_name' in kwargs.keys(), 'you have to apply vis_name'
        vis_name = kwargs['vis_name']
        save_name = os.path.join(opt.results_dir, opt.name, opt.phase, opt.test_name, vis_name+'.h5')
        with h5py.File(save_name, mode='w') as fw:
            for name, image in visuals.items():
                image = image.clone().detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
                fw.create_dataset(name=name, data=image)


def combine_and_show_result_all(opt, *args, metrics_list=None, visuals_list=None, **kwargs):
    if 'tensor_writer' in kwargs.keys():
        writer = kwargs['tensor_writer']
    else:
        # writer = SummaryWriter(write_to_disk=False)
        writer = None
    if 'message_logger' in kwargs.keys():
        message_logger = kwargs['message_logger']
    else:
        message_logger = get_logger('message_logger', is_save=False, fmt='%(message)s')

    # combine
    patient_num = len(metrics_list)
    metrics_names = metrics_list[0].keys()
    total_metrics = {}
    for key in metrics_names:
        value = 0
        try:
            for v in metrics_list:
                value += v[key]
        except TypeError as e:
            print('some worng of key :{} with value: {}'.format(key, metrics_list[0][key]))
            continue
        value /= patient_num
        total_metrics[key] = value

    # show metrics
    message = "total metrics on experiment: {}\n".format(opt.name)
    for k, v in total_metrics.items():
        try:
            message += '%s: %.4f ' % (k, v)
        except TypeError as e:
            message += '%s: %r ' % (k, list(v))
    message_logger.info(message)

    # combine visuals
    if visuals_list is not None:
        pass

    # show visuals
    if writer is not None:
        pass
    return total_metrics


def get_gauusian_kernel_v2(shape, dtype=None, device=None):
    dims = len(shape)
    target = np.zeros(shape, dtype=np.int16)
    # assert dims in (1, 2, 3, 4)
    for axis in range(dims):
        shp = shape[axis]
        # torch.swapaxes 是从1.8.0才开始支持
        target_tmp = target if axis == 0 else np.swapaxes(target, 0, axis)
        for i in range(shp):
            val = i if i <= (shp - 1)/2 else (shp - 1) - i
            target_tmp[i, ...] += val
        target = target_tmp if axis == 0 else np.swapaxes(target_tmp, 0, axis)

    result = torch.from_numpy(target)
    if dtype is not None:
        result = result.type(dtype)
    if device is not None:
        result = result.to(device)
    return result


if __name__ == '__main__':
    test()


