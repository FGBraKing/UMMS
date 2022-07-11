import os
import h5py
import torch
import logging
import importlib
import torch.distributed
import numpy as np
import torch.nn.functional as F
from glob import glob
from types import SimpleNamespace
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from skimage.transform import resize, rescale
from configs.utils_config import pretty_print_opt, get_pretty_opt, get_config
from data.dataloads.base_dataset import BaseDataset, TestOnePatientDataset
from data.connected_components import retain_the_largest_connected_component_binary
from models.modules.ummkd3d import UnetWithNormSpecficity
from utils.forLogs import get_logger
from utils.others.metrics import BinaryMetrics, MutiClassMetrics
from utils.others.utils import init_seed, init_torch, mkdirs, Timer, convert_str_to_list, DataPool
from utils.others.img_io import show_paired_image, show_array_3d, show_volume_label, show_volume_label_predict


domain_map_modal = {'source': 'mr', 'target': 'us'}


def main_predict():
    config_name = r'ummkd'
    opt_dict = get_config(f'configs/defaults/{config_name}_predict.yaml')
    opt = SimpleNamespace(**opt_dict)

    print(torch.cuda.is_available())
    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    print(torch.cuda.is_available())

    do_predict(opt)


def do_predict(opt):
    print('now is in do_predict')

    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

    log_dir = os.path.join(opt.results_dir, opt.name, opt.phase, opt.predict_name)
    mkdirs(log_dir)
    opt_logger = get_logger(logname='opt_logger', is_save=not opt.DEBUG,
                            save_name=os.path.join(log_dir, 'option.txt'), fmt='%(message)s')
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
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_threads,
                                 pin_memory=False,
                                 drop_last=False)
    # 因为要求每个病人的结果，因此一般设batch_size=1
    print('test_dataloader:{}'.format(len(test_dataloader)))

    test_network = define_net(opt, device)
    print('network created', type(test_network))

    source_dice_pool = DataPool(2, 0.5)
    target_dice_pool = DataPool(2, 0.5)

    preprocess = opt.preprocess
    for weight_path in weight_paths:
        test_network = load_weithts(test_network, weight_path, device, name=opt.networkname)
        print('weights loaded', type(test_network))
        if opt.eval:
            test_network.eval()
        checkpoint_name = os.path.basename(weight_path).split('.')[0]
        log_name = 'message_logger'+checkpoint_name
        log_path = os.path.join(log_dir, checkpoint_name+preprocess+'.txt')
        message_logger = get_logger(logname=log_name, is_save=not opt.DEBUG, save_name=log_path, fmt='%(message)s')
        print('visualizer created')

        all_source_result_metrics = []
        all_source_result_visuals = []
        all_target_result_metrics = []
        all_target_result_visuals = []
        for patient_id, data in enumerate(test_dataloader):
            volume_name = os.path.basename(data['mr_volume_path'][0]).split('.')[0]

            source_data = data['mr_volume'].to(device)
            target_data = data['us_volume'].to(device)

            # show_volume_label(data['mr_volume'].cpu().numpy()[0, 0],
            #                   data['mr_label'].cpu().numpy()[0, 0],
            #                   row=4, col=4, title=f'predict paired {patient_id}')

            with torch.no_grad():
                source_result = test_network(source_data, 'source')
                target_result = test_network(target_data, 'target')

                source_result = F.sigmoid(source_result)
                target_result = F.sigmoid(target_result)

            tmp_kwags = {
                'minimum_valid_object_size': opt.minimum_valid_object_size,
                'metric_names': opt.metric_names,
                'visual_names': opt.visual_names,
                'do_connected_component': opt.do_connected_component,
                'revert': opt.revert
            }
            source_metrics, source_visuals = process_predict_result(SimpleNamespace(**tmp_kwags), data, source_result, 'source')
            target_metrics, target_visuals = process_predict_result(SimpleNamespace(**tmp_kwags), data, target_result, 'target')

            show_result_by_patient(opt, patient_id, s_metrics=source_metrics, t_metrics=target_metrics,
                                   s_visuals=source_visuals, t_visuals=target_visuals,
                                   message_logger=message_logger, vis_name=checkpoint_name[:6]+volume_name)
            all_source_result_metrics.append(source_metrics)
            all_source_result_visuals.append(source_visuals)
            all_target_result_metrics.append(target_metrics)
            all_target_result_visuals.append(target_visuals)

        message_logger.info('source')
        source_total_metrics = combine_and_show_result_all(opt,
                                                           metrics_list=all_source_result_metrics,
                                                           visuals_list=all_source_result_visuals,
                                                           message_logger=message_logger)
        message_logger.info('target')
        target_total_metrics = combine_and_show_result_all(opt,
                                                           metrics_list=all_target_result_metrics,
                                                           visuals_list=all_target_result_visuals,
                                                           message_logger=message_logger)
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


def process_predict_result(opt, data, result, domain):
    '''
    :param opt: dict,keys: minimum_valid_object_size metric_names visual_names
    :param data:
    :param result:
    :param domain:
    :return:
    '''
    get_metrics = BinaryMetrics()
    print('BinaryMetrics finished')

    result_mask = torch.where(result > 0.5, 1, 0)

    segment = result_mask.cpu().numpy()[0, 0]
    label = data[domain_map_modal[domain] + '_label'].cpu().numpy()[0, 0]
    volume = data[domain_map_modal[domain] + '_volume'].cpu().numpy()[0, 0]

    if opt.do_connected_component:
        volume_per_voxel = float(np.prod(data[domain_map_modal[domain] + '_spacing'].mean(0).tolist(), dtype=np.float64))
        segment, kept_size, largest_removed = \
            retain_the_largest_connected_component_binary(segment, volume_per_voxel,
                                                          opt.minimum_valid_object_size)
        print(kept_size, largest_removed)

    if opt.revert:
        segment = resize(segment.astype(float), tuple(data[domain_map_modal[domain] + '_origin_shape']),
                         order=0, mode="constant", cval=0, clip=True, preserve_range=False, anti_aliasing=False)
        label = resize(label.astype(float), tuple(data[domain_map_modal[domain] + '_origin_shape']),
                       order=0, mode="constant", cval=0, clip=True, preserve_range=False, anti_aliasing=False)
        volume = resize(volume.astype(float), tuple(data[domain_map_modal[domain] + '_origin_shape']),
                        order=3, mode="constant", cval=0, clip=True, preserve_range=False, anti_aliasing=False)

    # show_volume_label_predict(volume, label, segment, row=4, col=4, title=f'predict paired {patient_id}')
    metrics = compute_metrics_by_patient(segment, label, *opt.metric_names,
                                         get_metrics=get_metrics, need_key=True,
                                         voxelspacing=data[domain_map_modal[domain] + '_spacing'].mean(0).tolist())
    visual = compute_visual_by_patient(segment, label, *opt.visual_names, volume=volume)

    return metrics, visual


def create_predict_dataset(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data.dataloads." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = 'predict' + dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    return dataset


def define_net(opt, device=torch.device('cpu')):
    net = UnetWithNormSpecficity(domains=['source', 'target'],
                                 norm_type='batch',
                                 in_channels=opt.input_nc,
                                 n_class=opt.output_nc,
                                 deptp=4, init_channel_number=opt.init_channel_number, final_sigmoid=False)

    net = net.to(device)

    if opt.verbose:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net, '\n[Network %s] Total number of parameters : %.3f M' % (opt.model_name, num_params / 1e6))

    return net


def load_weithts(net, weight_path, device, name='umms'):
    print('loading the model from %s' % weight_path)
    state_dict = torch.load(weight_path, map_location=device)

    if name in state_dict.keys():
        net_state_dict = state_dict.get(name)
    else:
        net_state_dict = state_dict

    net.load_state_dict(net_state_dict)
    return net


def compute_metrics_by_patient(predict, target, *metric_names, get_metrics=None, need_key=True, **kwargs):
    if get_metrics is None:
        get_metrics = BinaryMetrics()
    metrics = get_metrics(predict, target, *metric_names, **kwargs)
    keys = tuple(metric_names)
    metrics_dict = dict(zip(keys, metrics))
    if need_key:
        return metrics_dict
    else:
        return metrics


def compute_visual_by_patient(target, predict, *names, need_key=True, **kwargs):
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


def show_result_by_patient(opt, *args, s_metrics=None, t_metrics=None, s_visuals=None, t_visuals=None, **kwargs):
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
    for k, v in s_metrics.items():
        try:
            message += '%s: %.4f ' % ('s_'+k, v)
        except TypeError as e:
            message += '%s: %r ' % ('s_'+k, list(v))
    message += '\n'
    for k, v in t_metrics.items():
        try:
            message += '%s: %.4f ' % ('t_'+k, v)
        except TypeError as e:
            message += '%s: %r ' % ('t_'+k, list(v))
    message_logger.info(message)

    if writer is not None:
        for name, image in s_visuals.items():
            if image.ndim == 2:
                image = torch.unsqueeze(image, dim=0)
            writer.add_image(tag='s_'+name, img_tensor=image, global_step=paitent_id)
        for name, image in t_visuals.items():
            if image.ndim == 2:
                image = torch.unsqueeze(image, dim=0)
            writer.add_image(tag='t_' + name, img_tensor=image, global_step=paitent_id)

    # other visualization
    if opt.save_visuals:
        assert 'vis_name' in kwargs.keys(), 'you have to apply vis_name'
        vis_name = kwargs['vis_name']
        save_name = os.path.join(opt.results_dir, opt.name, opt.phase, opt.predict_name, vis_name+'.h5')
        with h5py.File(save_name, mode='w') as fw:
            for name, image in s_visuals.items():
                image = image.clone().detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
                fw.create_dataset(name='s_'+name, data=image)
            for name, image in t_visuals.items():
                image = image.clone().detach().cpu().numpy() if isinstance(image, torch.Tensor) else image
                fw.create_dataset(name='t_'+name, data=image)


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


if __name__ == '__main__':
    main_predict()


