import os
import h5py
import torch
import logging
import importlib
import torch.distributed
from glob import glob
from types import SimpleNamespace
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from skimage.transform import resize, rescale
from data.dataloads.base_dataset import BaseDataset, TestOnePatientDataset
# from models.modules.segmentation.three_d.unet3d_V0 import UNet3D
from models.modules.segmentation.three_d.unet3d_V0 import UNet3D as UNetV0
from models.modules.segmentation.three_d.unet3d_V1 import UNet3D as UNetV1
from models.modules.segmentation.three_d.unet3d_V2 import UNet3D as UNetV2
from models.modules.segmentation.three_d.unet3d_V3 import UNet3D as UNetV3
from configs.utils_config import pretty_print_opt, get_pretty_opt, get_config
from utils.forLogs import get_logger
from utils.others.metrics import BinaryMetrics, MutiClassMetrics
from utils.others.utils import init_seed, init_torch, mkdirs, Timer, convert_str_to_list
from utils.others.img_io import show_paired_image, show_array_3d, show_volume_label, show_volume_label_predict


def main_predict():
    # opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d_test.yaml', '--use_config'], save_log=False)
    # opt_dict = get_config('configs/defaults/trus_unet3d_predict.yaml')
    opt_dict = get_config('configs/defaults/promise_unet3d_predict.yaml')
    opt = SimpleNamespace(**opt_dict)

    print(torch.cuda.is_available())
    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    print(torch.cuda.is_available())

    do_predict(opt)


def do_predict(opt):
    print('now is in do_predict')

    if opt.local_gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.local_gpu)
        torch.cuda.empty_cache()
    else:
        opt.local_gpu = -1

    # binary_metrics = BinaryMetrics()
    # multi_metrics = MutiClassMetrics()
    log_dir = os.path.join(opt.results_dir, opt.name, opt.phase, opt.predict_name)
    mkdirs(log_dir)

    opt_logger = get_logger(logname='opt_logger', is_save=not opt.DEBUG,
                            save_name=os.path.join(log_dir, 'option.txt'), fmt='%(message)s')
    opt_logger.info(get_pretty_opt(opt))
    preprocess = opt.preprocess
    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

    get_metrics = BinaryMetrics()
    print('predict prepare finished')

    if opt.weight_on_dir:
        weight_paths = glob(os.path.join(opt.weight_dir, '*.pth'))
    else:
        weight_paths = [opt.weight_path]
    print('weight_paths ready')

    predict_dataset_class = create_predict_dataset(opt.dataset_name)
    test_dataset = predict_dataset_class(opt)
    print('test_dataset:{}'.format(len(test_dataset)))
    # {'volume', 'label', 'volume_path', 'label_path', 'origin_shape', 'now_shape'}

    # 因为要求每个病人的结果，因此一般设batch_size=1
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_threads,
                                 pin_memory=False,
                                 drop_last=False)
    print('test_dataloader:{}'.format(len(test_dataloader)))

    test_network = define_net(opt)
    best_dice = 0.5
    best_weight = weight_paths[0]
    for weight_path in weight_paths:
        test_network = load_weithts(test_network, weight_path, device, name='segment')
        if opt.eval:
            test_network.eval()
        print('network created', type(test_network))

        checkpoint_name = os.path.basename(weight_path).split('.')[0]
        log_name = 'message_logger'+checkpoint_name
        log_path = os.path.join(log_dir, checkpoint_name+preprocess+'.txt')
        message_logger = get_logger(logname=log_name, is_save=not opt.DEBUG, save_name=log_path, fmt='%(message)s')
        print('visualizer created')

        all_result_metrics = []
        all_result_visuals = []
        for patient_id, data in enumerate(test_dataloader):
            volume_name = os.path.basename(data['volume_path'][0]).split('.')[0]
            test_data = data['volume'].cuda()

            # show_volume_label(data['volume'].cpu().numpy()[0, 0],
            #                   data['label'].cpu().numpy()[0, 0],
            #                   row=4, col=4, title=f'predict paired {patient_id}')

            with torch.no_grad():
                test_result = test_network(test_data)
                test_result = torch.sigmoid(test_result)  # NCDHW, N=1, C=1
            test_result_mask = torch.where(test_result > 0.5, 1, 0)

            segment = test_result_mask.cpu().numpy()[0, 0]
            label = data['label'].cpu().numpy()[0, 0]
            volume = data['volume'].cpu().numpy()[0, 0]

            if opt.revert:
                segment = resize(segment.astype(float), tuple(data['origin_shape']), order=0, mode="constant", cval=0,
                                 clip=True, preserve_range=False, anti_aliasing=False)
                label = resize(label.astype(float), tuple(data['origin_shape']), order=0, mode="constant", cval=0,
                               clip=True, preserve_range=False, anti_aliasing=False)
                volume = resize(volume.astype(float), tuple(data['origin_shape']), order=3, mode="constant", cval=0,
                                clip=True, preserve_range=False, anti_aliasing=False)

            # show_volume_label_predict(volume, label, segment, row=4, col=4, title=f'predict paired {patient_id}')

            metrics = compute_metrics_by_patient(segment, label, *opt.metric_names,
                                                 get_metrics=get_metrics, need_key=True)
            visual = compute_visual_by_patient(segment, label, *opt.visual_names, volume=volume)
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


def define_net(opt):

    # net = UNet3D(in_channels=opt.input_nc,
    #              out_channels=opt.output_nc,
    #              final_sigmoid=True,
    #              conv_layer_order=opt.conv_order,
    #              init_channel_number=opt.init_channel_number)
    net = UNetV1(in_channels=opt.input_nc, out_channels=opt.output_nc, init_features=opt.init_channel_number)  # cbr

    net = net.cuda()

    if opt.verbose:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net, '\n[Network %s] Total number of parameters : %.3f M' % (opt.model_name, num_params / 1e6))

    return net


def load_weithts(net, weight_path, device, name='segment'):
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
    if opt.save_visuals:
        assert 'vis_name' in kwargs.keys(), 'you have to apply vis_name'
        vis_name = kwargs['vis_name']
        save_name = os.path.join(opt.results_dir, opt.name, opt.phase, opt.predict_name, vis_name+'.h5')
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


if __name__ == '__main__':
    main_predict()


