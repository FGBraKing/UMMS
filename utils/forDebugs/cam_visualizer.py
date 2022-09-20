import os
import time

import cv2
import torch
import importlib
import numpy as np
import torch.utils.data
from math import ceil
from types import SimpleNamespace
from skimage.transform import resize
from matplotlib import pyplot as plt
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from models.loss import get_loss_criterion
from utils.others.utils import init_seed, init_torch, print_numpy
from data.dataloads.base_dataset import BaseDataset
from data.transforms.transformOnArray import normalize, NormalizeRange
from models.modules.segmentation_model.unet_custom import UnetCustom as UNet
from models.modules.MultimodalSegmentation.DualStream import DualStreamUnetV1, DualStreamUnetV2, DualStreamUnetV3, DualStreamUnetV4, SingleUnet
from utils.others.img_io import show_volume_label

to_std_image_uint8 = NormalizeRange(0, 255, np.uint8)
to_std_image_float32 = NormalizeRange(0, 1, np.float32)


class BaseCAM3D:
    def __init__(self, model, target_layer_name, loss_name, **kwargs):
        self.model = model.eval()
        self.target_layer_name = target_layer_name
        self.target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
        assert self.target_layer is not None, "There is no name: {} on model: {}".format(target_layer_name,
                                                                                         model.__class__)
        self.criterion = get_loss_criterion(name=loss_name, **kwargs)
        self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self, inputs, target, activations, grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target):
        return self.criterion(output, target)

    def __call__(self, input_tensor, target):
        output = self.activations_and_grads(input_tensor)

        self.model.zero_grad()
        _, loss = self.get_loss(output, target)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]       # C ×
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]               # C ×

        weights = self.get_cam_weights(input_tensor, target, activations, grads)                # C

        cam = np.zeros(activations.shape[1:], dtype=np.float32)                                 # *
        for i, w in enumerate(weights):
            cam += w * activations[i, ...]

        cam = np.maximum(cam, 0)                                                                # *

        cam = resize(cam, input_tensor.shape[2:])                                               # DHW, [::-1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GradCAM3D(BaseCAM3D):
    def __init__(self, model, target_layer_name, loss_name, **kwargs):
        super(GradCAM3D, self).__init__(model, target_layer_name, loss_name, **kwargs)

    def get_cam_weights(self, inputs, target, activations, grads):
        return np.mean(np.reshape(grads, (len(grads), -1)), axis=-1)


class GradCAMPlusPlus3D(BaseCAM3D):
    def __init__(self, model, target_layer_name, loss_name, **kwargs):
        super(GradCAMPlusPlus3D, self).__init__(model, target_layer_name, loss_name, **kwargs)

    def get_cam_weights(self, inputs, target, activations, grads):
        eps = 1e-6
        axis = tuple(range(1, len(activations.shape)))
        grads_power_2 = grads**2                                 # C *
        grads_power_3 = grads_power_2*grads                      # C *

        sum_activations = np.sum(activations, axis=axis)         # C
        sum_activations = np.expand_dims(sum_activations, axis)  # C *

        aij = grads_power_2 / (2*grads_power_2 + sum_activations*grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)                      # C *

        weights = np.maximum(grads, 0)*aij                      # (C *)x(C *)
        weights = np.sum(weights, axis=axis)                    # C
        return weights


class XGradCAM3D(BaseCAM3D):
    def __init__(self, model, target_layer_name, loss_name, **kwargs):
        super(XGradCAM3D, self).__init__(model, target_layer_name, loss_name, **kwargs)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        eps = 1e-7
        axis = tuple(range(1, len(activations.shape)))

        sum_activations = np.sum(activations, axis=axis)  # C
        sum_activations = np.expand_dims(sum_activations, axis)  # C *

        weights = grads * activations / (sum_activations + eps)
        weights = weights.sum(axis=axis)
        return weights


def add_cam_to_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.stack([img, img, img], 2)

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_volume_cam(volume, cam, row=5, col=5, title='number'):
    assert volume.ndim == 3 and cam.ndim == 3, "the array'dim is not 3"

    channel = volume.shape[0]  # D H W
    total = row * col

    volume = to_std_image_float32(volume)
    cam = to_std_image_float32(cam)

    for i in range(ceil(channel / total)):
        if total == 1:
            image = add_cam_to_image(volume[i, :, :], cam[i, :, :])
            fig = plt.figure(num=i)
            ax = fig.add_subplot(111)
            ax.imshow(image, cmap='jet')
            if title:
                ax.set_title(title)
            ax.axis('off')
            fig.show()
        else:
            fig, ax = plt.subplots(row, col)
            fig.suptitle(f'{title}:{i + 1:2d}')
            for j in range(row):
                for k in range(col):
                    if i * total + j * col + k < channel:
                        vol_i = volume[i * total + j * col + k, :, :]
                        cam_i = cam[i * total + j * col + k, :, :]
                        image = add_cam_to_image(vol_i, cam_i)

                        ax[j][k].imshow(image, cmap='jet')  # f'{i * 25 + j * 5 + k}'
                        ax[j][k].set_title(f'{i * total + j * col + k + 1}', fontsize=5, color='r')
                        ax[j][k].axis('off')
                        # ax[j][k].xlabel('x')
                        ax[j][k].set(xlabel='x', ylabel='y')
            fig.show()


def show_cam(cam, row=5, col=5, title='number', cmap='jet'):
    assert cam.ndim == 3, "the array'dim is not 3"
    channel = cam.shape[0]   # D H W
    total = row * col

    cam = to_std_image_float32(cam)
    for i in range(ceil(channel/total)):
        if total == 1:
            fig = plt.figure(num=i)
            ax = fig.add_subplot(111)
            ax.imshow(cam[i, :, :], cmap=cmap)
            if title:
                ax.set_title(title)
            ax.axis('off')
            fig.show()
        else:
            fig, ax = plt.subplots(row, col)
            fig.suptitle(f'{title}:{i+1:2d}')
            for j in range(row):
                for k in range(col):
                    if i * total + j * col + k < channel:
                        data = cam[i * total + j * col + k, :, :]
                        ax[j][k].imshow(data, cmap=cmap)  # f'{i * 25 + j * 5 + k}'
                        ax[j][k].set_title(f'{i * total + j * col + k + 1}', fontsize=5, color='r')
                        ax[j][k].axis('off')
                        # ax[j][k].xlabel('x')
                        ax[j][k].set(xlabel='x', ylabel='y')
            fig.show()


def show_cam_label(cam, label, interval=1, add_line=False, **kwargs):
    # kwargs: max_num\fix_num\normalize_per\title\col\row
    assert label.ndim == 3, "the dim of the gray volume must be 3 of D H W"
    assert cam.shape == label.shape
    label = label.astype(cam.dtype)

    cam_max, cam_min = np.max(cam), np.min(cam)
    index = label > 0.5
    label[index] = cam_max
    label[~index] = cam_min
    if add_line:
        med_line = np.ones(shape=label.shape[:-1], dtype=cam.dtype) * cam_max
        med_line = med_line[..., np.newaxis]
        volume_label = np.concatenate([cam, med_line, label], axis=-1)
    else:
        volume_label = np.concatenate([cam, label], axis=-1)

    show_cam(volume_label[::interval, ...], **kwargs)


def show_volume_cam_label(volume, cam, label, interval=1, add_line=False, **kwargs):
    assert label.ndim == 3, "the dim of the gray volume must be 3 of D H W"
    assert volume.shape == cam.shape == label.shape
    volume = to_std_image_float32(volume)
    cam = to_std_image_float32(cam)
    label = np.where(label > 0.5, 1, 0).astype(np.float32)
    if add_line:
        med_line = np.ones(shape=label.shape[:-1], dtype=np.float32)
        med_line = med_line[..., np.newaxis]
        volume_cam_label = np.concatenate([volume, med_line, cam, med_line, label], axis=-1)
    else:
        volume_cam_label = np.concatenate([volume, cam, label], axis=-1)
    show_cam(volume_cam_label[::interval, ...], **kwargs)


def define_net(opt, device):

    net = UNet(norm_type='batch', in_channels=opt.input_nc, n_class=opt.output_nc, deptp=4,
               init_channel_number=opt.init_channel_number, final_sigmoid=False)
    net = net.to(device)
    if opt.verbose:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net, '\n[Network %s] Total number of parameters : %.3f M' % (opt.model_name, num_params / 1e6))

    return net


def define_dualstream(opt, device, domains=None):

    if opt.network_type == "V1":
        net = DualStreamUnetV1(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=False,
                               final_sigmoid=True,
                               interpolation=opt.up_interpolate,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    elif opt.network_type == "V2":
        net = DualStreamUnetV2(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=False,
                               final_sigmoid=True,
                               interpolation=True,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    elif opt.network_type == "V3":
        net = DualStreamUnetV3(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=False,
                               final_sigmoid=True,
                               interpolation=True,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    elif opt.network_type == "V4":
        net = DualStreamUnetV4(in_channels=opt.input_nc,
                               out_channels=opt.output_nc,
                               domains=domains,
                               f_maps=opt.init_channel_number,
                               num_levels=5,
                               with_activation=False,
                               final_sigmoid=True,
                               interpolation=True,
                               norm_type="batch",
                               act_type="lrelu").to(device)
    else:
        net = SingleUnet(in_channels=opt.input_nc,
                         out_channels=opt.output_nc,
                         domains=domains,
                         f_maps=opt.init_channel_number,
                         num_levels=5,
                         with_activation=False,
                         final_sigmoid=True,
                         interpolation=True,
                         norm_type="batch",
                         act_type="lrelu").to(device)
    net = net.to(device)
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


def print_module_name(model):
    for name, module in model.named_modules():
        print(name)


def test_cam():
    kwargs = {
        'model_name': 'dualstream',
        'network_type': 'single',
        'input_nc': 1,
        'output_nc': 1,
        'domains': ('source', 'target'),
        'init_channel_number': 16,
        'verbose': True,
        'weight_path': '/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/PhaseBaseLine/DualStream/mrus11211280_fold0_bs4_SingleBoth_ch16_combo_1_1_1.5_l2_1e-4_adam_1e-4_poly_3x400_0.6_baseline_1080Ti/latest_net_mrus11211280_fold0_bs4_SingleBoth_ch16_combo_1_1_1.5_l2_1e-4_adam_1e-4_poly_3x400_0.6_baseline_1080Ti.pth',

        'dataset_name': 'mrusmr',
        'dataroot': '/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280',
        'phase': 'test',
        'fold': 0,
        'custom': True,
        'preprocess': 'centercrop',
        'crop_size': (112, 112, 80),
        'num_threads': 16,

        # encoders.4  decoders.0  decoders.1 decoders.2 decoders.3
        # decoders.3.basic_module.conv2.conv  out_conv.conv3d
        # encoders.4.basic_module.conv2.conv decoders.3.basic_module.conv2.conv   outconv.conv3d
        'layer_name': 'decoders.3.basic_module.conv2.conv',
        'loss_name': 'custom',

        'local_gpu': 1,
        'visible_gpu': '1,2,3',
        'deterministic': True
    }

    opt = SimpleNamespace(**kwargs)
    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

    predict_dataset_class = create_predict_dataset(opt.dataset_name)
    test_dataset = predict_dataset_class(opt)
    print('test_dataset:{}'.format(len(test_dataset)))
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_threads,
                                                  pin_memory=False,
                                                  drop_last=False)
    print('test_dataloader:{}'.format(len(test_dataloader)))

    # network = define_net(opt, device)
    network = define_dualstream(opt, device, domains=('source', 'target'))
    network = load_weithts(network, opt.weight_path, device, name='umms')   # segment
    network.eval()

    get_cam = GradCAM3D(network, opt.layer_name, opt.loss_name)
    # get_cam = GradCAMPlusPlus3D(network, opt.layer_name, opt.loss_name)
    # get_cam = XGradCAM3D(network, opt.layer_name, opt.loss_name)

    for patient_id, data in enumerate(test_dataloader):
        if patient_id not in [3]:
            continue
        volume_name = os.path.basename(data['volume_path'][0]).split('.')[0]
        print(patient_id, volume_name)
        volume = data['volume'].to(device)
        label = data['label'].to(device)

        cam = get_cam(volume, label)
        volume = volume.cpu().numpy()[0, 0]
        label = label.cpu().numpy()[0, 0]
        # show_volume_cam(volume, cam, row=3, col=3, title=volume_name)
        # show_cam(cam, row=3, col=3, title=volume_name)
        show_cam_label(cam, label, interval=1, add_line=True, row=3, col=3, title=volume_name)
        # show_volume_cam_label(volume, cam, label, interval=1, add_line=True, row=3, col=2, title=volume_name)
        # show_volume_label(volume, label, interval=1, add_line=True, row=3, col=2, title=volume_name)
        # input()
        # while input() != 1:
        #     pass
        # os.system("pause")
        # time.sleep(16)


def single_model_test():
    kwargs = {
        'model_name': 'unet',
        'input_nc': 1,
        'output_nc': 1,
        'init_channel_number': 16,
        'verbose': True,
        'weight_path': '../../traces/checkpoints/mrusus128_fold1_bs3_unet3d_ch16_combo_1_1_2_l2_5e-4_adam_5e-5_poly_3x300_0.6_2080Ti/252_net_mrusus128_fold1_bs3_unet3d_ch16_combo_1_1_2_l2_5e-4_adam_5e-5_poly_3x300_0.6_2080Ti.pth',

        'local_gpu': 0
    }
    opt = SimpleNamespace(**kwargs)
    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')
    network = define_net(opt, device)

    network = load_weithts(network, opt.weight_path, device, name='segment')
    network.eval()
    print_module_name(network)
    # in_conv
    # in_conv.conv
    # in_conv.conv.conv1
    # in_conv.conv.conv1.conv
    # in_conv.conv.conv1.norm
    # in_conv.conv.conv1.norm.norm
    # in_conv.conv.conv1.act
    # in_conv.conv.conv2
    # in_conv.conv.conv2.conv
    # in_conv.conv.conv2.norm
    # in_conv.conv.conv2.norm.norm
    # in_conv.conv.conv2.act
    # encoders
    # encoders.0
    # encoders.0.max_pool
    # encoders.0.double_conv
    # encoders.0.double_conv.conv1
    # encoders.0.double_conv.conv1.conv
    # encoders.0.double_conv.conv1.norm
    # encoders.0.double_conv.conv1.norm.norm
    # encoders.0.double_conv.conv1.act
    # encoders.0.double_conv.conv2
    # encoders.0.double_conv.conv2.conv
    # encoders.0.double_conv.conv2.norm
    # encoders.0.double_conv.conv2.norm.norm
    # encoders.0.double_conv.conv2.act
    # encoders.1
    # encoders.1.max_pool
    # encoders.1.double_conv
    # encoders.1.double_conv.conv1
    # encoders.1.double_conv.conv1.conv
    # encoders.1.double_conv.conv1.norm
    # encoders.1.double_conv.conv1.norm.norm
    # encoders.1.double_conv.conv1.act
    # encoders.1.double_conv.conv2
    # encoders.1.double_conv.conv2.conv
    # encoders.1.double_conv.conv2.norm
    # encoders.1.double_conv.conv2.norm.norm
    # encoders.1.double_conv.conv2.act
    # encoders.2
    # encoders.2.max_pool
    # encoders.2.double_conv
    # encoders.2.double_conv.conv1
    # encoders.2.double_conv.conv1.conv
    # encoders.2.double_conv.conv1.norm
    # encoders.2.double_conv.conv1.norm.norm
    # encoders.2.double_conv.conv1.act
    # encoders.2.double_conv.conv2
    # encoders.2.double_conv.conv2.conv
    # encoders.2.double_conv.conv2.norm
    # encoders.2.double_conv.conv2.norm.norm
    # encoders.2.double_conv.conv2.act
    # encoders.3
    # encoders.3.max_pool
    # encoders.3.double_conv
    # encoders.3.double_conv.conv1
    # encoders.3.double_conv.conv1.conv
    # encoders.3.double_conv.conv1.norm
    # encoders.3.double_conv.conv1.norm.norm
    # encoders.3.double_conv.conv1.act
    # encoders.3.double_conv.conv2
    # encoders.3.double_conv.conv2.conv
    # encoders.3.double_conv.conv2.norm
    # encoders.3.double_conv.conv2.norm.norm
    # encoders.3.double_conv.conv2.act
    # decoders
    # decoders.0
    # decoders.0.upsample
    # decoders.0.upsample.deconv
    # decoders.0.upsample.norm
    # decoders.0.upsample.norm.norm
    # decoders.0.upsample.act
    # decoders.0.double_conv
    # decoders.0.double_conv.conv1
    # decoders.0.double_conv.conv1.conv
    # decoders.0.double_conv.conv1.norm
    # decoders.0.double_conv.conv1.norm.norm
    # decoders.0.double_conv.conv1.act
    # decoders.0.double_conv.conv2
    # decoders.0.double_conv.conv2.conv
    # decoders.0.double_conv.conv2.norm
    # decoders.0.double_conv.conv2.norm.norm
    # decoders.0.double_conv.conv2.act
    # decoders.1
    # decoders.1.upsample
    # decoders.1.upsample.deconv
    # decoders.1.upsample.norm
    # decoders.1.upsample.norm.norm
    # decoders.1.upsample.act
    # decoders.1.double_conv
    # decoders.1.double_conv.conv1
    # decoders.1.double_conv.conv1.conv
    # decoders.1.double_conv.conv1.norm
    # decoders.1.double_conv.conv1.norm.norm
    # decoders.1.double_conv.conv1.act
    # decoders.1.double_conv.conv2
    # decoders.1.double_conv.conv2.conv
    # decoders.1.double_conv.conv2.norm
    # decoders.1.double_conv.conv2.norm.norm
    # decoders.1.double_conv.conv2.act
    # decoders.2
    # decoders.2.upsample
    # decoders.2.upsample.deconv
    # decoders.2.upsample.norm
    # decoders.2.upsample.norm.norm
    # decoders.2.upsample.act
    # decoders.2.double_conv
    # decoders.2.double_conv.conv1
    # decoders.2.double_conv.conv1.conv
    # decoders.2.double_conv.conv1.norm
    # decoders.2.double_conv.conv1.norm.norm
    # decoders.2.double_conv.conv1.act
    # decoders.2.double_conv.conv2
    # decoders.2.double_conv.conv2.conv
    # decoders.2.double_conv.conv2.norm
    # decoders.2.double_conv.conv2.norm.norm
    # decoders.2.double_conv.conv2.act
    # decoders.3
    # decoders.3.upsample
    # decoders.3.upsample.deconv
    # decoders.3.upsample.norm
    # decoders.3.upsample.norm.norm
    # decoders.3.upsample.act
    # decoders.3.double_conv
    # decoders.3.double_conv.conv1
    # decoders.3.double_conv.conv1.conv
    # decoders.3.double_conv.conv1.norm
    # decoders.3.double_conv.conv1.norm.norm
    # decoders.3.double_conv.conv1.act
    # decoders.3.double_conv.conv2
    # decoders.3.double_conv.conv2.conv
    # decoders.3.double_conv.conv2.norm
    # decoders.3.double_conv.conv2.norm.norm
    # decoders.3.double_conv.conv2.act
    # out_conv
    # out_conv.conv3d
    # out_conv.upsampling


def dualstream_model_test():
    kwargs = {
        'model_name': 'dualname',
        'network_type': 'single',
        'input_nc': 1,
        'output_nc': 1,
        'domains': ('source', 'target'),
        'init_channel_number': 16,
        'verbose': True,
        'weight_path': '/home/lf/data_fong/PROJECT/UMMS/traces/checkpoints/PhaseBaseLine/DualStream/mrus11211280_fold0_bs4_SingleBoth_ch16_combo_1_1_1.5_l2_1e-4_adam_1e-4_poly_3x400_0.6_baseline_1080Ti/latest_net_mrus11211280_fold0_bs4_SingleBoth_ch16_combo_1_1_1.5_l2_1e-4_adam_1e-4_poly_3x400_0.6_baseline_1080Ti.pth',
        'local_gpu': 0
    }

    opt = SimpleNamespace(**kwargs)
    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')
    network = define_dualstream(opt, device, domains=('source', 'target'))

    network = load_weithts(network, opt.weight_path, device, name='umms')
    network.eval()
    print_module_name(network)
    # [Network dualname] Total number of parameters : 4.119 M
    # # encoders
    # # encoders.0
    # # encoders.0.basic_module
    # # encoders.0.basic_module.conv1
    # # encoders.0.basic_module.conv1.conv
    # # encoders.0.basic_module.conv1.norm
    # # encoders.0.basic_module.conv1.act
    # # encoders.0.basic_module.conv2
    # # encoders.0.basic_module.conv2.conv
    # # encoders.0.basic_module.conv2.norm
    # # encoders.0.basic_module.conv2.act
    # # encoders.1
    # # encoders.1.pooling
    # # encoders.1.basic_module
    # # encoders.1.basic_module.conv1
    # # encoders.1.basic_module.conv1.conv
    # # encoders.1.basic_module.conv1.norm
    # # encoders.1.basic_module.conv1.act
    # # encoders.1.basic_module.conv2
    # # encoders.1.basic_module.conv2.conv
    # # encoders.1.basic_module.conv2.norm
    # # encoders.1.basic_module.conv2.act
    # # encoders.2
    # # encoders.2.pooling
    # # encoders.2.basic_module
    # # encoders.2.basic_module.conv1
    # # encoders.2.basic_module.conv1.conv
    # # encoders.2.basic_module.conv1.norm
    # # encoders.2.basic_module.conv1.act
    # # encoders.2.basic_module.conv2
    # # encoders.2.basic_module.conv2.conv
    # # encoders.2.basic_module.conv2.norm
    # # encoders.2.basic_module.conv2.act
    # # encoders.3
    # # encoders.3.pooling
    # # encoders.3.basic_module
    # # encoders.3.basic_module.conv1
    # # encoders.3.basic_module.conv1.conv
    # # encoders.3.basic_module.conv1.norm
    # # encoders.3.basic_module.conv1.act
    # # encoders.3.basic_module.conv2
    # # encoders.3.basic_module.conv2.conv
    # # encoders.3.basic_module.conv2.norm
    # # encoders.3.basic_module.conv2.act
    # # encoders.4
    # # encoders.4.pooling
    # # encoders.4.basic_module
    # # encoders.4.basic_module.conv1
    # # encoders.4.basic_module.conv1.conv
    # # encoders.4.basic_module.conv1.norm
    # # encoders.4.basic_module.conv1.act
    # # encoders.4.basic_module.conv2
    # # encoders.4.basic_module.conv2.conv
    # # encoders.4.basic_module.conv2.norm
    # # encoders.4.basic_module.conv2.act
    # # decoders
    # # decoders.0
    # # decoders.0.upsampling
    # # decoders.0.basic_module
    # # decoders.0.basic_module.conv1
    # # decoders.0.basic_module.conv1.conv
    # # decoders.0.basic_module.conv1.norm
    # # decoders.0.basic_module.conv1.act
    # # decoders.0.basic_module.conv2
    # # decoders.0.basic_module.conv2.conv
    # # decoders.0.basic_module.conv2.norm
    # # decoders.0.basic_module.conv2.act
    # # decoders.1
    # # decoders.1.upsampling
    # # decoders.1.basic_module
    # # decoders.1.basic_module.conv1
    # # decoders.1.basic_module.conv1.conv
    # # decoders.1.basic_module.conv1.norm
    # # decoders.1.basic_module.conv1.act
    # # decoders.1.basic_module.conv2
    # # decoders.1.basic_module.conv2.conv
    # # decoders.1.basic_module.conv2.norm
    # # decoders.1.basic_module.conv2.act
    # # decoders.2
    # # decoders.2.upsampling
    # # decoders.2.basic_module
    # # decoders.2.basic_module.conv1
    # # decoders.2.basic_module.conv1.conv
    # # decoders.2.basic_module.conv1.norm
    # # decoders.2.basic_module.conv1.act
    # # decoders.2.basic_module.conv2
    # # decoders.2.basic_module.conv2.conv
    # # decoders.2.basic_module.conv2.norm
    # # decoders.2.basic_module.conv2.act
    # # decoders.3
    # # decoders.3.upsampling
    # # decoders.3.basic_module
    # # decoders.3.basic_module.conv1
    # # decoders.3.basic_module.conv1.conv
    # # decoders.3.basic_module.conv1.norm
    # # decoders.3.basic_module.conv1.act
    # # decoders.3.basic_module.conv2
    # # decoders.3.basic_module.conv2.conv
    # # decoders.3.basic_module.conv2.norm
    # # decoders.3.basic_module.conv2.act
    # # outconv
    # # outconv.conv3d
    # # outconv.upsampling
    # # outconv.activation


def model_test():
    # single_model_test()
    dualstream_model_test()


def data_test():
    kwargs = {
        'dataset_name': 'mrusmr',
        'dataroot': '/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280',
        'phase': 'test',
        'fold': 0,
        'custom': True,
        'preprocess': 'centercrop',
        'crop_size': (112, 112, 80),
        'num_threads': 16,
        'local_gpu': 0
    }
    opt = SimpleNamespace(**kwargs)
    device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

    predict_dataset_class = create_predict_dataset(opt.dataset_name)
    test_dataset = predict_dataset_class(opt)
    print('test_dataset:{}'.format(len(test_dataset)))
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_threads,
                                                  pin_memory=False,
                                                  drop_last=False)
    print('test_dataloader:{}'.format(len(test_dataloader)))

    for patient_id, data in enumerate(test_dataloader):
        volume_name = os.path.basename(data['volume_path'][0]).split('.')[0]
        test_data = data['volume'].to(device)
        label = data['label'].to(device)
        print(volume_name, test_data.shape, label.shape)


def main():
    # model_test()
    # data_test()
    test_cam()


if __name__ == "__main__":
    main()

# in_conv
# in_conv.conv
# in_conv.conv.conv1
# in_conv.conv.conv1.conv
# in_conv.conv.conv1.norm
# in_conv.conv.conv1.norm.norm
# in_conv.conv.conv1.act
# in_conv.conv.conv2
# in_conv.conv.conv2.conv
# in_conv.conv.conv2.norm
# in_conv.conv.conv2.norm.norm
# in_conv.conv.conv2.act
# encoders
# encoders.0
# encoders.0.max_pool
# encoders.0.double_conv
# encoders.0.double_conv.conv1
# encoders.0.double_conv.conv1.conv
# encoders.0.double_conv.conv1.norm
# encoders.0.double_conv.conv1.norm.norm
# encoders.0.double_conv.conv1.act
# encoders.0.double_conv.conv2
# encoders.0.double_conv.conv2.conv
# encoders.0.double_conv.conv2.norm
# encoders.0.double_conv.conv2.norm.norm
# encoders.0.double_conv.conv2.act
# encoders.1
# encoders.1.max_pool
# encoders.1.double_conv
# encoders.1.double_conv.conv1
# encoders.1.double_conv.conv1.conv
# encoders.1.double_conv.conv1.norm
# encoders.1.double_conv.conv1.norm.norm
# encoders.1.double_conv.conv1.act
# encoders.1.double_conv.conv2
# encoders.1.double_conv.conv2.conv
# encoders.1.double_conv.conv2.norm
# encoders.1.double_conv.conv2.norm.norm
# encoders.1.double_conv.conv2.act
# encoders.2
# encoders.2.max_pool
# encoders.2.double_conv
# encoders.2.double_conv.conv1
# encoders.2.double_conv.conv1.conv
# encoders.2.double_conv.conv1.norm
# encoders.2.double_conv.conv1.norm.norm
# encoders.2.double_conv.conv1.act
# encoders.2.double_conv.conv2
# encoders.2.double_conv.conv2.conv
# encoders.2.double_conv.conv2.norm
# encoders.2.double_conv.conv2.norm.norm
# encoders.2.double_conv.conv2.act
# encoders.3
# encoders.3.max_pool
# encoders.3.double_conv
# encoders.3.double_conv.conv1
# encoders.3.double_conv.conv1.conv
# encoders.3.double_conv.conv1.norm
# encoders.3.double_conv.conv1.norm.norm
# encoders.3.double_conv.conv1.act
# encoders.3.double_conv.conv2
# encoders.3.double_conv.conv2.conv
# encoders.3.double_conv.conv2.norm
# encoders.3.double_conv.conv2.norm.norm
# encoders.3.double_conv.conv2.act
# decoders
# decoders.0
# decoders.0.upsample
# decoders.0.upsample.deconv
# decoders.0.upsample.norm
# decoders.0.upsample.norm.norm
# decoders.0.upsample.act
# decoders.0.double_conv
# decoders.0.double_conv.conv1
# decoders.0.double_conv.conv1.conv
# decoders.0.double_conv.conv1.norm
# decoders.0.double_conv.conv1.norm.norm
# decoders.0.double_conv.conv1.act
# decoders.0.double_conv.conv2
# decoders.0.double_conv.conv2.conv
# decoders.0.double_conv.conv2.norm
# decoders.0.double_conv.conv2.norm.norm
# decoders.0.double_conv.conv2.act
# decoders.1
# decoders.1.upsample
# decoders.1.upsample.deconv
# decoders.1.upsample.norm
# decoders.1.upsample.norm.norm
# decoders.1.upsample.act
# decoders.1.double_conv
# decoders.1.double_conv.conv1
# decoders.1.double_conv.conv1.conv
# decoders.1.double_conv.conv1.norm
# decoders.1.double_conv.conv1.norm.norm
# decoders.1.double_conv.conv1.act
# decoders.1.double_conv.conv2
# decoders.1.double_conv.conv2.conv
# decoders.1.double_conv.conv2.norm
# decoders.1.double_conv.conv2.norm.norm
# decoders.1.double_conv.conv2.act
# decoders.2
# decoders.2.upsample
# decoders.2.upsample.deconv
# decoders.2.upsample.norm
# decoders.2.upsample.norm.norm
# decoders.2.upsample.act
# decoders.2.double_conv
# decoders.2.double_conv.conv1
# decoders.2.double_conv.conv1.conv
# decoders.2.double_conv.conv1.norm
# decoders.2.double_conv.conv1.norm.norm
# decoders.2.double_conv.conv1.act
# decoders.2.double_conv.conv2
# decoders.2.double_conv.conv2.conv
# decoders.2.double_conv.conv2.norm
# decoders.2.double_conv.conv2.norm.norm
# decoders.2.double_conv.conv2.act
# decoders.3
# decoders.3.upsample
# decoders.3.upsample.deconv
# decoders.3.upsample.norm
# decoders.3.upsample.norm.norm
# decoders.3.upsample.act
# decoders.3.double_conv
# decoders.3.double_conv.conv1
# decoders.3.double_conv.conv1.conv
# decoders.3.double_conv.conv1.norm
# decoders.3.double_conv.conv1.norm.norm
# decoders.3.double_conv.conv1.act
# decoders.3.double_conv.conv2
# decoders.3.double_conv.conv2.conv
# decoders.3.double_conv.conv2.norm
# decoders.3.double_conv.conv2.norm.norm
# decoders.3.double_conv.conv2.act
# out_conv
# out_conv.conv3d
# out_conv.upsampling
