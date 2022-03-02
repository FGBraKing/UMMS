import os
import argparse
import numpy as np
from data.utils_data import nii_loader
from data.dataloads.base_dataset import BaseDataset, CustomDataset, NIIDataset
from data.transforms.transformOnArray import get_transform, get_pre_transform, get_post_transform, ToTensor
from configs.utils_config import get_pretty_opt
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram
# from batchgenerators.augmentations.crop_and_pad_augmentations import pad_nd_image_and_seg, crop
from utils.others.utils import Timer
import time


def get_data_path(dataroot, data_phase):
    root = os.path.join(dataroot, data_phase)
    return [{'volume': os.path.join(root, name.replace('label', 'image')), 'label': os.path.join(root, name)}
            for name in os.listdir(root) if 'label' in name]


class MrusmrDataset(NIIDataset):
    def __init__(self, opt):
        super(MrusmrDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase)
        self.data_size = len(self.paths)

        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)
        self.to_tensor = ToTensor(expand_dims=True)

    def __getitem__(self, index):

        index_used = self._get_used_index(index)

        volume_path = self.paths[index_used]['volume']
        label_path = self.paths[index_used]['label']
        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)
        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        volume = self._apply_pre_transform(volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        #
        volume, label = self._apply_transform(volume, label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        volume = self._apply_post_transform(volume)

        # volume, label = crop(volume, label, self.opt.crop_size[::-1], crop_type='center')

        volume = self.to_tensor(volume)
        label = self.to_tensor(label)

        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}

    def custom_debug(self, *args, **kwargs):
        print(f'data_size:{self.data_size}')
        for index in range(self.data_size):
            if index < 10:
                tt = self.__getitem__(index)
                print(tt['volume_path'])
                # print(tt['volume'].shape)
                # print(type(tt['volume']))
                data = tt['volume'].cpu().numpy()
                label = tt['label'].cpu().numpy()
                print(type(label), label.shape)
                print(type(data), data.shape)
                title = os.path.basename(tt['volume_path']).split('.')[0]
                print_numpy(data, shp=True, percentile=True)
                # show_array_3d(data[0, ...], 4, 4)
                show_volume_label(data[0, ...], label[0, ...], row=4, col=4, title=title)
                # show_array_3d(label[0, ...], 4, 4)


class TestMrusmrDataset(BaseDataset):
    def __init__(self, opt, loader=nii_loader):
        super(TestMrusmrDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase)
        self.data_size = len(self.paths)
        self.loader = loader

    def __getitem__(self, index):
        volume_path = self.paths[index]['volume']
        label_path = self.paths[index]['label']
        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)
        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}

    def __len__(self):
        return self.data_size


class PredictTestMrusmrDataset(BaseDataset):
    def __init__(self, opt, loader=nii_loader):
        # save the option and dataset root
        super(PredictTestMrusmrDataset, self).__init__(opt)

        self.paths = get_data_path(opt.dataroot, opt.phase)  # should be [{'volume':volume,'label':label}, ...]
        self.data_size = len(self.paths)

        self.loader = loader
        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)

        self.to_tensor = ToTensor(expand_dims=True)

    def __getitem__(self, index):
        volume_path = self.paths[index]['volume']
        label_path = self.paths[index]['label']
        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)
        origin_shape = label.shape

        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        if self.pre_transform:
            volume = self.pre_transform(volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        if self.transform:
            volume, label = self.transform(volume, label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        if self.post_transform:
            volume = self.post_transform(volume)

        now_shape = label.shape

        volume = self.to_tensor(volume)
        label = self.to_tensor(label)

        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path,
                'origin_shape': origin_shape, 'now_shape': now_shape}

    def __len__(self):
        """Return the total number of images."""
        return self.data_size


def main():
    parser = argparse.ArgumentParser(description='for the test of promise dataset')
    parser.add_argument('--dataroot', type=str,
                        default=r'/home/lf/raid_lf/PROJECT/DLForPytorch/traces/datasets/MR-USviaFenster20_pre')
    parser.add_argument('--phase', type=str, default='mrtrain')
    parser.add_argument('--seed', type=int, default=1008)
    parser.add_argument('--preprocess', type=str, default=r'elastic_randomscale_randomcrop_ranomrotate_centercrop_'
                                                          r'rot90_mirror_gaussianNoise_GaussianBlur_'
                                                          r'BrightnessMultiplicative_contrast_simulate_gammatransform')
    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--custom', action='store_true')
    parser.add_argument('--rot_axes', type=list, default=[2, 1], help='the rot90 axes')
    parser.add_argument('--order_data', type=int, default=3)
    parser.add_argument('--order_seg', type=int, default=1)
    parser.add_argument('--elastic_alpha', type=list, default=[0., 900])
    parser.add_argument('--elastic_sigma', type=list, default=[9., 13.])
    parser.add_argument('--scale_range', type=list, default=[0.85, 1.25])
    parser.add_argument('--crop_size', type=list, default=[96, 96, 32])
    opt = parser.parse_args(args=['--serial_batches', '--custom'])
    # opt.preprocess = r'elastic_randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror_gaussianNoise_' \
    #                  r'GaussianBlur_BrightnessMultiplicative_contrast_simulate_gammatransform'
    opt.random_state = np.random.RandomState(seed=opt.seed)

    # opt.preprocess = r'elastic_randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror_gaussianNoise_GaussianBlur_BrightnessMultiplicative_contrast_simulate_gammatransform'
    #
    print(get_pretty_opt(opt))
    dataset = MrusmrDataset(opt)
    print(len(dataset))
    with Timer('running with custom_debug, using time:%ss'):
        # dataset.custom_debug()
        start_time = time.time()
        for ind, test_data in enumerate(dataset):
            print('using time:%s'%(time.time()-start_time))
            if ind > 100:
                break
            print(ind)
            print(test_data.keys())
            start_time = time.time()


if __name__ == '__main__':
    main()

