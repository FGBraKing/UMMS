import os
import re
import time
import torch
import argparse
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from itertools import combinations
from configs.utils_config import get_pretty_opt
from data.utils_data import print_data_describe
from data.dataloads.base_dataset import BaseDataset, CustomDataset, NIIDataset
from data.transforms.transformOnArray import get_transform, get_pre_transform, get_post_transform, ToTensor
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list, Timer
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram
from utils.others.metrics import BinaryMetrics
from data.dataloads.base_dataset import AugmentationIndex


def get_data_path(dataroot, data_phase, fold=1, k_fold=5, random_seed=1008):
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))

    if not isinstance(fold, int):
        used_ids = pat_ids
    elif os.path.isfile(os.path.join(dataroot, f'split_{fold}.csv')):
        split_df = pd.read_csv(os.path.join(dataroot, f'split_{fold}.csv'), keep_default_na=True)
        test_ids = split_df['test'].dropna().tolist()
        train_ids = split_df['train'].dropna().tolist()
        used_ids = test_ids if data_phase == "test" else train_ids
    else:
        print('creating the split files!')
        np.random.RandomState(seed=random_seed).shuffle(pat_ids)

        fold_number = (len(pat_ids)+k_fold-1)//k_fold  # 每折的数量，向上取整比如16=44440

        # split_save
        for i in range(k_fold):
            k_fold_dict = {
                'test': pat_ids[i * fold_number:(i + 1) * fold_number],
                'train': pat_ids[0:i * fold_number] + pat_ids[(i + 1) * fold_number:]
            }
            print(k_fold_dict)
            k_fold_df = pd.DataFrame.from_dict(k_fold_dict, orient='index')
            k_fold_df.T.to_csv(os.path.join(dataroot, f'split_{i}.csv'), index=False)

        # all save
        kfold_dict = dict.fromkeys(range(k_fold))
        for i in range(k_fold):
            kfold_dict[i] = {
                'test': pat_ids[i * fold_number:(i + 1) * fold_number],
                'train': pat_ids[0:i * fold_number] + pat_ids[(i + 1) * fold_number:]
            }
        split_df = pd.DataFrame.from_dict(kfold_dict)
        split_df.T.to_csv(os.path.join(dataroot, 'split.csv'))

        # print('Please rerun the program!')
        raise FileExistsError(f'split_{fold}.csv has been created, please rerun the program')

    us_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'roi'))
        }
        for p_id in used_ids
    ]

    mr_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'roi'))
        }
        for p_id in used_ids
    ]
    return mr_paths, us_paths


# 先选样本，再选扩增方式。batch内样本可重叠
class MrusPlusPlusDataset(NIIDataset):
    def __init__(self, opt):
        super(MrusPlusPlusDataset, self).__init__(opt)
        self.mr_paths, self.us_paths = get_data_path(opt.dataroot, opt.phase, opt.fold)
        self.mr_size = len(self.mr_paths)
        self.us_size = len(self.us_paths)

        self.mirror_axes = self.get_mirror_axis(opt.mirror_axes)
        self.mirror_num = 2**len(opt.mirror_axes)
        self.rotate_axes = self.get_rot_axis(opt.rot_axes)
        self.rotate_num = len(self.rotate_axes) * 4

        # self.mr_index_generator = AugmentationIndex(opt.mirror_axes, opt.rot_axes, True, replacement=False, seed=24)
        self.us_index_generator = AugmentationIndex(opt.mirror_axes, opt.rot_axes,
                                                    shuffle=True, replacement=False, seed=28)
        # self.mr_aug_index = [list(self.mr_index_generator) for i in range(self.mr_size)]
        self.us_aug_index = [list(self.us_index_generator) for i in range(self.us_size)]

        self.true_size = min(self.us_size, self.mr_size)

        self.data_size = self.true_size * self.mirror_num * self.rotate_num

        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)
        self.to_tensor = ToTensor(expand_dims=True)
        self.copy_aug = False

    def _get_used_index(self, index):
        if self.opt.serial_batches:  # make sure index is within then range
            index_used = index % self.data_size
        else:
            index_used = self.opt.random_state.randint(0, self.data_size - 1)
        return index_used

    def __getitem__(self, index):

        index_used = self._get_used_index(index)

        data_index = index_used // (self.mirror_num * self.rotate_num)

        mr_path = self.mr_paths[data_index]
        us_path = self.us_paths[data_index]

        mr_volume = self.loader(mr_path['volume'])      # 'label'
        us_volume = self.loader(us_path['volume'])
        mr_label = self.loader(mr_path['label'])
        us_label = self.loader(us_path['label'])
        mr_spacing = sitk.ReadImage(mr_path['volume']).GetSpacing()
        us_spacing = sitk.ReadImage(us_path['volume']).GetSpacing()
        mr_origin_shape = mr_label.shape
        us_origin_shape = us_label.shape

        mr_aug_index = index_used % (self.mirror_num * self.rotate_num)
        if self.copy_aug:
            us_aug_index = mr_aug_index
        else:
            us_aug_index = self.get_aug_index(data_index)
        mr_volume, mr_label = self.apply_augmentation_args(mr_volume, mr_label, *self.parse_index(mr_aug_index))
        us_volume, us_label = self.apply_augmentation_args(us_volume, us_label, *self.parse_index(us_aug_index))
        # print(index, index_used, mr_aug_index, us_aug_index, *self.parse_index(mr_aug_index))
        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        mr_volume = self._apply_pre_transform(mr_volume)
        us_volume = self._apply_pre_transform(us_volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        mr_volume, mr_label = self._apply_transform(mr_volume, mr_label)
        us_volume, us_label = self._apply_transform(us_volume, us_label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        mr_volume = self._apply_post_transform(mr_volume)
        us_volume = self._apply_post_transform(us_volume)

        mr_volume = self.to_tensor(mr_volume)
        mr_label = self.to_tensor(mr_label)
        us_volume = self.to_tensor(us_volume)
        us_label = self.to_tensor(us_label)
        mr_spacing = torch.Tensor(mr_spacing[::-1])
        us_spacing = torch.Tensor(us_spacing[::-1])
        mr_now_shape = mr_label.shape
        us_now_shape = us_label.shape

        return {
            'mr_volume': mr_volume, 'mr_volume_path': mr_path['volume'],
            'mr_label': mr_label, 'mr_label_path': mr_path['label'], 'mr_spacing': mr_spacing,
            'mr_origin_shape': mr_origin_shape, 'mr_now_shape': mr_now_shape,
            'us_volume': us_volume, 'us_volume_path': us_path['volume'],
            'us_label': us_label, 'us_label_path': us_path['label'], 'us_spacing': us_spacing,
            'us_origin_shape': us_origin_shape, 'us_now_shape': us_now_shape
        }

    @staticmethod
    def get_mirror_axis(axes):
        length = len(axes)
        all_result = []
        for i in range(length):
            all_result += list(combinations(axes, i + 1))
        return tuple(all_result)

    @staticmethod
    def get_rot_axis(axes):
        return tuple(combinations(axes, 2))

    def get_aug_index(self, index):
        if isinstance(self.us_aug_index[index], list) and len(self.us_aug_index[index]) > 0:
            us_aug_index = self.us_aug_index[index].pop()
        else:
            self.us_aug_index[index] = list(self.us_index_generator)
            us_aug_index = self.us_aug_index[index].pop()
        return us_aug_index

    def parse_index(self, index):
        mirror_index = index // self.rotate_num
        index = index % self.rotate_num
        rot_axis_index = index // 4
        index = index % 4
        rot_angle_index = index // 1
        return mirror_index, rot_axis_index, rot_angle_index

    def apply_augmentation_args(self, data, seg, mirror_aixs, rot_axis, rot_times):
        data = np.flip(data, self.mirror_axes[mirror_aixs-1]) if mirror_aixs > 0 else data
        data = np.rot90(data, rot_times, axes=self.rotate_axes[rot_axis])

        seg = np.flip(seg, self.mirror_axes[mirror_aixs - 1]) if mirror_aixs > 0 else seg
        seg = np.rot90(seg, rot_times, axes=self.rotate_axes[rot_axis])
        return data, seg

    def custom_debug(self, *args, **kwargs):
        print(f'data_size:{self.data_size}')
        for index in range(self.data_size):
            if index < 10:
                pass


class PredictMrusPlusPlusDataset(NIIDataset):
    def __init__(self, opt):
        super(PredictMrusPlusPlusDataset, self).__init__(opt)
        self.mr_paths, self.us_paths = get_data_path(opt.dataroot, opt.phase, opt.fold)
        self.mr_size = len(self.mr_paths)
        self.us_size = len(self.us_paths)

        self.data_size = max(self.us_size, self.mr_size)

        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)
        self.to_tensor = ToTensor(expand_dims=True)

    def __getitem__(self, index):

        index_used = self._get_used_index(index)

        mr_path = self.mr_paths[index_used]
        us_path = self.us_paths[index_used]

        mr_volume = self.loader(mr_path['volume'])      # 'label'
        us_volume = self.loader(us_path['volume'])
        mr_label = self.loader(mr_path['label'])
        us_label = self.loader(us_path['label'])
        mr_spacing = sitk.ReadImage(mr_path['volume']).GetSpacing()
        us_spacing = sitk.ReadImage(us_path['volume']).GetSpacing()
        mr_origin_shape = mr_label.shape
        us_origin_shape = us_label.shape

        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        mr_volume = self._apply_pre_transform(mr_volume)
        us_volume = self._apply_pre_transform(us_volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        mr_volume, mr_label = self._apply_transform(mr_volume, mr_label)
        us_volume, us_label = self._apply_transform(us_volume, us_label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        mr_volume = self._apply_post_transform(mr_volume)
        us_volume = self._apply_post_transform(us_volume)

        mr_volume = self.to_tensor(mr_volume)
        mr_label = self.to_tensor(mr_label)
        us_volume = self.to_tensor(us_volume)
        us_label = self.to_tensor(us_label)
        mr_spacing = torch.Tensor(mr_spacing[::-1])
        us_spacing = torch.Tensor(us_spacing[::-1])
        mr_now_shape = mr_label.shape
        us_now_shape = us_label.shape

        return {
            'mr_volume': mr_volume, 'mr_volume_path': mr_path['volume'],
            'mr_label': mr_label, 'mr_label_path': mr_path['label'], 'mr_spacing': mr_spacing,
            'mr_origin_shape': mr_origin_shape, 'mr_now_shape': mr_now_shape,
            'us_volume': us_volume, 'us_volume_path': us_path['volume'],
            'us_label': us_label, 'us_label_path': us_path['label'], 'us_spacing': us_spacing,
            'us_origin_shape': us_origin_shape, 'us_now_shape': us_now_shape
        }


def main():
    from types import SimpleNamespace
    kwargs = {
        "dataroot": r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280',
        "phase": "train",
        "seed": 1008,
        "fold": 2,
        "preprocess": r'elastic_randomscale_centercrop_gaussianNoise_GaussianBlur_'
                      r'BrightnessMultiplicative_contrast_simulate_gammatransform',
        "serial_batches": True,
        "custom": True,
        "mirror_axes": [0, 1, 2],
        "rot_axes": [2, 1],
        "order_data": 3,
        "order_seg": 1,
        "elastic_alpha": [0., 70],
        "elastic_sigma": [8., 12.],
        "scale_range": [0.7, 1.3],
        "crop_size": [112, 112, 80],
        "crop_stride": 2,
        "g_noise_variance": [0.0, 0.1],
        "bright_multiplier_range": [0.7, 1.3],
        "contrast_range": [0.65, 1.35],
        "simulate_zoom_range": [0.5, 1.0],
        "gamma_range": (0.7, 1.3),
    }

    opt = SimpleNamespace(**kwargs)
    opt.random_state = np.random.RandomState(seed=opt.seed)
    print(get_pretty_opt(opt))
    dataset = MrusPlusPlusDataset(opt)
    print(len(dataset))
    with Timer('running with custom_debug, using time:%ss'):
        # dataset.custom_debug()
        start_time = time.time()
        for ind, test_data in enumerate(dataset):
            print('using time:%s' % (time.time() - start_time))
            if ind > 100:
                break
            print(ind)
            print(test_data.keys())
            start_time = time.time()


if __name__ == "__main__":
    main()
