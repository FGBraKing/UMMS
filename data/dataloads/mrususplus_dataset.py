import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from itertools import combinations
from configs.utils_config import get_pretty_opt
from data.utils_data import nii_loader
from data.dataloads.base_dataset import BaseDataset, CustomDataset, NIIDataset
from data.transforms.transformOnArray import get_transform, get_pre_transform, get_post_transform, ToTensor
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list, Timer
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram


def get_data_path(dataroot, data_phase, fold=0, k_fold=5, random_seed=1008):

    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))

    if not isinstance(fold, int):
        return [
            {
                'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume')),
                'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'roi'))
            }
            for p_id in pat_ids
        ]

    if os.path.isfile(os.path.join(dataroot, f'split_{fold}.csv')):
        split_df = pd.read_csv(os.path.join(dataroot, f'split_{fold}.csv'), keep_default_na=True)
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

    test_ids = split_df['test'].dropna().tolist()
    train_ids = split_df['train'].dropna().tolist()

    used_ids = test_ids if data_phase == "test" else train_ids

    data_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'roi'))
        }
        for p_id in used_ids
    ]
    return data_paths


class MrususPlusDataset(NIIDataset):
    axis_database = (
        (0,),
        ((0,), (1,), (1, 2)),
        ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)),
    )

    def __init__(self, opt):
        super(MrususPlusDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase, opt.fold)

        self.mirror_axes = self.get_mirror_axis(opt.mirror_axes)
        self.mirror_num = len(self.mirror_axes) + 1                 # self.mirror_num = 8
        self.rotate_axes = self.get_rot_axis(opt.rot_axes)
        self.rotate_num = len(self.rotate_axes) * 4                 # self.rotate_num = 4

        self.true_size = len(self.paths)
        # 顺序是path、mirror、rotate
        self.data_size = self.true_size*self.mirror_num*self.rotate_num

        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)
        self.to_tensor = ToTensor(expand_dims=True)

    @staticmethod
    def get_mirror_axis(axes):
        length = len(axes)
        all_result = []
        for i in range(length):
            all_result += list(combinations(axes, i+1))
        return tuple(all_result)

    @staticmethod
    def get_rot_axis(axes):
        return tuple(combinations(axes, 2))

    def get_rot_data(self, data, index):
        index = index % (self.mirror_num*self.rotate_num)
        index = index % self.rotate_num
        axis_num = index // 4
        index = index % 4
        rot_num = index // 1
        data = np.rot90(data, rot_num, axes=self.rotate_axes[axis_num])
        return data

    def get_mirror_data(self, data, index):
        index = index % (self.mirror_num*self.rotate_num)
        index = index // self.rotate_num
        if index == 0:
            return data
        else:
            return np.flip(data, self.mirror_axes[index-1])

    def get_data_index(self, index):
        return index // (self.mirror_num*self.rotate_num)

    def get_augmentation(self, data, seg, index):
        data = self.get_mirror_data(data, index)
        data = self.get_rot_data(data, index)

        seg = self.get_mirror_data(seg, index)
        seg = self.get_rot_data(seg, index)
        return data, seg

    def __getitem__(self, index):
        index_used = self._get_used_index(index)

        data_index = self.get_data_index(index_used)

        volume_path = self.paths[data_index]['volume']
        label_path = self.paths[data_index]['label']

        spacing = sitk.ReadImage(volume_path).GetSpacing()

        volume = self.loader(volume_path)  # DHW, zyx
        label = self.loader(label_path)

        volume, label = self.get_augmentation(volume, label, index_used)

        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        volume = self._apply_pre_transform(volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        volume, label = self._apply_transform(volume, label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        volume = self._apply_post_transform(volume)

        # volume, label = crop(volume, label, self.opt.crop_size[::-1], crop_type='center')

        volume = self.to_tensor(volume)  # NCDHW
        label = self.to_tensor(label)
        spacing = torch.Tensor(spacing[::-1])

        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path,
                'spacing': spacing}


class TestMrususPlusDataset(BaseDataset):
    def __init__(self, opt, loader=nii_loader):
        super(TestMrususPlusDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase, opt.fold)
        self.data_size = len(self.paths)
        self.loader = loader

    def __getitem__(self, index):
        volume_path = self.paths[index]['volume']
        label_path = self.paths[index]['label']
        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)
        spacing = sitk.ReadImage(volume_path).GetSpacing()
        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path,
                'spacing': tuple(spacing[::-1])}

    def __len__(self):
        return self.data_size


class PredictMrususPlusDataset(BaseDataset):
    def __init__(self, opt, loader=nii_loader):
        super(PredictMrususPlusDataset, self).__init__(opt)

        self.paths = get_data_path(opt.dataroot, opt.phase, opt.fold)  # should be [{'volume':volume,'label':label},...]
        self.data_size = len(self.paths)

        self.loader = loader
        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)

        self.to_tensor = ToTensor(expand_dims=True)

    def __getitem__(self, index):
        volume_path = self.paths[index]['volume']
        label_path = self.paths[index]['label']

        spacing = sitk.ReadImage(volume_path).GetSpacing()

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
        spacing = torch.Tensor(spacing[::-1])

        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path,
                'origin_shape': origin_shape, 'now_shape': now_shape, 'spacing': spacing}

    def __len__(self):
        """Return the total number of images."""
        return self.data_size


def main():
    from types import SimpleNamespace
    kwargs = {
        "dataroot": r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280',
        "phase": "train",
        "seed": 1008,
        "fold": 2,
        "preprocess": r'elastic_randomscale_randomcropwithstride_randomrotate_centercrop_gaussianNoise_GaussianBlur_'
                      r'BrightnessMultiplicative_contrast_simulate_gammatransform',
        "serial_batches": True,
        "custom": True,
        "mirror_axes": [0,1,2],
        "rot_axes": [2, 1],
        "rot_angle_spectrum": 30,
        "order_data": 3,
        "order_seg": 1,
        "elastic_alpha": [0., 70],
        "elastic_sigma": [8., 12.],
        "scale_range": [0.7, 1.3],
        "crop_size": [80, 80, 64],
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
    dataset = MrususPlusDataset(opt)
    print(len(dataset))
    with Timer('running with custom_debug, using time:%ss'):
        # dataset.custom_debug()
        start_time = time.time()
        for ind, test_data in enumerate(dataset):
            print('using time:%s' % (time.time()-start_time))
            if ind > 100:
                break
            print(ind)
            print(test_data.keys())
            start_time = time.time()


def test_reading_order(dataroot=r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280', folds=5):
    print(os.getcwd())
    for phase in ['test', 'train']:
        for fold in range(folds):
            print('{:*^12}'.format(f'{phase}:{fold}'))
            paths = get_data_path(dataroot, phase, fold)
            for path in paths:
                print('{:-^12}'.format(os.path.basename(path['volume']).split('.')[0][:4]))


if __name__ == '__main__':
    main()
    # test_reading_order()

