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
                'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'volume')),
                'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'roi'))
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

    mr_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'roi'))
        }
        for p_id in used_ids
    ]
    return mr_paths


class MrusmrPlusDataset(NIIDataset):
    axis_database = (
        (0,),
        ((0,), (1,), (1, 2)),
        ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)),
    )

    def __init__(self, opt):
        super(MrusmrPlusDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase, opt.fold)

        self.mirror_axes = self.get_mirror_axis(opt.mirror_axes)
        self.mirror_num = len(self.mirror_axes) + 1             # self.mirror_num = 8
        self.rotate_axes = self.get_rot_axis(opt.rot_axes)
        self.rotate_num = len(self.rotate_axes) * 4             # self.rotate_num = 4

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
        # ndim = len(data.shape)
        if index == 0:
            return data
        else:
            # return np.flip(data, MrusmrPlusDataset.axis_database[ndim-1][index])
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

        return {'volume': volume, 'label': label,
                'volume_path': volume_path, 'label_path': label_path, 'spacing': spacing}


class MrusmrDataset(NIIDataset):
    def __init__(self, opt):
        super(MrusmrDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase, opt.fold)
        self.data_size = len(self.paths)

        self.pre_transform = get_pre_transform(opt)
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform(opt)
        self.to_tensor = ToTensor(expand_dims=True)

    def __getitem__(self, index):

        index_used = self._get_used_index(index)

        volume_path = self.paths[index_used]['volume']
        label_path = self.paths[index_used]['label']

        spacing = sitk.ReadImage(volume_path).GetSpacing()

        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)
        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        volume = self._apply_pre_transform(volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        volume, label = self._apply_transform(volume, label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        volume = self._apply_post_transform(volume)

        # volume, label = crop(volume, label, self.opt.crop_size[::-1], crop_type='center')

        volume = self.to_tensor(volume)     # NCDHW
        label = self.to_tensor(label)
        spacing = torch.Tensor(spacing[::-1])

        return {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path,
                'spacing': spacing}

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


class PredictMrusmrDataset(BaseDataset):
    def __init__(self, opt, loader=nii_loader):
        # save the option and dataset root
        super(PredictMrusmrDataset, self).__init__(opt)

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


TestMrusmrPlusDataset = TestMrusmrDataset

PredictMrusmrPlusDataset = PredictMrusmrDataset


def main():
    parser = argparse.ArgumentParser(description='for the test of promise dataset')
    parser.add_argument('--dataroot', type=str,
                        default=r'/home/lf/raid_lf/PROJECT/UMMS/traces/datasets/MR-USviaFenster20-pre128')
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


def test_reading_order(dataroot=r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USviaFenster20-pre12812896', folds=5):
    print(os.getcwd())
    for phase in ['test', 'train']:
        for fold in range(folds):
            print('{:*^12}'.format(f'{phase}:{fold}'))
            paths = get_data_path(dataroot, phase, fold)
            for path in paths:
                print('{:-^12}'.format(os.path.basename(path['volume']).split('.')[0][:4]))


if __name__ == '__main__':
    # main()
    test_reading_order()

