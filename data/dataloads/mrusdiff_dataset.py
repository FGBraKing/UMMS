import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from copy import copy
from configs.utils_config import get_pretty_opt
from data.utils_data import print_data_describe
from data.dataloads.base_dataset import BaseDataset, CustomDataset, NIIDataset
from data.transforms.transformOnArray import get_transform, get_pre_transform, get_post_transform, ToTensor
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list, Timer
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram


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
    # print(us_paths[0], mr_paths[0])
    # print(os.path.isfile(us_paths[0]['volume']),
    #       os.path.isfile(us_paths[0]['label']),
    #       os.path.isfile(mr_paths[0]['volume']),
    #       os.path.isfile(mr_paths[0]['label']))
    return mr_paths, us_paths


def correct_source_opt(opt):
    opt.crop_size = opt.source_crop_size
    return opt


def correct_target_opt(opt):
    opt.crop_size = opt.target_crop_size
    return opt


class MrusDiffDataset(NIIDataset):
    def __init__(self, opt):
        super(MrusDiffDataset, self).__init__(opt)
        self.mr_paths, self.us_paths = get_data_path(opt.dataroot, opt.phase)
        self.mr_size = len(self.mr_paths)
        self.us_size = len(self.us_paths)

        self.data_size = max(self.us_size, self.mr_size)

        self.pre_transform = get_pre_transform(opt)
        opt = correct_source_opt(opt)
        self.source_transform = get_transform(opt)
        opt = correct_target_opt(opt)
        self.target_transfoem = get_transform(opt)
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

        # 进行形状变换前的对volume进行的一些特殊处理,目前为空
        mr_volume = self._apply_pre_transform(mr_volume)
        us_volume = self._apply_pre_transform(us_volume)
        # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
        if self.source_transform:
            mr_volume, mr_label = self.source_transform(mr_volume, mr_label)
        if self.target_transfoem:
            us_volume, us_label = self.target_transfoem(us_volume, us_label)
        # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
        mr_volume = self._apply_post_transform(mr_volume)
        us_volume = self._apply_post_transform(us_volume)

        mr_volume = self.to_tensor(mr_volume)
        mr_label = self.to_tensor(mr_label)
        us_volume = self.to_tensor(us_volume)
        us_label = self.to_tensor(us_label)
        mr_spacing = torch.Tensor(mr_spacing[::-1])
        us_spacing = torch.Tensor(us_spacing[::-1])

        return {'mr_volume': mr_volume, 'mr_volume_path': mr_path['volume'],
                'mr_label': mr_label, 'mr_label_path': mr_path['label'], 'mr_spacing': mr_spacing,
                'us_volume': us_volume, 'us_volume_path': us_path['volume'],
                'us_label': us_label, 'us_label_path': us_path['label'], 'us_spacing': us_spacing}

    def print_data_describe(self, *args, **kwargs):
        for phase, case_list in zip(['mr', 'us'], [self.mr_paths, self.us_paths]):
            print("{:*^120s}".format(phase))
            print_data_describe(case_list)

    def custom_debug(self, *args, **kwargs):
        print(f'data_size:{self.data_size}')
        for index in range(self.data_size):
            if index < 10:
                tt = self.__getitem__(index)
                print(tt['mr_volume_path'])
                # print(tt['volume'].shape)
                # print(type(tt['volume']))
                data = tt['mr_volume'].cpu().numpy()
                label = tt['mr_label'].cpu().numpy()
                print(type(label), label.shape)
                print(type(data), data.shape)
                title = os.path.basename(tt['mr_volume_path']).split('.')[0]
                print_numpy(data, shp=True, percentile=True)
                # show_array_3d(data[0, ...], 4, 4)
                show_volume_label(data[0, ...], label[0, ...], row=4, col=4, title=title)
                # show_array_3d(label[0, ...], 4, 4)


def main():
    parser = argparse.ArgumentParser(description='for the TRUS dataset')
    parser.add_argument('--dataroot', type=str,
                        default=r'/home/lf/data_fong/CODE/PycharmProject/UMMS/traces/datasets/MR-USviaFenster20_pre')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--seed', type=int, default=1008)
    parser.add_argument('--preprocess', type=str, default=r'elastic_randomscale_randomcrop_ranomrotate_centercrop_'
                                                          r'rot90_mirror_gaussianNoise_GaussianBlur_'
                                                          r'BrightnessMultiplicative_contrast_simulate_gammatransform')

    parser.add_argument('--serial_batches', action='store_true')
    parser.add_argument('--custom', action='store_true')
    parser.add_argument('--rot_axes', type=list, default=[1, 2], help='the rot90 axes')
    parser.add_argument('--mirror_axes', type=list, default=[1, 2], help='the rot90 axes')
    parser.add_argument('--rot_angle_spectrum', type=int, default=25)
    parser.add_argument('--scale_range', type=list, default=[0.85, 1.25])
    parser.add_argument('--order_data', type=int, default=3)
    parser.add_argument('--order_seg', type=int, default=1)

    parser.add_argument('--elastic_alpha', type=list, default=[0., 900])
    parser.add_argument('--elastic_sigma', type=list, default=[9., 13.])
    parser.add_argument('--g_noise_variance', type=list, default=[0.3, 0.7])

    parser.add_argument('--crop_size', type=list, default=[160, 160, 16])
    opt = parser.parse_args(args=['--serial_batches', '--custom'])
    opt.random_state = np.random.RandomState(seed=opt.seed)

    print(get_pretty_opt(opt))
    dataset = MrusDiffDataset(opt)
    print(len(dataset))
    with Timer('running with custom_debug, using time:%ss'):
        dataset.custom_debug()
        # start_time = time.time()
        # for ind, test_data in enumerate(dataset):
        #     print('using time:%s' % (time.time()-start_time))
        #     if ind > 100:
        #         break
        #     print(ind)
        #     print(test_data.keys())
        #     start_time = time.time()


if __name__ == '__main__':
    main()

