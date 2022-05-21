import os
import torch
import argparse
import numpy as np
import pandas as pd
from data.utils_data import nii_loader
from data.dataloads.base_dataset import BaseDataset, CustomDataset, NIIDataset
from data.transforms.transformOnArray import get_transform, get_pre_transform, get_post_transform, ToTensor
from configs.utils_config import get_pretty_opt
from utils.others.utils import print_numpy, clip_array, slim_array, convert_str_to_list
from utils.others.img_io import show_array_3d, show_volume_label, show_array_histogram, show_pired_histogram
# from batchgenerators.augmentations.crop_and_pad_augmentations import pad_nd_image_and_seg, crop
from utils.others.utils import Timer
import time
from data.dataloads.base_dataset import TestOnePatientDataset
from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from utils.others.img_io import show_image, show_array_3d
import SimpleITK as sitk


def get_data_path(dataroot, data_phase, fold=0, k_fold=5, random_seed=1008):
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))
    split_df = pd.read_csv(os.path.join(dataroot, f'split_{fold}.csv'), keep_default_na=True)

    test_ids = split_df['test'].dropna().tolist()
    train_ids = split_df['train'].dropna().tolist()

    used_ids = test_ids if data_phase == "test" else train_ids

    us_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'roi'))
        }
        for p_id in used_ids
    ]
    return us_paths


class TestTrusDataset(BaseDataset):
    def __init__(self, opt, loader=nii_loader):
        super(TestTrusDataset, self).__init__(opt)
        self.paths = get_data_path(opt.dataroot, opt.phase, opt.fold)
        self.data_size = len(self.paths)
        self.loader = loader

    def __getitem__(self, index):
        volume_path = self.paths[index]['volume']
        label_path = self.paths[index]['label']
        volume = self.loader(volume_path)   # DHW, zyx
        label = self.loader(label_path)
        spacing = sitk.ReadImage(volume_path).GetSpacing()
        return {'volume': volume, 'label': label,
                'volume_path': volume_path, 'label_path': label_path, 'spacing': tuple(spacing[::-1])}

    def __len__(self):
        return self.data_size


def test_val_dataset():
    opt = CN(new_allowed=True)
    opt.dataroot = './traces/datasets/prostate_daf3d_pre'
    opt.phase = 'test'
    opt.crop_size = 96
    opt.stride = 96
    opt.no_augment = False

    test_dataset = TestTrusDataset(opt)
    #  {'volume': volume, 'label': label, 'volume_path': volume_path, 'label_path': label_path}
    print('test_dataset:{}'.format(len(test_dataset)))
    for data in test_dataset:
        print(data['volume'].shape)     # (175, 224, 224)
        show_image(data['volume'][:, :, 100], title='origin image')

        one_patient_dataset = TestOnePatientDataset(data['volume'][:, :, 100], opt)
        print('one_patient_dataset:{}'.format(len(one_patient_dataset)))

        dataset_info = one_patient_dataset.get_info()   # 'crop_size' 'stride' 'origin_shape'  'pad_shape'
        dataset_volumes = one_patient_dataset.get_volume()   # 'origin_volume'  'pad_volume'
        row, column = one_patient_dataset.get_crop_num_list()

        test_dataloader = DataLoader(one_patient_dataset,
                                     batch_size=len(one_patient_dataset),
                                     shuffle=False,
                                     num_workers=8,
                                     drop_last=False)
        print('test_dataloader:{}'.format(len(test_dataloader)))
        for test_data in test_dataloader:
            print(test_data.shape)      # N C ...
            data_to_show = test_data[:, 0, ...].numpy()
            show_array_3d(data_to_show, row, column, title='crop_image')
            # 还原的时候，axis的顺序是由大到小，2D先1后0，3D是210。也就是从循环的最深层开始，逐层还原
            # concat_array = [np.concatenate(data_to_show[i*column:i*column+column], axis=1) for i in range(row)]
            # show_image(concat_array[1], title='partly concat image')
            # concat_array = np.concatenate(concat_array, 0)
            # show_image(concat_array, title='concat image')
            for kk in range(test_data.shape[1]):
                data_to_show = test_data[kk].numpy()
                show_array_3d(data_to_show, 2, 2, title='crop_image')
                break

            pass
        break