# -*- coding:utf-8 -*-
import os
import re
import random
import pandas as pd
import numpy as np
import SimpleITK as sitk
from data.utils_data import save_nii, nii_loader, npy_loader, h5_loader
# from skimage.transform import resize
# from scipy.ndimage.interpolation import zoom
# from scipy.ndimage.interpolation import map_coordinates
# from data.transforms.transforms import resize_image_itk, Compose
from data.pre_process.dataset_pre import DatasetPre
from data.transforms.transformOnArray import standardize
from data.utils_nnunet import generate_dataset_json
from data.preprocessing import resample_data_or_seg, get_lowres_axis, get_do_separate_z
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p
from utils.others.utils import get_foreground_shape, cut_off_outliers, print_numpy, clip_array, slim_array
from batchgenerators.augmentations.crop_and_pad_augmentations import crop


class DatasetTool:

    @staticmethod
    def _read_img(path):
        filename, filetype = os.path.splitext(path)
        if filetype.lower() == '.nii' or filetype.lower() == '.mhd':
            itk_img = sitk.ReadImage(path)
            img_array = sitk.GetArrayFromImage(itk_img)     # indexes are z,y,x    DHW
            origin = itk_img.GetOrigin()
            direction = itk_img.GetDirection()
            spacing = itk_img.GetSpacing()
        elif filetype.lower() == '.npy':
            img_array = npy_loader(path)
            origin, direction, spacing = None, None, None
        elif filetype.lower() == '.h5':
            img_array = h5_loader(path, 'image', 'label')[-1]
            origin, direction, spacing = None, None, None
        else:
            raise TypeError('Filetype(%s) is unsupported' % filetype)
        return img_array, (origin, direction, spacing)

    def __init__(self, dataroot, mode='all', **kwargs):
        self.dataroot = dataroot
        self.mode = mode
        self.kwargs = kwargs
        assert os.path.isdir(self.dataroot)
        mr_case = []
        us_case = []
        for root, dirs, files in os.walk(self.dataroot):
            for name in files:
                if name.endswith('.nii') and 'MR' in name and 'image' in name:
                    # print(name)
                    mr_case.append(os.path.join(root, name))
                if name.endswith('.nii') and 'US' in name and 'image' in name:
                    # print(name)
                    us_case.append(os.path.join(root, name))

        pat_mr = re.compile(r'_MR_image')
        self.case_mr = [{'image': path,
                         'label': pat_mr.sub('_MR_label', path)} for path in mr_case]
        pat_us = re.compile(r'_US_image')
        self.case_us = [{'image': path,
                         'label': pat_us.sub('_US_label', path)} for path in us_case]

    def get_patient_list(self):
        if self.mode.lower() == 'mr':
            return self.case_mr
        elif self.mode.lower() == 'us':
            return self.case_us
        else:
            return {'mr': self.case_mr, 'us': self.case_us}

    def get_patient_num(self):
        if self.mode.lower() == 'mr':
            return len(self.case_mr)
        elif self.mode.lower() == 'us':
            return len(self.case_us)
        else:
            return {'mr': len(self.case_mr), 'us': len(self.case_us)}

    def print_custom_info(self, *args, **kwargs):
        self.print_data_describe(*args, **kwargs)

    def print_data_describe(self, *args, **kwargs):
        print(self.get_patient_num())

        for phase, case_list in zip(['us', 'mr'], [self.case_us, self.case_mr]):
            print("{:*^120s}".format(phase))
            molecule = 10 if phase == 'mr' else 100
            origin_set = set()
            direction_set = set()

            label_size_set = set()
            label_shape_set = set()

            shape_set = set()
            shape_x_set = set()
            shape_z_set = set()

            spacing_set = set()
            spacing_x_set = set()
            spacing_z_set = set()

            physical_set = set()
            physical_z_set = set()
            physical_x_set = set()
            for patient in case_list:
                print(patient['image'])
                img, img_info = self._read_img(patient['image'])
                label, label_info = self._read_img(patient['label'])

                print_numpy(img, shp=True, percentile=True)
                print_numpy(label, shp=True, percentile=True)

                label_size = tuple(map(lambda x: x[1]-x[0], get_foreground_shape(label)))
                scan_size = tuple(map(lambda x, y: round(x*y/molecule, 2), img.shape[::-1], img_info[2]))
                label_act_size = tuple(map(lambda x, y: round(x*y/molecule, 2), label_size[::-1], img_info[2]))
                print(f'scan_size:{scan_size}cm \t label_act_size:{label_act_size}cm')
                label_shape_set.add(tuple(get_foreground_shape(label)))
                label_size_set.add(label_size)

                origin_set.add(img_info[0])
                direction_set.add(img_info[1])

                shape_set.add(img.shape)
                shape_x_set.add(img.shape[-1])
                shape_z_set.add(img.shape[0])

                spacing_set.add(img_info[2])
                spacing_x_set.add(img_info[2][0])
                spacing_z_set.add(img_info[2][-1])

                physical_length = np.around(np.array(img.shape[::-1])*np.array(img_info[2])/molecule, 1)

                physical_set.add(tuple(physical_length.tolist()))
                physical_z_set.add(physical_length[-1])
                physical_x_set.add(physical_length[0])

            print('label_size: ')
            for lb_size in label_size_set:
                print(lb_size)
            print('label_shape: ')
            for lb_shape in label_shape_set:
                print(lb_shape)

            print('all origin', origin_set)
            for origin in origin_set:
                print('origin:', origin)
            print('all direction', direction_set)
            for direction in direction_set:
                print('direction:', direction)

            print('shape_set::', shape_set)
            print('shape_x_set:', shape_x_set)
            print('shape_z_set:', shape_z_set)
            print('space set:', spacing_set)
            print('space x:', spacing_x_set)
            print('space z:', spacing_z_set)
            print('physical set:', physical_set)
            print('physical_z_set', physical_z_set)
            print('physical_x_set', physical_x_set)

            for shape in shape_set:
                print('shape: ', shape)
            print('min_x:{:<5.0f}, max_x:{:<5.0f}'.format(min(shape_x_set), max(shape_x_set)))
            print('min_z:{:<5.0f}, max_z:{:<5.0f}'.format(min(shape_z_set), max(shape_z_set)))
            for space in spacing_set:
                print('spacing:', space)
            print('min_x:{:<5.4f}mm, max_x:{:<5.4f}mm'.format(min(spacing_x_set)/molecule*10,
                                                              max(spacing_x_set)/molecule*10))
            print('min_z:{:<5.4f}mm, max_z:{:<5.4f}mm'.format(min(spacing_z_set)/molecule*10,
                                                              max(spacing_z_set)/molecule*10))
            for phy in physical_set:
                print('physical length:{}cm'.format(phy))
            print('min_x:{:<4.2f}cm, max_x:{:<4.2f}cm'.format(min(physical_x_set), max(physical_x_set)))
            print('min_z:{:<4.2f}cm, max_z:{:<4.2f}cm'.format(min(physical_z_set), max(physical_z_set)))


def main():
    dataroot = r'/home/lf/data_fong/CODE/PycharmProject/UMMS/traces/datasets/MR-USviaFenster20_pre/'

    dataset = DatasetTool(dataroot=dataroot, mode='us')

    dataset.print_custom_info()

    print("{:*^120s}".format('end'))


if __name__ == "__main__":
    main()


