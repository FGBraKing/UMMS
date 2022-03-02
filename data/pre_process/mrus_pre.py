# -*- coding:utf-8 -*-
import os
import re
import random
import pandas as pd
import numpy as np
import SimpleITK as sitk
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


class MrusPre(DatasetPre):
    @staticmethod
    def addition_process(img, img_info, *args, separate_z_anisotropy_threshold=3, **kwargs):
        '''
        :param img: DHW
        :param img_info: origin, direction, spacing
        :param args:
        :param separate_z_anisotropy_threshold:
        :param kwargs: 'do_separate_z','is_label', 'new_spacing'
        :return:
        '''

        if 'is_label' in kwargs.keys():
            is_label = kwargs['is_label']
        else:
            is_label = False

        if 'new_spacing' in kwargs.keys():
            new_spacing = kwargs['new_spacing']
        else:
            new_spacing = [0.625, 0.625, 1.5]

        old_origin = img_info[0]
        old_direction = img_info[1]
        old_spacing = img_info[2]

        if 'do_separate_z' in kwargs.keys():
            do_separate_z = kwargs['do_separate_z']
            if do_separate_z:
                axis = get_lowres_axis(old_spacing)
            else:
                axis = None
        elif get_do_separate_z(old_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(old_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

        if axis is not None:
            if len(axis) == 3:
                # every axis has the spacing, this should never happen, why is this code here?
                do_separate_z = False
            elif len(axis) == 2:
                # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
                # separately in the out of plane axis
                do_separate_z = False
            else:
                pass

        if is_label:
            order = 1
            order_z = 0
        else:
            order = 3
            order_z = 1

        img = np.expand_dims(img, axis=(0,1))
        print(img.shape)
        img, _ = crop(img, crop_size=(350, 448, 448), margins=(0, 0, 0), crop_type="center")
        print(img.shape)
        img = np.squeeze(img, axis=(0, 1))

        old_shape = img.shape[::-1]     # x y z
        new_shape = np.round(((np.array(old_spacing) / np.array(new_spacing)).astype(float) * old_shape)).astype(int)

        # appending the most priority "new_shape"
        if 'new_shape' in kwargs.keys():
            new_shape = kwargs['new_shape']

        img = np.expand_dims(img.transpose([2, 1, 0]), axis=0)  # cxyz
        out_img = resample_data_or_seg(img, new_shape, is_label, axis=axis,
                                       order=order, do_separate_z=do_separate_z, order_z=order_z)
        out_img = out_img[0]

        # other info
        new_spacing_refine = (np.array(old_shape) * old_spacing / out_img.shape).tolist()
        out_info = old_origin, old_direction, new_spacing_refine
        out_img = out_img.transpose([2, 1, 0])      # zyx
        print('new spacing:{}'.format(new_spacing_refine))

        if not is_label:
            out_img = standardize(out_img, out_img.mean(), out_img.std())

        return out_img, out_info

    def __init__(self, dataroot, seed=1008, mode='all', **kwargs):
        super(MrusPre, self).__init__(dataroot, seed, **kwargs)
        # save the parameters
        self.mode = mode
        # get the image and label path among all modal
        assert os.path.isdir(self.dataroot)
        random.seed(seed)
        mr_case = []
        for root, dirs, files in os.walk(self.dataroot):
            for name in files:
                if name.endswith('.nii') and 'MR' in name and 'state' not in name:
                    # print(name)
                    mr_case.append(os.path.join(root, name))
        pat = re.compile(r'_MR')
        self.case_mr = [{'image': path,
                         'label': pat.sub('_MR_Prostate', path)} for path in mr_case]
        self.case_us = [{'image': pat.sub('_US', path),
                         'label': pat.sub('_US_Prostate', path)} for path in mr_case]

    def shuffle_list(self):
        if self.mode.lower() == 'mr':
            random.shuffle(self.case_mr)
        elif self.mode.lower() == 'us':
            random.shuffle(self.case_us)
        else:
            random.shuffle(self.case_mr)
            random.shuffle(self.case_us)

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

    def split_train_val_test(self, *ratio, shuffle=True):
        if np.sum(ratio[:3]) != 1:
            ratio = np.array(ratio) / np.sum(ratio[:3])

        if shuffle:
            self.shuffle_list()

        mr_num = len(self.case_mr)
        train_mr_num = int(mr_num * ratio[0])
        val_mr_num = int(mr_num * ratio[1])

        us_num = len(self.case_us)
        train_us_num = int(us_num * ratio[0])
        val_us_num = int(us_num * ratio[1])

        if self.mode.lower() == 'mr':
            return self.case_mr[:train_mr_num], \
                   self.case_mr[train_mr_num:train_mr_num+val_mr_num], \
                   self.case_mr[train_mr_num+val_mr_num:]

        elif self.mode.lower() == 'us':
            return self.case_us[:train_us_num], \
                   self.case_us[train_us_num:train_us_num+val_us_num], \
                   self.case_us[train_us_num+val_us_num:]
        else:
            return {'mr': [self.case_mr[:train_mr_num],
                           self.case_mr[train_mr_num:train_mr_num+val_mr_num],
                           self.case_mr[train_mr_num+val_mr_num:]],
                    'us': [self.case_us[:train_us_num],
                           self.case_us[train_us_num:train_us_num+val_us_num],
                           self.case_us[train_us_num+val_us_num:]]}

    def process_and_save_data(self, save_root, split_ratio=(3, 1, 1), transform=None,
                              save_csv=False, split_name='split.csv', **kwargs):
        '''
        :param save_root: The save_root of preprocess data
        :param split_ratio: train:val:test
        :param transform:
        :param save_csv: whether to save csv file
        :param split_name: the csv file's name
        :param kwargs:
            1 process parameter:
                modal: folder's prefix
                save_type: file save type
                if_slim: whether to slim array
            2. addition_process's parameter:
                ...
        :return: None
        '''
        phase_list = ['train', 'val', 'test']
        data_list = self.split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        if save_csv:
            if self.mode.lower() == 'mr':
                data_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list]  # 三个map对象
                data_df = pd.DataFrame(data=data_name_list, index=phase_list)
                data_df.T.to_csv(os.path.join(save_root, 'mr_'+split_name), index=False)
            elif self.mode.lower() == 'us':
                data_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list]  # 三个map对象
                data_df = pd.DataFrame(data=data_name_list, index=phase_list)
                data_df.T.to_csv(os.path.join(save_root, 'us_'+split_name), index=False)
            else:
                mr_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list['mr']]
                us_name_list = [map(lambda x:os.path.basename(x['image']).split('.')[0], data) for data in data_list['us']]
                data_df = pd.DataFrame(data=mr_name_list+us_name_list, index=phase_list+phase_list)
                data_df.T.to_csv(os.path.join(save_root, 'mr_us_'+split_name), index=False)

        if self.mode.lower() == 'mr':
            for data, phase in zip(data_list, phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='mr', **kwargs)
        elif self.mode.lower() == 'us':
            for data, phase in zip(data_list, phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='us', **kwargs)
        else:
            for data, phase in zip(data_list['mr'], phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='mr', **kwargs)
            for data, phase in zip(data_list['us'], phase_list):
                self._process_and_save_data(data, phase, transform, save_root, modal='us', **kwargs)

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


def convert_dataset_for_nnunet(data_root):
    def load_convert_save(path, is_label, output_folder):
        dataname = os.path.basename(path)

        if 'MR' in dataname:
            prefix = 'MRprostate'
        elif 'US' in dataname:
            prefix = 'USprostate'
        else:
            raise ValueError
        dataid = re.search(r'\d+', dataname).group()
        if is_label:
            modal_name = ''
        elif 'MR' in name or 'US' in name:
            modal_name = '_0000'
        else:
            raise ValueError
        suffix = r'.nii.gz'
        sitk.WriteImage(sitk.ReadImage(path), join(output_folder, prefix+'_'+dataid+modal_name+suffix))
    nnunet_raw_data = r'/home/lf/raid_lf/nnUNet_materials/nnUNet_raw/nnUNet_raw_data'
    us_dir = join(nnunet_raw_data, 'Task600_ProstateTRUS')
    mr_dir = join(nnunet_raw_data, 'Task601_ProstateMR')
    for task_dir in [us_dir, mr_dir]:
        for data_type in ['images', 'labels']:
            for suffix in ['Tr', 'Ts']:
                used_dir = join(task_dir, data_type+suffix)
                maybe_mkdir_p(used_dir)

    mr_case = []
    for root, dirs, files in os.walk(data_root):
        for name in files:
            if name.endswith('.nii') and 'MR' in name and 'state' not in name:
                # print(name)
                mr_case.append(os.path.join(root, name))
    pat = re.compile(r'_MR')
    case_list = [{
        'MR_image': path,
        'MR_label': pat.sub('_MR_Prostate', path),
        'US_image': pat.sub('_US', path),
        'US_label': pat.sub('_US_Prostate', path)} for path in mr_case]

    random.seed(1008)
    random.shuffle(case_list)
    for case in case_list[:16]:
        load_convert_save(case['MR_image'], False, join(mr_dir, "imagesTr"))
        load_convert_save(case['US_image'], False, join(us_dir, "imagesTr"))
        load_convert_save(case['MR_label'], True, join(mr_dir, "labelsTr"))
        load_convert_save(case['US_label'], True, join(us_dir, "labelsTr"))
    for case in case_list[16:]:
        load_convert_save(case['MR_image'], False, join(mr_dir, "imagesTs"))
        load_convert_save(case['US_image'], False, join(us_dir, "imagesTs"))
        load_convert_save(case['MR_label'], True, join(mr_dir, "labelsTs"))
        load_convert_save(case['US_label'], True, join(us_dir, "labelsTs"))

    generate_dataset_json(join(us_dir, 'dataset.json'),
                          join(us_dir, "imagesTr"),
                          join(us_dir, "imagesTs"),
                          ("US",),
                          {"0": "background", "1": "prostate"},
                          dataset_name="USProstate",
                          dataset_description="prostate in us image",
                          dataset_release="0.0")

    generate_dataset_json(join(mr_dir, 'dataset.json'),
                          join(mr_dir, "imagesTr"),
                          join(mr_dir, "imagesTs"),
                          ("MR",),
                          {"0": "background", "1": "prostate"},
                          dataset_name="MRProstate",
                          dataset_description="prostate in mr image",
                          dataset_release="0.0")

    # us_dict = OrderedDict()
    # us_dict['name'] = "USProstate"
    # us_dict['description'] = 'prostate in us image'
    # us_dict['tensorImageSize'] = '4D'
    # us_dict['reference'] = 'lab 315'
    # us_dict['licence'] = 'lab 315'
    # us_dict['release'] = '0.0'
    # us_dict['modality'] = {
    #     "0": "US",
    # }
    # us_dict['labels'] = {
    #     "0": "background",
    #     "1": "prostate"
    # }
    # us_dict['numTraining'] = 16
    # us_dict['numTest'] = 4
    #
    # us_dict['training'] = [{'image': "", 'label': ""}]
    # us_dict['test'] = [""]
    #
    # mr_dict = OrderedDict()


def main():
    dataroot = r'/home/lf/data_fong/DATA/MR-USviaFenster20/'
    # saveroot = r'/raid/lf/PROJECT/DLForPytorch/traces/datasets/MR-USviaFenster20_pre'

    saveroot = r'/home/lf/data_fong/CODE/PycharmProject/UMMS/traces/datasets/MR-USviaFenster20_pre'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)

    dataset = MrusPre(dataroot=dataroot, mode='us')

    dataset.process_and_save_data(saveroot,
                                  save_csv=True,
                                  split_name='split.csv',
                                  save_type='nii',        # _process_and_save_data
                                  if_slim=False,          # _process_and_save_data
                                  do_separate_z=False,    # addition_process
                                  new_spacing=[2, 2, 2],  # addition_process
                                  new_shape=[192, 192, 160])

    # dataset.print_custom_info()
    # convert_dataset_for_nnunet(dataroot)

    print("{:*^120s}".format('end'))


if __name__ == "__main__":
    main()


