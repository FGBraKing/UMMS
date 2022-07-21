# -*- coding:utf-8 -*-
import os
import random
import re
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import OrderedDict

from utils.others.utils import mkdir, get_foreground_shape
from data.transforms.transforms import resize_image_itk, Compose
from data.transforms.transformOnArray import standardize, normalize
from data.pre_process.dataset_pre import DatasetPreOne
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom

from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p, subfiles

# 二维：cv2.resize()，np.resize()
# 三维
# 1. scipy.ndimage.interpolation.zoom()
# 2. torch.nn.functional.interpolate()


def h52nii(data_root):
    phase_list = ['train', 'test', 'val']

    def h5_read(path, name):
        with h5py.File(path, mode='a') as f:
            data = f.get(name)[:]   # # W D H
            del f[name]
            f[name] = data.transpose([2, 0, 1])  # H W D
        return data.transpose([1, 2, 0])  # D H W

    def nii_save(path, data):
        itk_img = sitk.GetImageFromArray(data)
        sitk.WriteImage(itk_img, path)

    for phase in phase_list:
        root_dir = os.path.join(data_root, phase)
        filelist = [file for file in os.listdir(root_dir) if file.endswith('h5')]
        for file in filelist:
            print(file)
            path = os.path.join(root_dir, file)
            image = h5_read(path, 'img')  # D H W (80,132,170)
            label = h5_read(path, 'label')
            print(image.shape)
            label_path = os.path.join(root_dir, file.replace('data', 'label').replace('h5', 'nii'))
            image_path = os.path.join(root_dir, file.replace('data', 'image').replace('h5', 'nii'))
            print(image_path)
            nii_save(label_path, label)
            nii_save(image_path, image)


class TrusPre(DatasetPreOne):
    @staticmethod
    def addition_process(img, img_info, *args, **kwargs):
        '''
        :param img: DHW
        :param img_info: origin, direction, spacing
        :param args:
        :param kwargs: 'kit','do_separate_z','is_label', 'new_spacing'
        :return:
        '''
        if 'kit' in kwargs.keys():
            kit = kwargs['kit']
        else:
            kit = 'itk'

        if 'do_separate_z' in kwargs.keys():
            do_separate_z = kwargs['do_separate_z']
        else:
            do_separate_z = False

        if 'is_label' in kwargs.keys():
            is_label = kwargs['is_label']
        else:
            is_label = False

        if 'new_spacing' in kwargs.keys():
            new_spacing = kwargs['new_spacing']
        else:
            new_spacing = [2, 2, 2]

        old_origin = img_info[0]
        old_direction = img_info[1]
        old_spacing = img_info[2]

        itk_img = sitk.GetImageFromArray(img)
        itk_img.SetSpacing(old_spacing)
        itk_img.SetOrigin(old_origin)
        itk_img.SetDirection(old_direction)

        if is_label:
            resamplemethod = sitk.sitkNearestNeighbor
            # N4BiasCorrect = False
            order = 0
        else:
            resamplemethod = sitk.sitkBSplineResamplerOrder3        # sitkBSplineResamplerOrder3     sitk.sitkLinear
            order = 3
            # N4BiasCorrect = True

        if do_separate_z:
            old_shape = img.shape[::-1]
            new_shape = np.array(old_shape) * old_spacing / new_spacing
            new_shape = np.round(new_shape)
            # x, y
            reshaped_data = []
            for slice_id in range(img.shape[0]):
                reshaped_data.append(resize(img[slice_id, :, :], new_shape[:-1][::-1], order,
                                            cval=0, mode='edge', anti_aliasing=False))
            reshaped_data = np.stack(reshaped_data, axis=0)     # z y x
            # z
            resize_factor_z = old_spacing[0] / new_spacing[0]
            resize_factor = [1, 1, resize_factor_z]
            out_img = zoom(reshaped_data.transpose([2, 1, 0]), resize_factor, order=0, mode='nearest', cval=0.0)
            # other info
            new_spacing_refine = (np.array(old_shape) * old_spacing / out_img.shape).tolist()
            out_info = img_info[0], img_info[1], new_spacing_refine
            out_img = out_img.transpose([2, 1, 0])
            print('new spacing:{}'.format(new_spacing_refine))
            # if new_shape[-1] != img.shape[0]:
            #     # copied from nnunet
            #     rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
            #     orig_dim, orig_cols, orig_rows = reshaped_data.shape
            #
            #     row_scale = float(orig_rows) / rows
            #     col_scale = float(orig_cols) / cols
            #     dim_scale = float(orig_dim) / dim
            #
            #     map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
            #     map_rows = row_scale * (map_rows + 0.5) - 0.5
            #     map_cols = col_scale * (map_cols + 0.5) - 0.5
            #     map_dims = dim_scale * (map_dims + 0.5) - 0.5
            #
            #     coord_map = np.array([map_rows, map_cols, map_dims])
            #
            #     reshaped_final_data = map_coordinates(reshaped_data, coord_map,
            #                                           order=order, cval=0, mode='nearest')[None]
        elif kit == 'itk':
            print('origin spacing:{}'.format(old_spacing))
            itk_img_resized = resize_image_itk(itk_img,
                                               newSpacing=new_spacing,
                                               newOrigin=old_origin,
                                               newDirection=old_direction,
                                               resamplemethod=resamplemethod,
                                               N4BiasCorrect=False)
            out_img = sitk.GetArrayFromImage(itk_img_resized)  # z,y,x
            out_info = itk_img_resized.GetOrigin(), itk_img_resized.GetDirection(), itk_img_resized.GetSpacing()
            print('new spacing:{}'.format(old_spacing))
        else:
            print('origin spacing:{}'.format(old_spacing))
            resize_factor = np.array(old_spacing, float) / new_spacing
            out_img = zoom(img.transpose([2, 1, 0]), resize_factor, order=order, mode='constant', cval=0.0)
            new_spacing_refine = (np.array(img.shape[::-1]) * old_spacing / out_img.shape).tolist()
            out_info = img_info[0], img_info[1], new_spacing_refine
            out_img = out_img.transpose([2, 1, 0])
            print('new spacing:{}'.format(new_spacing_refine))
        if not is_label:
            out_img = standardize(out_img, out_img.mean(), out_img.std())

        return out_img, out_info

    def __init__(self, data_root, seed=1008, **kwargs):
        super(TrusPre, self).__init__(data_root, seed, **kwargs)
        # self.img_list = []
        # self.label_list = []
        # for dirpath, dirnames, filenames in os.walk(data_root):
        #     for filename in filenames:
        #         if filename.endswith('image.nii'):
        #             self.img_list.append(os.path.join(dirpath, filename))
        #         elif filename.endswith('label.nii'):
        #             self.label_list.append(os.path.join(dirpath, filename))
        self.img_root = os.path.join(self.dataroot, 'image')
        self.label_root = os.path.join(self.dataroot, 'label')
        pat_num = re.compile(r'P(\d+)\_')
        patient_numlist = [pat_num.findall(name)[0] for name in os.listdir(self.img_root)]
        self.case_list = [{'image': os.path.join(self.img_root, 'P'+name+r'_image.nii'),
                          'label': os.path.join(self.label_root, 'P' + name + r'_label.nii')}
                          for name in patient_numlist]
        self.case_num = len(self.case_list)

    def print_custom_info(self, *args, **kwargs):
        self.print_data_describe(*args, **kwargs)

    def print_data_describe(self, *args, **kwargs):
        print(self.get_patient_num())
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
        for patient in self.case_list:
            img, img_info = self._read_img(patient['image'])
            label, label_info = self._read_img(patient['label'])

            label_size = tuple(map(lambda x: x[1]-x[0], get_foreground_shape(label)))
            scan_size = tuple(map(lambda x, y: round(x*y/100, 2), img.shape[::-1], img_info[2]))
            label_act_size = tuple(map(lambda x, y: round(x*y/100, 2), label_size[::-1], img_info[2]))
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

            physical_length = np.around(np.array(img.shape[::-1])*np.array(img_info[2])/100, 1)

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
        print('min_x:{:<5.4f}mm, max_x:{:<5.4f}mm'.format(min(spacing_x_set)/10, max(spacing_x_set)/10))
        print('min_z:{:<5.4f}mm, max_z:{:<5.4f}mm'.format(min(spacing_z_set)/10, max(spacing_z_set)/10))
        for phy in physical_set:
            print('physical length:{}cm'.format(phy))
        print('min_x:{:<4.2f}cm, max_x:{:<4.2f}cm'.format(min(physical_x_set), max(physical_x_set)))
        print('min_z:{:<4.2f}cm, max_z:{:<4.2f}cm'.format(min(physical_z_set), max(physical_z_set)))

    def process_for_nnunet(self, save_root, split_ratio=(3, 1, 1)):
        def nii2gz(src_file, save_dir, is_label):
            name = os.path.basename(src_file)
            try:
                name_id = re.search(r'\d+', name.split('/')[-1]).group()
            except AttributeError as e:
                print(e, "your name do not have number. name:{}".format(name))
                raise AttributeError
            print(name, name_id)
            if is_label:
                save_file = os.path.join(save_dir, 'USprostate_'+name_id+'.nii.gz')
            else:
                save_file = os.path.join(save_dir, 'USprostate_'+name_id+'_0000.nii.gz')
            sitk.WriteImage(sitk.ReadImage(src_file), save_file)
        phase_list = ['Tr', 'Ts']
        data_list_list = self.split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        data_list_list = data_list_list[0]+data_list_list[1], data_list_list[2]
        self._process_save_json_for_nnunet(save_root, data_list_list)
        for data_list, phase in zip(data_list_list, phase_list):
            img_save_dir = os.path.join(save_root, 'images'+phase)
            label_save_dir = os.path.join(save_root, 'labels'+phase)
            mkdir(img_save_dir)
            mkdir(label_save_dir)
            for data in data_list:
                nii2gz(data['image'], img_save_dir, False)
                nii2gz(data['label'], label_save_dir, True)

    @staticmethod
    def _process_save_json_for_nnunet(save_root, train_test_list):
        id_pat = re.compile(r'\d+')
        json_dict = OrderedDict()
        json_dict['name'] = "MoreUSProstate"
        json_dict['description'] = "prostate"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "no"
        json_dict['licence'] = "lab 315"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "US",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "prostate"
        }

        train_list = train_test_list[0]
        test_list = train_test_list[1]
        json_dict['numTraining'] = len(train_list)
        json_dict['numTest'] = len(test_list)

        json_dict['training'] = [{'image': "./imagesTr/USprostate_%s.nii.gz" % id_pat.search(train['image'].split("/")[-1]).group(),
                                  "label": "./labelsTr/USprostate_%s.nii.gz" % id_pat.search(train['image'].split("/")[-1]).group()}
                                 for train in train_list]
        json_dict['test'] = ["./imagesTs/USprostate_%s.nii.gz" % id_pat.search(test['image'].split("/")[-1]).group()
                             for test in test_list]

        save_json(json_dict, os.path.join(save_root, "dataset.json"))


def check_data_info(dataroot=r'/home/lf/data_fong/DATA/prostate_daf3d'):
    from data.utils_data import print_data_describe
    img_root = os.path.join(dataroot, 'image')
    label_root = os.path.join(dataroot, 'label')
    pat_num = re.compile(r'P(\d+)\_')
    patient_numlist = [pat_num.findall(name)[0] for name in os.listdir(img_root)]
    case_list = [{'volume': os.path.join(img_root, 'P' + name + r'_image.nii'),
                  'label': os.path.join(label_root, 'P' + name + r'_label.nii')}
                 for name in patient_numlist]
    print_data_describe(case_list)


def main():
    dataroot = '/home/lf/data_fong/DATA/prostate_daf3d'
    # saveroot = '/raid/lf/PROJECT/DLForPytorch/traces/datasets/prostate_daf3d_pre'
    # nnunet_save_root = r'/home/lf/raid_lf/nnUNet_materials/nnUNet_raw/nnUNet_raw_data/Task603_ProstateDaf'
    # maybe_mkdir_p(saveroot)
    # maybe_mkdir_p(nnunet_save_root)

    dataset = TrusPre(dataroot, seed=1008)

    # dataset.process_and_save_data(save_root=saveroot,
    #                               split_ratio=(3, 1, 1),
    #                               transform=None,
    #                               save_csv=True,
    #                               split_name='split.csv',
    #                               save_type='nii',
    #                               if_slim=True,
    #                               do_separate_z=False,
    #                               kit='sci',
    #                               new_spacing=[2, 2, 2])
    # dataset.convert_dataset_for_nnunet(shuffle=False)
    dataset.print_custom_info()

    # process_and_save_data: save_root split_ratio transform save_csv split_name
    # _process_and_save_data: modal save_type if_sllim
    # addition_process: 'kit','do_separate_z','is_label', 'new_spacing'

    # img_resized = resize_image_itk(img_itk, newSize=(170, 132, 80), resamplemethod=sitk.sitkLinear)

    print("end")


if __name__ == "__main__":
    # main()
    check_data_info()

# sitk.ReadImage(img_path)
# <class 'SimpleITK.SimpleITK.Image'>
# ['CopyInformation', 'EraseMetaData',
# 'GetDimension', 'GetDirection', 'GetDepth', 'GetHeight', 'GetWidth', 'GetSize', 'GetSpacing', 'GetOrigin',
# 'GetITKBase', 'GetMetaData', 'GetMetaDataKeys',
# 'GetNumberOfComponentsPerPixel', 'GetNumberOfPixels',
# 'GetPixel', 'GetPixelAsComplexFloat64', 'GetPixelID', 'GetPixelIDTypeAsString', 'GetPixelIDValue',
# 'HasMetaDataKey', 'MakeUnique',
# 'SetDirection', 'SetMetaData', 'SetOrigin', 'SetPixel', 'SetPixelAsComplexFloat64', 'SetSpacing',
# 'TransformContinuousIndexToPhysicalPoint', 'TransformIndexToPhysicalPoint',
# 'TransformPhysicalPointToContinuousIndex', 'TransformPhysicalPointToIndex', 'this']

# nib.load(img_path)
# <class 'nibabel.nifti1.Nifti1Image'>
# ['ImageArrayProxy', 'ImageSlicer', 'affine', 'as_reoriented', 'dataobj', 'extra',
# 'file_map', 'files_types', 'filespec_to_file_map', 'filespec_to_files', 'from_bytes',
# 'from_file_map', 'from_filename', 'from_files', 'from_image', 'get_affine', 'get_data',
# 'get_data_dtype', 'get_fdata', 'get_filename', 'get_header', 'get_qform', 'get_sform',
# 'get_shape', 'header', 'header_class', 'in_memory', 'instance_to_filename', 'load',
# 'make_file_map', 'makeable', 'ndim', 'orthoview', 'path_maybe_image', 'rw', 'set_data_dtype',
# 'set_filename', 'set_qform', 'set_sform', 'shape', 'slicer', 'to_bytes', 'to_file_map',
# 'to_filename', 'to_files', 'to_filespec', 'uncache', 'update_header', 'valid_exts']



