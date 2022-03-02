# -*- coding:utf-8 -*-
import os
import re
import numpy as np
import SimpleITK as sitk
from glob import glob
from collections import OrderedDict
# from skimage.transform import resize
# from scipy.ndimage.interpolation import zoom
# from scipy.interpolate import interp1d, interp2d
# from data.transforms.transforms import resize_image_itk, Compose
from data.transforms.transformOnArray import standardize, normalize
from data.pre_process.dataset_pre import DatasetPreOne
from data.preprocessing import resample_data_or_seg, get_lowres_axis, get_do_separate_z
# from data.utils_data import save_nii, nii_loader, npy_loader, h5_loader
from utils.others.utils import cut_off_outliers, slim_array, get_foreground_shape
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p


class PromisePre(DatasetPreOne):
    @staticmethod
    def addition_process(img, img_info, *args, separate_z_anisotropy_threshold=3, **kwargs):
        '''
        :param img: DHW
        :param img_info: origin, direction, spacing
        :param args:
        :param separate_z_anisotropy_threshold:
        :param kwargs: 'kit','do_separate_z','is_label', 'new_spacing'
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

        old_shape = img.shape[::-1]     # x y z
        new_shape = np.round(((np.array(old_spacing) / np.array(new_spacing)).astype(float) * old_shape)).astype(int)

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

    def __init__(self, dataroot, seed=1008):
        super(PromisePre, self).__init__(dataroot, seed)

        used_dir_list = glob(self.dataroot+'/Training*')
        segment_list = []
        for used_dir in used_dir_list:
            filelist = [os.path.join(used_dir, name) for name in os.listdir(used_dir)
                        if name.endswith('segmentation.mhd')]
            segment_list.extend(filelist)
        case_pat = re.compile(r'\_segment\w+')
        self.case_list = [{'image': case_pat.sub('', name), 'label': name} for name in segment_list]
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
            scan_size = tuple(map(lambda x, y: round(x*y/10, 2), img.shape[::-1], img_info[2]))
            label_act_size = tuple(map(lambda x, y: round(x*y/10, 2), label_size[::-1], img_info[2]))
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

            physical_length = np.around(np.array(img.shape[::-1])*np.array(img_info[2])/10, 1)

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
        print('min_x:{:<5.4f}mm, max_x:{:<5.4f}mm'.format(min(spacing_x_set), max(spacing_x_set)))
        print('min_z:{:<5.4f}mm, max_z:{:<5.4f}mm'.format(min(spacing_z_set), max(spacing_z_set)))
        for phy in physical_set:
            print('physical length:{}cm'.format(phy))
        print('min_x:{:<4.2f}cm, max_x:{:<4.2f}cm'.format(min(physical_x_set), max(physical_x_set)))
        print('min_z:{:<4.2f}cm, max_z:{:<4.2f}cm'.format(min(physical_z_set), max(physical_z_set)))

    def convert_dataset_for_nnunet(self, shuffle=False,
                                   nnunet_raw_data=r'/home/lf/raid_lf/nnUNet_materials/nnUNet_raw/nnUNet_raw_data',
                                   split_ratio=(3, 1, 1)):

        task_dir = join(nnunet_raw_data, 'Task602_Promise12')
        data_list_list = self.split_train_val_test(*split_ratio, shuffle=shuffle)
        train_val_list, test_list = data_list_list[0]+data_list_list[1], data_list_list[2]

        maybe_mkdir_p(join(task_dir, "imagesTr"))
        maybe_mkdir_p(join(task_dir, "imagesTs"))
        maybe_mkdir_p(join(task_dir, "labelsTr"))
        maybe_mkdir_p(join(task_dir, "labelsTs"))

        for data in train_val_list:
            save_image_path = join(task_dir, "imagesTr", data['image'].split("/")[-1][:-4]+"_0000.nii.gz")
            save_label_path = join(task_dir, "labelsTr", data['label'].split("/")[-1][:-17]+".nii.gz")
            sitk.WriteImage(sitk.ReadImage(data['image']), save_image_path)
            sitk.WriteImage(sitk.ReadImage(data['label']), save_label_path)
        for data in test_list:
            save_image_path = join(task_dir, "imagesTs", data['image'].split("/")[-1][:-4]+"_0000.nii.gz")
            save_label_path = join(task_dir, "labelsTs", data['label'].split("/")[-1][:-17]+".nii.gz")
            sitk.WriteImage(sitk.ReadImage(data['image']), save_image_path)
            sitk.WriteImage(sitk.ReadImage(data['label']), save_label_path)

        json_dict = OrderedDict()
        json_dict['name'] = "PROMISE12"
        json_dict['description'] = "prostate"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "see challenge website"
        json_dict['licence'] = "see challenge website"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "MRI",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "prostate"
        }
        json_dict['numTraining'] = len(train_val_list)
        json_dict['numTest'] = len(test_list)
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % data['image'].split("/")[-1][:-4],
                                  "label": "./labelsTr/%s.nii.gz" % data['image'].split("/")[-1][:-4]}
                                 for data in train_val_list]
        json_dict['test'] = ["./imagesTs/%s.nii.gz" % data['image'].split("/")[-1][:-4] for data in test_list]
        save_json(json_dict, os.path.join(task_dir, "dataset.json"))


def main():
    dataroot = '/home/lf/raid_lf/DATA/promise12/origin_data'
    save_root = '/home/lf/raid_lf/PROJECT/DLForPytorch/traces/datasets/promise12_pre'
    maybe_mkdir_p(save_root)

    dataset = PromisePre(dataroot, seed=1008)
    dataset.process_and_save_data(save_root=save_root,
                                  split_ratio=(3, 1, 1),
                                  transform=None,
                                  save_csv=True,
                                  split_name='split.csv',
                                  modal='',         # _process_and_save_data
                                  save_type='nii',  # _process_and_save_data
                                  if_slim=True,     # _process_and_save_data
                                  do_separate_z=True,  # addition_process
                                  new_spacing=[0.625, 0.625, 1.5])  # addition_process
    # dataset.convert_dataset_for_nnunet(shuffle=False)
    # dataset.print_custom_info()

    print("end")

    # for root, dirs, files in os.walk(save_root):
    #     for file in files:
    #         if file.endswith('.nii') and 'itk' not in file and 'sci' not in file:
    #             print(os.path.join(root, file))
    #             os.remove(os.path.join(root, file))


if __name__ == "__main__":
    main()


