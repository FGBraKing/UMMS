# -*- coding:utf-8 -*-
import os
import h5py
import random
import nibabel as nib
import numpy as np
from glob import glob


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist path: ', path)


# /data/project_data_lf/BraTS2018
class BraTs2018_pre():
    def __init__(self, dataroot, deep_dir=3, patient_base=True, seed=1008):
        random.seed(seed)
        self.dataroot = dataroot
        self.deep_dir = deep_dir
        self.patient_base = patient_base
        assert os.path.isdir(dataroot)

        patientlist = [dataroot]
        deep = 1
        while deep < deep_dir:
            patientlist_new = []
            for patient in patientlist:
                tmp_list = [patient+'/'+tt for tt in os.listdir(patient) if os.path.isdir(os.path.join(patient, tt))]
                for pat in tmp_list:
                    patientlist_new.append(pat)
            patientlist = patientlist_new
            deep += 1
        self.patientlist = patientlist
        self.filelist = []
        for dirpath, dirnames, filenames in os.walk(dataroot):
            for filename in filenames:
                self.filelist.append(os.path.join(dirpath, filename))

    def get_patient_num(self):
        return len(self.patientlist)

    def shuffle_list(self):
        random.shuffle(self.patientlist)
        random.shuffle(self.filelist)

    def _split_train_val_test(self, *ratio, shuffle=True):
        if np.sum(ratio[:3]) != 1:
            ratio = np.array(ratio) / np.sum(ratio[:3])
        patient_num = self.get_patient_num()
        train_num = int(patient_num * ratio[0])
        val_num = int(patient_num * ratio[1])
        # test_num = patient_num - train_num - val_num
        if shuffle:
            self.shuffle_list()
        return self.patientlist[:train_num], self.patientlist[train_num:train_num+val_num], self.patientlist[train_num+val_num:]

    def get_patient_list(self):
        return self.patientlist

    def process_and_save_data(self, save_root, split_ratio=(7, 1, 1), transform=None):
        phase_list = ['train', 'val', 'test']
        data_list = self._split_train_val_test(*split_ratio)  # train_list, val_list, test_list
        for data, phase in zip(data_list, phase_list):
            self._process_and_save_data(data, phase, transform, save_root)

    def _process_and_save_data(self, patient_list, phase, transform=None, save_root=None):
        mkdir(os.path.join(save_root, phase))
        for patient in patient_list:
            t1_path = glob(patient+'/*t1.nii')
            t2_path = glob(patient+'/*t2.nii')
            t1_img = self._read_img(t1_path[0])
            t2_img = self._read_img(t2_path[0])

            slice_start = t1_img.shape[-1]//2 - 5
            t1_warp = self._process_img(t1_path[0], transform)

            patient_name = os.path.basename(patient)
            # print(patient_name)
            save_path = os.path.join(save_root, phase, patient_name+'.h5')
            # print(save_path)
            self._save_img(save_path, t1_img[:, :, slice_start:slice_start+10],
                           t2_img[:, :, slice_start:slice_start+10], t1_warp)

    def _read_img(self, path):
        img = nib.load(path)
        img_array = img.get_data()
        return img_array

    def _process_img(self, img_path, transform):
        img_data = self._read_img(img_path)
        img_shape_half = img_data.shape[-1]//2
        img_data_used = img_data[:, :, img_shape_half-5:img_shape_half+5]
        img_data_warp = img_data_used
        # img_data_warp = elastic_transform(img_data_used, 0.3, 0.2, 10)
        if transform:
            img_data_warp = transform(img_data_warp)
        return img_data_warp

    def _save_img(self, path, img_t1, img_t2, warp_t1):
        f_write = h5py.File(path, 'w')
        f_write.create_dataset('t1', data=img_t1)
        f_write.create_dataset('t2', data=img_t2)
        f_write.create_dataset('wrap_t1', data=warp_t1)
        f_write.close()


def split_AB(dataroot):
    assert os.path.isdir(os.path.join(dataroot, 'train')), 'there is not train'
    assert os.path.isdir(os.path.join(dataroot, 'test')), 'there is not test'
    data_classes = ['A', 'B']
    phases = ['train', 'test', 'val']
    for data_class in data_classes:
        for phase in phases:
            mkdir(os.path.join(dataroot, phase+data_class))
    for dirroot, _, filenames in os.walk(dataroot):
        for filename in filenames:
            if filename.endswith('h5'):
                path = os.path.join(dirroot, filename)
                f_read = h5py.File(path, mode='r')
                img_t1 = f_read.get('t1')[:]
                img_t2 = f_read.get('t2')[:]
                img_warp_t1 = f_read.get('wrap_t1')[:]
                f_read.close()
                save_path_t1 = os.path.join(dirroot+'A', filename[:-2]+'npy')
                save_path_t2 = os.path.join(dirroot+'B', filename[:-2]+'npy')
                np.save(save_path_t1, img_warp_t1)
                np.save(save_path_t2, img_t2)


if __name__ == "__main__":
    dataroot = '/data/project_data_lf/BraTS2018'
    patient_deep = 3
    seed = 1008
    dataset = BraTs2018_pre(dataroot, patient_deep, seed=seed)
    save_root = '/data/project_data_lf/DLForPytorch/datasets/BraTs2018-IPML'
    split_ratio = (7, 1, 1)
    # dataset.process_and_save_data(save_root, split_ratio)
    split_AB(save_root)
    print(dataset.get_patient_num())
    print("end")


