# -*- coding:utf-8 -*-
import os
import re
import random
import shutil
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk

from glob import glob
from skimage.transform import resize
# from scipy.ndimage.interpolation import zoom
# from scipy.ndimage.interpolation import map_coordinates
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p

# from data.transforms.transforms import resize_image_itk, Compose
from data.utils_data import nii_loader, save_nii, print_data_describe
from data.pre_process.dataset_pre import DatasetPre
from data.transforms.transformOnArray import standardize
from data.utils_nnunet import generate_dataset_json
from data.preprocessing import resample_data_or_seg, get_lowres_axis, get_do_separate_z
from utils.others.utils import get_foreground_shape, cut_off_outliers, print_numpy, clip_array, get_bbox_from_mask, slim_array
from utils.others.img_io import show_volume_label, show_image_label
from utils.forDebugs.data_visualizer import show_density_on_one_figure


#  The following are not used
def translate_nii_to_ras(data_root=r'L:\DATA\temp_data\Formation__MR-USviaFenster20_MRI',
                         save_root=r'L:\DATA\temp_data\MR-USviaFenster20',
                         modal_flag='MR', label_flag='Prostate'):
    # RAS -> LPI

    pass


class OriginProcess(object):
    def __init__(self, data_root):
        self.data_root = data_root

    def read_vtk(self):
        import vtk
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(r'/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068_US_Prostate.vtk')
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
        reader.Update()
        vtk_data = reader.GetOutput()  # GetOutput获取文件的数据
        print(vtk_data.GetNumberOfPoints())  # GetNumberOfPoint获取点的个数
        # print(vtk_data.GetPoint())  # GetPoint获取点i的坐标

    def read_stl(self):
        import open3d as o3d
        from stl import mesh
        stl_data = o3d.io.read_triangle_mesh \
            ('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068_US_Prostate.stl')
        stl_data1 = mesh.Mesh.from_file \
            ('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068_US_Prostate.stl')

    def read_off(self):
        import open3d as o3d
        off_data = o3d.io.read_triangle_mesh \
            ('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068_US_Prostate.off')

    def read_tif(self):
        import skimage.io
        tif_data = skimage.io.imread('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068_US.tif')

    def read_mat(self):
        import scipy.io as scio
        mat_data = scio.loadmat('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068.mat')
        mr_mat_data = scio.loadmat('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_MRI/P068/P068.mat')

    def read_nii(self):
        import nibabel as nib
        nii_data = nib.load('/home/lf/raid_lf/DATA/Formation__MR-USviaFenster20_TRUS/P068/P068_US.nii').get_data()
        old_nii_data = nib.load('/home/lf/raid_lf/DATA/MR-USviaFenster20/P068/P068_US.nii').get_data()


def process_sf_sp(sf, sp):
    pass


def get_lbs_for_seg_crop(crop_size, data_shape, label_range):
    lbs = []

    for i in range(len(data_shape)):
        if crop_size[i] < (label_range[i][1] - label_range[i][0] + 1):
            warnings.warn('crop size can not cover the ROI')
            lbs.append((label_range[i][1] - label_range[i][0])//2 - crop_size[i]//2)
        else:
            # 左右两边剩下多少个点
            l_left = label_range[i][0] - 0
            r_left = data_shape[i] - label_range[i][1] - 1
            # 还需要多少个点
            need_size = crop_size[i] - (label_range[i][1] - label_range[i][0] + 1)
            if l_left <= need_size//2:
                lbs.append(0)
            elif r_left <= need_size//2:
                lbs.append(data_shape[i] - crop_size[i])
            else:
                lbs.append(max((label_range[i][1] + label_range[i][0])//2 - crop_size[i]//2, 0))
    return lbs


def do_process(spacing_s, sVol, sMask, spacing_t, tVol, tMask, aim_shape, retain=True):
    def crop_img(img, seg, aim_sh):
        img = np.expand_dims(img, axis=0)  # ncxyz
        seg = np.expand_dims(seg, axis=0)  # ncxyz
        print(img.shape)
        img, seg = crop(img, seg,  crop_size=tuple(aim_sh), margins=(0, 0, 0), crop_type="center")
        print(img.shape)
        img = np.squeeze(img, axis=(0, 1))
        seg = np.squeeze(seg, axis=(0, 1))
        return img, seg

    def crop_img1(img, seg, aim_sh):
        lab_range = get_foreground_shape(seg, number=10)
        print(aim_sh, img.shape, lab_range)
        lbs = get_lbs_for_seg_crop(aim_sh, img.shape, lab_range)

        data_slice = tuple([slice(lbs[i], lbs[i]+aim_sh[i]) for i in range(len(lbs))])
        print('lbs:', lbs)
        return img[data_slice], seg[data_slice]

    def get_roi_info(label, spacing):
        roi_range = tuple(get_foreground_shape(label, number=10))
        roi_shape = tuple(map(lambda x: x[1] - x[0], roi_range))
        roi_size = tuple(map(lambda x, y: round(x * y / 10, 2), roi_shape, spacing))
        return roi_range, roi_shape, roi_size

    # 实际大小
    s_act_size = np.array(spacing_s) * sVol.shape
    t_act_size = np.array(spacing_t) * tVol.shape
    # 求目标spacing
    aim_shape = np.array(aim_shape)
    aim_size = np.where(s_act_size > t_act_size, t_act_size, s_act_size)
    aim_spacing = aim_size / aim_shape
    # 微调后的spacing
    s_temp_shape = np.round(s_act_size / aim_spacing).astype(int)
    s_act_spacing = (s_act_size / s_temp_shape).tolist()
    t_temp_shape = np.round(t_act_size / aim_spacing).astype(int)
    t_act_spacing = (t_act_size / t_temp_shape).tolist()

    # 保证roi区域完整
    if retain:
        _, _, s_roi_size = get_roi_info(sMask, spacing_s)
        s_aim_size = np.where(aim_size > s_roi_size, aim_size, s_roi_size)
        s_aim_spacing = s_aim_size / aim_shape
        s_temp_shape = np.round(s_act_size / s_aim_spacing).astype(int)
        s_act_spacing = (s_act_size / s_temp_shape).tolist()

        _, _, t_roi_size = get_roi_info(tMask, spacing_t)
        t_aim_size = np.where(aim_size > t_roi_size, aim_size, t_roi_size)
        t_aim_spacing = t_aim_size / aim_shape
        t_temp_shape = np.round(t_act_size / t_aim_spacing).astype(int)
        t_act_spacing = (t_act_size / t_temp_shape).tolist()

    # 增加维度以方便进行resample
    sVol = np.expand_dims(sVol, axis=0)  # cxyz
    sMask = np.expand_dims(sMask, axis=0)  # cxyz
    tVol = np.expand_dims(tVol, axis=0)  # cxyz
    tMask = np.expand_dims(tMask, axis=0)  # cxyz

    sVol = resample_data_or_seg(sVol, s_temp_shape, False, do_separate_z=True, axis=[2], order=3, order_z=1)
    sMask = resample_data_or_seg(sMask, s_temp_shape, True, do_separate_z=True, axis=[2], order=1, order_z=0)
    tVol = resample_data_or_seg(tVol, t_temp_shape, False, do_separate_z=False, axis=None, order=3, order_z=1)
    tMask = resample_data_or_seg(tMask, t_temp_shape, True, do_separate_z=False, axis=None, order=1, order_z=0)
    # 裁剪到目标大小
    # 旧裁剪
    # sVol, sMask = crop_img(sVol, sMask, aim_shape)
    # tVol, tMask = crop_img(tVol, tMask, aim_shape)
    # 新裁剪
    sVol, sMask = crop_img1(sVol[0], sMask[0], aim_shape)
    tVol, tMask = crop_img1(tVol[0], tMask[0], aim_shape)
    # # 不裁剪
    # sVol, sMask = sVol[0], sMask[0]
    # tVol, tMask = tVol[0], tMask[0]

    # 归一化
    sVol = standardize(sVol, sVol.mean(), sVol.std())
    tVol = standardize(tVol, tVol.mean(), tVol.std())
    # from RAI to RAS, from xyz to zyx
    sVol = np.transpose(np.flip(sVol, axis=2), [2, 1, 0])       # WHD, RAI-->DHW, RAS
    sMask = np.transpose(np.flip(sMask, axis=2), [2, 1, 0])
    tVol = np.transpose(np.flip(tVol, axis=2), [2, 1, 0])
    tMask = np.transpose(np.flip(tMask, axis=2), [2, 1, 0])

    return sVol, sMask, s_act_spacing, tVol, tMask, t_act_spacing


def process_data(data_root, save_root, aim_shape=(128, 128, 112), do_filter=True, if_slim=True):
    assert os.path.isdir(data_root), f"{data_root}"

    pat_ids = os.listdir(data_root)

    for pat_id in pat_ids:
        if not os.path.isdir(os.path.join(data_root, pat_id)):
            continue
        if not os.path.isdir(os.path.join(save_root, pat_id)):
            os.mkdir(os.path.join(save_root, pat_id))
        print(f'{pat_id} start!')
        us_path = os.path.join(data_root, pat_id, f'{pat_id}_US.nii')
        mr_path = os.path.join(data_root, pat_id, f'{pat_id}_MR.nii')
        us_label_path = os.path.join(data_root, pat_id, f'{pat_id}_US_Prostate.nii')
        mr_label_path = os.path.join(data_root, pat_id, f'{pat_id}_MR_Prostate.nii')

        us_origin_spacing = sitk.ReadImage(us_path).GetSpacing()    # WHD
        us_image_data = nib.load(us_path).get_fdata()               # WHD, RAI
        us_label_data = nib.load(us_label_path).get_fdata()

        mr_origin_spacing = sitk.ReadImage(mr_path).GetSpacing()
        mr_image_data = nib.load(mr_path).get_fdata()
        mr_label_data = nib.load(mr_label_path).get_fdata()

        if do_filter:
            # cut_off_lower = np.percentile(us_image_data, 25)
            us_image_data = cut_off_outliers(us_image_data, 25, 99.999)
            mr_image_data = cut_off_outliers(mr_image_data, 0.001, 99.999)

        if if_slim:
            img_label_slim = slim_array(np.stack([mr_image_data, mr_label_data], axis=0), dims=(1, 2, 3), number=100)
            mr_image_data = img_label_slim[0, ...]
            mr_label_data = img_label_slim[1, ...]

            img_label_slim = slim_array(np.stack([us_image_data, us_label_data], axis=0), dims=(1, 2, 3), number=100)
            us_image_data = img_label_slim[0, ...]
            us_label_data = img_label_slim[1, ...]

        # print('get_foreground_shape: ', get_foreground_shape(mr_label_data, number=1))
        # print('get_bbox_from_mask: ', get_bbox_from_mask(mr_label_data))
        # continue
        # DHW
        mr_data, mr_mask, mr_spacing, us_data, us_mask, us_spacing =\
            do_process(mr_origin_spacing, mr_image_data, mr_label_data,
                       us_origin_spacing, us_image_data, us_label_data, aim_shape)  # WHD, RAI-->DHW, RAS

        # 保存
        save_mr_volume_path = os.path.join(save_root, pat_id, f'{pat_id}_mr_volume.nii')
        save_us_volume_path = os.path.join(save_root, pat_id, f'{pat_id}_us_volume.nii')
        save_mr_roi_path = os.path.join(save_root, pat_id, f'{pat_id}_mr_roi.nii')
        save_us_roi_path = os.path.join(save_root, pat_id, f'{pat_id}_us_roi.nii')
        save_nii(save_us_volume_path, us_data, spacing=us_spacing, direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))    # DHW, RAS
        save_nii(save_mr_volume_path, mr_data, spacing=mr_spacing, direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        save_nii(save_us_roi_path, us_mask, spacing=us_spacing, direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        save_nii(save_mr_roi_path, mr_mask, spacing=mr_spacing, direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        # print('{:*^80} finished!'.format(pat_id))
        print('{:*^80}'.format('{} finished!'.format(pat_id)))


def check_volume_mask_direction(volume, mask, **kwargs):
    if 'title' in kwargs.keys():
        title = kwargs['title']
    else:
        title = 'image-label'
    half_slice = volume.shape[0]//2
    show_image_label(volume[half_slice, :, :], mask[half_slice, :, :], title=title)


def check_all_volume_mask_direction(data_root=r'L:\DATA\temp_data\MR-USviaFenster20', data_flag='MR'):
    # L:\DATA\temp_data\MR-USviaFenster20
    pat_list = os.listdir(data_root)
    for pat_id in pat_list:
        pat_volume_path = os.path.join(data_root, pat_id, f'{pat_id}_{data_flag}.nii')
        pat_mask_path = os.path.join(data_root, pat_id, f'{pat_id}_{data_flag}_Prostate.nii')

        volume = nii_loader(pat_volume_path)
        mask = nii_loader(pat_mask_path)
        check_volume_mask_direction(volume, mask, title=pat_id)
        # os.system("pause")


def check_all_data_percentile(data_root=r'/home/lf/data_fong/DATA/MR-USviaFenster20', percentile=25, data_flag='US'):
    patients=[]
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(data_root, a)), os.listdir(data_root)))
    for i, pat_id in enumerate(pat_ids):
        pat_volume_path = os.path.join(data_root, pat_id, f'{pat_id}_{data_flag}.nii')
        pat_mask_path = os.path.join(data_root, pat_id, f'{pat_id}_{data_flag}_Prostate.nii')

        # mr_data = nii_loader(mr_path)
        us_data = nii_loader(pat_volume_path)
        print('{:*^60}'.format(pat_id))
        print_numpy(us_data, shp=True, percentile=True)
        print(f'percentile_{percentile}:', np.percentile(us_data, percentile))
        patients.append(us_data.flatten())
    patients = np.concatenate(patients)
    print('{:*^60}'.format('all patients'))
    print_numpy(patients, shp=True, percentile=True)
    print(f'percentile_{percentile}:', np.percentile(patients, percentile))


def check_all_data_density(dataroot=r'/home/lf/data_fong/DATA/MR-USviaFenster20', data_flag='MR'):
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))

    for i, p_id in enumerate(pat_ids):
        if i != 3:
            continue
        pat_volume_path = os.path.join(dataroot, p_id, f'{p_id}_{data_flag}.nii')
        pat_mask_path = os.path.join(dataroot, p_id, f'{p_id}_{data_flag}_Prostate.nii')

        pat_data = nii_loader(pat_volume_path)

        resized_data = resize(pat_data, (96, 128, 128), order=3, preserve_range=True)

        do_data = resized_data

        print('{:*^60}'.format(p_id))
        show_density_on_one_figure(do_data, None, data_flag, None, title=f'{i + 1}:{p_id}_origin')
        do_data = cut_off_outliers(do_data, 0.0001, 99.9999)
        print_numpy(do_data, shp=True, percentile=True)
        print(np.percentile(do_data, 25))
        # do_data = cut_off_outliers(do_data, 25, 99.9, per_channel=False)
        print_numpy(do_data, shp=True, percentile=True)
        # do_data = resize(do_data, (96, 128, 128), order=3, preserve_range=True, )
        show_density_on_one_figure(do_data, None, data_flag, None, title=f'{i + 1}:{p_id}_cut25_99.9')
        # , cumulative=True


def one_time_trans_us_to_origin():
    def _process_and_save_label(src_lab, src_vol, aim_path1, aim_path2):
        s_label = sitk.ReadImage(src_lab)    # LPI
        s_label_data = sitk.GetArrayFromImage(s_label)

        aim_label_data = np.transpose(s_label_data, [0, 2, 1])

        aim_label = sitk.GetImageFromArray(aim_label_data)
        aim_label.CopyInformation(sitk.ReadImage(src_vol))

        sitk.WriteImage(aim_label, aim_path1)
        sitk.WriteImage(aim_label, aim_path2)

    all_volume_path_root=r'L:\DATA\temp_data\Formation__MR-USviaFenster20'
    us_path_root=r'L:\DATA\temp_data\Formation__MR-USviaFenster20_TRUS'

    source_us_root = r'L:\DATA\temp_data\TRUS_label_mat'

    pat_ids = os.listdir(all_volume_path_root)

    for pat_id in pat_ids:
        src_label_path = os.path.join(source_us_root, f'{pat_id}_US_label.nii')
        src_volume_path = os.path.join(us_path_root, pat_id, f'{pat_id}_US.nii')

        save_path1 = os.path.join(us_path_root, pat_id, f'{pat_id}_US_Prostate.nii')
        save_path2 = os.path.join(all_volume_path_root, pat_id, f'{pat_id}_US_Prostate.nii')

        _process_and_save_label(src_label_path, src_volume_path, save_path2, save_path1)

        print(f'{pat_id} finished!')


# shutil.copy=copyfile+copymode
# shutil.copy2=copyfile+copystat
def one_time_copy_datas_to_dirs(src_root=r'L:\DATA\temp_data\Formation__MR-USviaFenster20',
                                aim_root=r'L:\DATA\temp_data\MR-USviaFenster20'):
    pat_ids = os.listdir(src_root)

    for pat_id in pat_ids:
        files = glob(os.path.join(src_root, pat_id, '*.nii'))
        save_dir = os.path.join(aim_root, pat_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print(save_dir)
        for file in files:
            shutil.copy2(file, save_dir)


def main():
    dataroot = r'/home/lf/data_fong/DATA/MR-USvia20'
    saveroot = r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280'
    # dataroot = r'F:\Code\NEW_doing\UMMS\traces\datasets\MR-USviaFenster20-origin'
    # saveroot = r'F:\Code\NEW_doing\UMMS\traces\datasets\MR-USviaFenster20-pre12812896-filter'
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
    process_data(dataroot, saveroot, aim_shape=(112, 112, 80), do_filter=True)


if __name__ == "__main__":
    main()
    # check_all_data_percentile()
    # check_all_data_density()
    # check_all_volume_mask_direction()
    # tt=nib.orientations.axcodes2ornt(('L','P','I'), (('L', 'R'), ('P', 'A'), ('I', 'S')))
    # print(tt)

    # aff = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    # aa = nib.orientations.aff2axcodes(aff, (('L', 'R'), ('P', 'A'), ('I', 'S')))
    # print(aa)
    # one_time_copy_datas_to_dirs()
    pass
