import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import torch
import SimpleITK as sitk
from data.utils_data import nii_loader, save_nii
from utils.others.img_io import show_image_label,show_image
import torch.nn as nn
from torch.nn import functional as F
from scipy.ndimage.filters import sobel, convolve
from skimage.filters.edges import sobel
from skimage.feature import canny
from skimage.morphology.binary import binary_erosion, binary_closing, binary_dilation, binary_opening


def get_edge(volume):
    # 使用canny比较麻烦，需要高斯滤波平滑+sobel梯度检测+NMS。在这个简单的二值图上，倒不如直接遍历判断
    new = copy.deepcopy(volume)
    dep, row, col = volume.shape
    for i in range(row-1):
        for j in range(col-1):
            for k in range(dep-1):
                if (i-1 > 0 and i+1 < row-1) and (j-1 > 0 and j+1 < col-1) and (k-1 > 0 and k+1 < dep-1):
                    if volume[k][i][j-1] and volume[k][i][j+1] and volume[k][i-1][j] and volume[k][i+1][j] and volume[k-1][i][j] and volume[k+1][i][j]:
                        new[k][i][j] = 0
    return new


def get_edge_v1(volume):
    # 六连通
    kernel_1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    kernel_2 = np.array([[0,1,0],[1,0,1],[0,1,0]])
    kernel_3 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    kernel = np.stack((kernel_1,kernel_2,kernel_3), axis=0)

    conv_result = convolve(volume, kernel)
    conv_max = conv_result.max()
    conv_result[conv_result == conv_max]=0
    conv_result[conv_result>0]=1
    conv_result1 = conv_result*volume   # 去除一些伪边缘
    return conv_result1


def get_edge_v2(volume):
    # 全连通
    kernel_1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    kernel_2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    kernel_3 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    kernel = np.stack((kernel_1,kernel_2,kernel_3), axis=0)

    conv_result = convolve(volume, kernel)
    conv_max = conv_result.max()
    conv_result[conv_result == conv_max]=0
    conv_result[conv_result>0]=1
    conv_result1 = conv_result*volume   # 去除一些伪边缘
    return conv_result1


def get_distance_map(array):
    depth, height, width = array.shape
    dismap = np.zeros_like(array)
    if np.sum(array):
        edge = get_edge_v1(array.astype('uint8'))
        edgez, edgey, edgex = np.nonzero(edge)      # or np.where(test_edge==1)
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    dismap[d][h][w] = np.min(np.sqrt(np.square(edgez-d)+np.square(edgey-h)+np.square(edgex-w)))
    else:
        dismap[dismap == 0] = 1000
    max_distance = (112**2+112**2+80**2)**0.5
    return dismap/max_distance


def patch_create_dm():
    dataroot = '/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280'
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))
    us_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'roi'))
        }
        for p_id in pat_ids
    ]

    mr_paths = [
        {
            'volume': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'volume')),
            'label': os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'roi'))
        }
        for p_id in pat_ids
    ]

    for i in range(len(pat_ids)):
        print(pat_ids[i], os.path.join(dataroot, pat_ids[i], "{}_{}_{}.nii".format(pat_ids[i], 'mr', 'roi')), mr_paths[i]['label'])
        mr_label_path = mr_paths[i]['label']
        us_label_path = us_paths[i]['label']
        mr_label = nii_loader(mr_label_path)
        us_label = nii_loader(us_label_path)
        mr_spacing = sitk.ReadImage(mr_label_path).GetSpacing()
        us_spacing = sitk.ReadImage(mr_label_path).GetSpacing()

        mr_dm = get_distance_map(mr_label)
        us_dm = get_distance_map(us_label)

        mr_dm_savepath = os.path.join(dataroot, pat_ids[i], "{}_{}_{}.nii".format(pat_ids[i], 'mr', 'dm'))
        us_dm_savepath = os.path.join(dataroot, pat_ids[i], "{}_{}_{}.nii".format(pat_ids[i], 'us', 'dm'))
        save_nii(us_dm_savepath, us_dm, spacing=us_spacing,
                 direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        save_nii(mr_dm_savepath, mr_dm, spacing=mr_spacing,
                 direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        show_image(mr_dm[40], num=i, cmap='hot', title='distance map')


if __name__ == "__main__":
    patch_create_dm()

    # test_maskpath = r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280/P068/P068_mr_roi.nii'
    # test_maskdata = nii_loader(test_maskpath)
    # test_edge = get_edge(test_maskdata)
    # show_image_label(test_maskdata[40], test_edge[40])
    # dm = get_distance_map(test_maskdata)
    # show_image(dm[40], cmap='hot')




