import os
import cv2
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from skimage import measure
from data.utils_data import h5_loader, nii_loader
from utils.others.utils import print_numpy
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict, show_image
from data.transforms.transformOnArray import normalize, NormalizeRange

# plt.style.use('seaborn')


def show_density_on_one_figure(data1, data2):
    fig, ax = plt.subplots()
    # fig.suptitle('Overall  intensity value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Overall  intensity value')
    ax.set_ylabel('Frequency')
    ax.hist(data1.flatten(), density=True, label='MR')
    ax.hist(data2.flatten(), density=True, label='US')
    ax.legend()
    fig.show()


def show_density(data):
    fig, ax = plt.subplots()
    # fig.suptitle('Overall  intensity value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Overall  intensity value')
    ax.set_ylabel('Frequency')
    ax.hist(data.flatten(), bins=100, density=True, stacked=True, label='data')
    ax.legend()
    fig.show()


if __name__=="__main__":
    data1_path=r'/home/lf/data_fong/CODE/PycharmProject/UMMS/traces/datasets/MR-USviaFenster20_pre/mrtrain/P070_MR_image.nii'
    data2_path=r'/home/lf/data_fong/CODE/PycharmProject/UMMS/traces/datasets/MR-USviaFenster20_pre/ustrain/P070_US_image.nii'

    data_path = r'/home/lf/data_fong/DATA/MR-USviaFenster20/P070/P070_MR.nii'
    data_path1 = r'/home/lf/data_fong/DATA/MR-USviaFenster20/P070/P070_US.nii'

    data_test = nii_loader(data_path)
    data_test1 = nii_loader(data_path1)

    mr_data = nii_loader(data1_path)
    us_data = nii_loader(data2_path)
    print_numpy(mr_data, shp=True, percentile=True)
    print_numpy(us_data, shp=True, percentile=True)
    print_numpy(data_test, shp=True, percentile=True)

    # sns.distplot(data_test, hist=False, kde=True, kde_kws={'color':'red', 'linestyle':'-'},
    #              fit=stats.norm, fit_kws={'color':'black', 'label':'u=0,s=1','linestyle':'-'})
    sns.distplot(mr_data[30, :, :], hist=False, kde=True, kde_kws={'color': 'red', 'linestyle': '-'})
    sns.distplot(us_data[60, :, :], hist=False, kde=True, kde_kws={'color': 'blue', 'linestyle': '-'})
    # sns.kdeplot(data=tips, x="total_bill", hue="time", multiple="stack")
    # sns.kdeplot(mr_data[30, :, :], x='Overall  intensity value')
    # sns.kdeplot(us_data[60, :, :])
    plt.show()

    # show_density(us_data)
    show_density_on_one_figure(mr_data[30, :, :], us_data[60, :, :])
