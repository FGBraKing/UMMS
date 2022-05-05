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


def show_density_on_one_figure(data1, data2, label1='MR', label2='US', title=None):
    fig, ax = plt.subplots()
    # fig.suptitle('Overall  intensity value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Overall  intensity value')
    ax.set_ylabel('Frequency')
    sns.kdeplot(data1.flatten(), label=label1, color='red', ax=ax)
    sns.kdeplot(data2.flatten(), label=label2, color='blue', ax=ax)
    # ax.hist(data1.flatten(), density=True, label=label1)
    # ax.hist(data2.flatten(), density=True, label=label2)
    ax.legend()
    if title:
        ax.set_title(title)
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
    data1_path = r'/home/lf/raid_lf/PROJECT/UMMS/traces/datasets/MR-USviaFenster20_pre_std/mr/P070_MR_image.nii'
    data2_path = r'/home/lf/raid_lf/PROJECT/UMMS/traces/datasets/MR-USviaFenster20_pre_std/us/P070_US_image.nii'

    data_path = r'/home/lf/raid_lf/DATA/MR-USviaFenster20/P070/P070_MR.nii'
    data_path1 = r'/home/lf/raid_lf/DATA/MR-USviaFenster20/P070/P070_US.nii'

    data_test = nii_loader(data_path)
    data_test1 = nii_loader(data_path1)

    mr_data = nii_loader(data1_path)
    us_data = nii_loader(data2_path)
    print_numpy(mr_data, shp=True, percentile=True)
    print_numpy(us_data, shp=True, percentile=True)
    print_numpy(data_test, shp=True, percentile=True)

    # sns.distplot(data_test, hist=False, kde=True, kde_kws={'color':'red', 'linestyle':'-'},
    #              fit=stats.norm, fit_kws={'color':'black', 'label':'u=0,s=1','linestyle':'-'})

    # sns.distplot(mr_data, hist=False, kde=True, kde_kws={'color': 'red', 'linestyle': '-'}, label='mr')
    # ax = sns.distplot(us_data, hist=False, kde=True, kde_kws={'color': 'blue', 'linestyle': '-'}, label='us')
    # ax.set_xlabel('vulue')
    # ax.set_ylabel('Data Density')
    # ax.set_title('ttt')
    # ax.legend()
    # # sns.kdeplot(data=tips, x="total_bill", hue="time", multiple="stack")
    # # sns.kdeplot(mr_data[30, :, :], x='Overall  intensity value')
    # # sns.kdeplot(us_data[60, :, :])
    # plt.show()

    # plt.figure()
    # from scipy.stats import kde
    # density = kde.gaussian_kde(us_data.flatten())    # 得到了概率密度函数，density是个函数
    # density.covariance_factor = lambda: 0.1
    # density._compute_covariance()
    #
    # x = np.linspace(-5, 5, 300)
    # y = density(x)
    # plt.plot(x, y)
    # plt.show()

    # sns.kdeplot(x=mr_data.flatten(), label='mr', linestyle='-', color='red')
    # sns.kdeplot(us_data.flatten(), label='us', linestyle='-', color='blue')
    # plt.legend()
    # plt.title('image')
    # plt.xlabel('vulue')
    # plt.ylabel('vulue')
    # plt.show()

    # show_density(us_data)
    show_density_on_one_figure(mr_data, us_data)
