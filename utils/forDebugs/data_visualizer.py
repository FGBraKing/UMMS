import os
import cv2
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from skimage import measure
from data.utils_data import h5_loader, nii_loader
from utils.others.utils import print_numpy, cut_off_outliers
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict, show_image
from data.transforms.transformOnArray import normalize, NormalizeRange

# plt.style.use('seaborn')


def show_density_on_one_figure(data1, data2=None, label1='MR', label2='US', title=None, **kwargs):
    fig, ax = plt.subplots()
    # fig.suptitle('Overall  intensity value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Overall  intensity value')
    ax.set_ylabel('Frequency')
    # vertical=True shade=True  cumulative=True

    sns.kdeplot(data1.flatten(), label=label1, color='red', ax=ax, **kwargs)
    if data2 is not None:
        sns.kdeplot(data2.flatten(), label=label2, color='blue', ax=ax, **kwargs)
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


def check_all_data_density(dataroot):
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))

    all_patients = np.zeros((2, 20, 80, 112, 112), dtype=np.float)
    for i, p_id in enumerate(pat_ids):
        us_path = os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume'))
        mr_path = os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'volume'))

        mr_data = nii_loader(mr_path)
        us_data = nii_loader(us_path)

        # print('{:*^60}'.format(p_id))
        # print_numpy(us_data, shp=True, percentile=True)
        # print(np.percentile(us_data, 25))
        # # us_data = cut_off_outliers(us_data, 25, 99.9, per_channel=False)
        # print_numpy(us_data, shp=True, percentile=True)
        show_density_on_one_figure(us_data, mr_data, 'us', 'mr', title=f'{i + 1}:{p_id}')
        # , cumulative=True
        all_patients[0, i, ...] = mr_data
        all_patients[1, i, ...] = us_data
    show_density_on_one_figure(all_patients[1], all_patients[0], 'us', 'mr', title=f'all patients')
    # , cumulative=True


def check_all_data_percentile(dataroot, percentile=20):
    all_patients = np.zeros((20, 80, 112, 112), dtype=np.float)
    pat_ids = list(filter(lambda a: os.path.isdir(os.path.join(dataroot, a)), os.listdir(dataroot)))
    for i, p_id in enumerate(pat_ids):
        us_path = os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'us', 'volume'))
        mr_path = os.path.join(dataroot, p_id, "{}_{}_{}.nii".format(p_id, 'mr', 'volume'))

        # mr_data = nii_loader(mr_path)
        us_data = nii_loader(us_path)
        print('{:*^60}'.format(p_id))
        print_numpy(us_data, shp=True, percentile=True)
        print(np.percentile(us_data, percentile))
        all_patients[i] = us_data
    print('{:*^60}'.format('all patients'))
    print(np.percentile(all_patients, percentile))


if __name__ == "__main__":
    #
    # r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USviaFenster20-pre12812896'
    # data_root=r'/home/users/lf/data_lf/PROJECT/UMMS/traces/datasets/MR-USviaFenster20-pre12812896-filter'
    data_root = r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USvia20-full-11211280'
    # data_root=r'F:\Code\NEW_doing\UMMS\traces\datasets\MR-USviaFenster20-pre12812896-filter'

    check_all_data_density(data_root)
    # data1_path=r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USviaFenster20-pre12812896/P108/P108_mr_volume.nii'
    # data2_path=r'/home/lf/data_fong/PROJECT/UMMS/traces/datasets/MR-USviaFenster20-pre12812896/P108/P108_us_volume.nii'
    check_all_data_percentile(data_root, 25)
    #
    # data_path  = r'/home/lf/data_fong/DATA/MR-USviaFenster20/P108/P108_MR_Prostate.nii'
    # data_path1 = r'/home/lf/data_fong/DATA/MR-USviaFenster20/P108/P108_US_Prostate.nii'
    # #
    # data_test = nii_loader(data_path)
    # data_test1 = nii_loader(data_path1)
    # #
    # # mr_data = nii_loader(data1_path)
    # # us_data = nii_loader(data2_path)
    # #
    # # print_numpy(mr_data, shp=True, percentile=True)
    # # print_numpy(us_data, shp=True, percentile=True)
    # print_numpy(data_test, shp=True, percentile=True)
    # print_numpy(data_test1, shp=True, percentile=True)
    # print(np.percentile(data_test1, 20))
    # print('*'*50)
    # print_numpy(us_data, shp=True, percentile=True)
    # print(np.percentile(us_data, 20))
    # us_data = cut_off_outliers(us_data, 20, 99.9, per_channel=False)
    # print_numpy(us_data, shp=True, percentile=True)
    #
    # # sns.distplot(data_test, hist=False, kde=True, kde_kws={'color':'red', 'linestyle':'-'},
    # #              fit=stats.norm, fit_kws={'color':'black', 'label':'u=0,s=1','linestyle':'-'})
    #
    # # sns.distplot(mr_data, hist=False, kde=True, kde_kws={'color': 'red', 'linestyle': '-'}, label='mr')
    # # ax = sns.distplot(us_data, hist=False, kde=True, kde_kws={'color': 'blue', 'linestyle': '-'}, label='us')
    # # ax.set_xlabel('vulue')
    # # ax.set_ylabel('Data Density')
    # # ax.set_title('ttt')
    # # ax.legend()
    # # # sns.kdeplot(data=tips, x="total_bill", hue="time", multiple="stack")
    # # # sns.kdeplot(mr_data[30, :, :], x='Overall  intensity value')
    # # # sns.kdeplot(us_data[60, :, :])
    # # plt.show()
    #
    # # plt.figure()
    # # from scipy.stats import kde
    # # density = kde.gaussian_kde(us_data.flatten())    # 得到了概率密度函数，density是个函数
    # # density.covariance_factor = lambda: 0.1
    # # density._compute_covariance()
    # #
    # # x = np.linspace(-5, 5, 300)
    # # y = density(x)
    # # plt.plot(x, y)
    # # plt.show()
    #
    # # sns.kdeplot(x=mr_data.flatten(), label='mr', linestyle='-', color='red')
    # # sns.kdeplot(us_data.flatten(), label='us', linestyle='-', color='blue')
    # # plt.legend()
    # # plt.title('image')
    # # plt.xlabel('vulue')
    # # plt.ylabel('vulue')
    # # plt.show()
    #
    # # show_density(us_data)
    # show_density_on_one_figure(us_data, mr_data, label1='US', label2='MR', title='P108')
    # cumulative=True
