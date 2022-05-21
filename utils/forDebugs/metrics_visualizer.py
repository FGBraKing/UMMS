import re
import os
# import time
import math
from glob import glob
from matplotlib import pyplot as plt
import pandas as pd
# import numpy as np
# import seaborn as sns
# from utils.others.img_io import plot_2d
# from .img_io import plot_2d


def extract_metrics(resolving_file, dividually=False, interval=1, title=''):
    # visual_names=('DC', 'recall', 'precision', 'accuracy'),
    vaild_pat = re.compile(r'^\((.*)\)\s*?(.*)\s*$')
    dict_pat = re.compile(r'(\w+):\s*([+-]?\d+(?:\.\d+)?)')

    info_dict_list = []
    meta_keys = None
    metrics_keys = None
    with open(resolving_file, 'r') as f_metrics:
        for line in f_metrics.readlines():
            vaild_match = vaild_pat.match(line)
            if vaild_match is not None:
                if meta_keys is None and metrics_keys is None:
                    meta_str = vaild_match.groups()[0]
                    metrics_str = vaild_match.groups()[1]
                    meta_keys = list(dict(dict_pat.findall(meta_str)).keys())
                    metrics_keys = list(dict(dict_pat.findall(metrics_str)).keys())
                info_dict_list.append(dict(dict_pat.findall(line)))
    print('meta_keys: {}\nmetrics_keys: {}'.format(meta_keys, metrics_keys))
    info_df = pd.DataFrame(data=info_dict_list, dtype=float)
    info_df[meta_keys] = info_df[meta_keys].astype(int)

    # print(info_df.info())
    print(info_df.describe())
    plt.figure()
    info_df[metrics_keys].plot()
    plt.title('metrics all', fontsize=14)
    plt.show()
    if dividually:
        for key in metrics_keys:
            plot_2d(range(math.ceil(len(info_df[key])/interval)), info_df[key][::interval],
                    label=key, fig_title='Metrics dividually '+title)
    return info_df


def plot_2d(x, y, *args, fig_title=None, ax_title=None, x_label=None, y_label=None, **kwargs):
    fig, ax = plt.subplots()
    if fig_title:
        fig.suptitle(fig_title, fontsize=14)    # , fontweight='bold'
    if ax_title:
        ax.set_title(ax_title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.plot(x, y, *args, **kwargs)
    ax.legend()
    fig.show()
    plt.close(fig)


def one_time_repair_metrics_file(file_name):
    from utils.others.utils import DataPool
    data_pat = re.compile(r'^.*epoch:\s*?(\d+).*-\d{1,2}.*?DC:\s*?(0\.\d+).*$')

    datapool = DataPool(3, 0.80)
    with open(file_name, 'r+') as f:
        for line in f.readlines():
            data = data_pat.match(line)
            if data is not None:
                epoch_str = data.groups()[0]
                dc_str = data.groups()[1]
                # print(epoch_str, dc_str)
                datapool.update(int(epoch_str), float(dc_str))
        f.write('\n')
        f.writelines(repr(datapool.get_complete_data()))
    print(repr(datapool.get_complete_data()))


def main():
    logs_dir = r'/home/lf/raid_lf/PROJECT/UMMS/traces/logs'
    exp_name = r'mrusmr128_fold0_patch_bs8_unet3d_ch16_combo_1_1_1.5_adam_2e-4_poly_3x300_0.6'

    metrics_file = os.path.join(logs_dir, exp_name, r'metrics.txt')
    test_metrics_file = os.path.join(logs_dir, exp_name, r'test_metrics.txt')
    slide_metrics_file = os.path.join(logs_dir, exp_name, r'slide_test_metrics.txt')

    data_df = extract_metrics(metrics_file, dividually=True, interval=3, title='train data')

    test_data_df = extract_metrics(test_metrics_file, dividually=True, interval=3, title='test data')

    slide_data_df = extract_metrics(slide_metrics_file, dividually=True, interval=3, title='slide data')


if __name__ == '__main__':
    # main()
    testfile = r'/home/users/lf/data_lf/PROJECT/UMMS/traces/logs/mrusmr128_fold0_patch_bs8_unet3d_ch16_combo_1_1_2_l2_adam_2e-4_poly_2x300_0.6_1080Ti/slide_test_metrics.txt'
    one_time_repair_metrics_file(testfile)
