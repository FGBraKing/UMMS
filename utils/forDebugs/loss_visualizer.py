import re
import os
# import time
import math
# from glob import glob
import numpy as np

from matplotlib import pyplot as plt
# import pandas as pd
# import seaborn as sns
# from utils.others.img_io import plot_2d
# from .img_io import plot_2d


def extract_loss(loss_file, pat=r'^\(epoch.*\).*?(({}):\s+?(\d+\.\d+))', loss_name='dice', interval=1):
    pat = re.compile(pat.format(loss_name))
    loss_list = []
    with open(loss_file, 'r') as f_loss:
        for line in f_loss.readlines():
            match_result = pat.match(line)
            if match_result is not None:
                # print(match_result.groups())
                loss_list.append(float(match_result.groups()[-1]))
    print_numpy(np.array(loss_list), shp=True, percentile=True)
    print_numpy(np.array(loss_list[100:]), shp=True, percentile=True)
    batchs = math.ceil(len(loss_list) / interval)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(batchs), loss_list[::interval])
    ax.set_title('loss curve')
    # ax.axis('off')
    ax.set(xlabel='batch', ylabel=loss_name)
    plt.show()


def print_numpy(x, val=True, shp=False, percentile=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
        percentile (bool) -- if print the percentile of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
    if percentile:
        x = x.flatten()
        print(np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75))
        percentile_99_5 = np.percentile(x, 99.5)
        percentile_00_5 = np.percentile(x, 00.5)
        print('percentile_99_5 = %5.3f, percentile_00_5 = %5.3f' % (percentile_99_5, percentile_00_5))


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


if __name__ == '__main__':
    logsdir = r'/home/lf/data_fong/CODE/PycharmProject/DLForPytorch/traces/logs'
    exp_name = r'mrusmr_unet3dV1_969632_bs6_ch32_kaiming_combo_0_1.0_adam_2e-4_cosine_1.0_0.3_2x500_warmup_10_1e-5'
    loss_name = r'loss_log.txt'

    loss_file = os.path.join(logsdir, exp_name, loss_name)
    extract_loss(loss_file, loss_name='seg', interval=5)



