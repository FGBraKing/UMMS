# encoding: utf-8
import os
import sys
import random
import torch
import numpy as np
import warnings
import csv
import time
import datetime
from collections import namedtuple
from types import SimpleNamespace
from functools import singledispatch
from torchvision.utils import make_grid
warnings.filterwarnings('ignore')


def get_device_name():
    device_name = torch.cuda.get_device_name()
    if device_name == 'NVIDIA GeForce GTX 1080 Ti':
        return '1080Ti'
    elif device_name == 'NVIDIA GeForce RTX 2080 Ti':
        return '2080Ti'
    elif device_name == 'NVIDIA TITAN Xp':
        return 'TITAN'
    elif device_name == 'Tesla V100-DGXS-32GB':
        return 'Tesla'
    else:
        return ''


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def init_seed(seed=1008):
    random.seed(seed+1)
    os.environ['PYTHONHASHSEED'] = str(seed+2)
    np.random.seed(seed+3)
    torch.manual_seed(seed+4)
    torch.cuda.manual_seed(seed+5)   # 当前GPU
    torch.cuda.manual_seed_all(seed+6)    # 所有GPU


# if len(opt.gpu_ids) > 0:
#     torch.cuda.set_device(opt.gpu_ids[0])
def init_torch(gpu_id='0', deterministic=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.multiprocessing.set_sharing_strategy('file_system')
    assert torch.cuda.is_available()
    torch.backends.cudnn.enabled = True
    if deterministic:
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist path: ', path)


def dict_to_csv(dct, csv_path):
    with open(csv_path, 'wt') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, dct.keys())
        w.writeheader()
        w.writerow(dct)


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
        percentile_99_5 = np.percentile(x, 99.5)
        percentile_00_5 = np.percentile(x, 00.5)
        print('percentile_99_5 = %5.3f, percentile_00_5 = %5.3f' % (percentile_99_5, percentile_00_5))


def list_to_csv(dct, csv_path):
    with open(csv_path, 'wt') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f)
        w.writerow(dct)


#  timer

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

# from MUNIT


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        loader = yaml.FullLoader(stream)
        out = loader.get_single_data()
        out1 = yaml.load(stream)
        return out


def eformat(f, prec):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d" % (mantissa, int(exp))


#  dict to object and object 2 dict
def dict2obj(d):
    assert isinstance(d, dict)
    return SimpleNamespace(**d)
# def dict2obj(d):
#     # Myobject = namedtuple('MyObject', d.keys())
#     # return Myobject(*d.values())
#     func = lambda xd: namedtuple('MyObject', xd.keys())(*xd.values())
#     return func(d)


def obj2dict(o):
    # return o.__dict__
    return vars(o)


@singledispatch
def dict2obj_all(o):
    return o


@dict2obj_all.register(dict)
def handle_list(obj):
    return SimpleNamespace(**{k: dict2obj_all(v) for k, v in obj.items()})


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# output: NM,N个样本，M个类别
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)   # N*maxk
        pred = pred.t()     # maxk*N
        correct = pred.eq(target.view(1, -1).expand_as(pred))   # 1*N
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def human_readable_size(size, decimal_places=1):
    for unit in ['', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


# 截断
def clip_array(array, rate=0.99, bins=256, side_bin=False):
    assert isinstance(array, np.ndarray), 'array must be ndarray'
    assert 0 <= rate <= 1
    shape = array.shape
    return_array = array.copy()
    num_all = array.size
    hist, bin_edges = np.histogram(array.ravel(), bins=bins, range=[np.min(array), np.max(array)], density=False)
    one_side_rate = (1-rate) / 2
    if one_side_rate == 0:
        return return_array
    elif one_side_rate == 1:
        return None
    prob = hist / num_all
    comu_prob = np.cumsum(prob)
    low_bin = bin_edges[0]
    high_bin = bin_edges[-1]
    for i in range(1, len(comu_prob)-1):
        if comu_prob[i] > one_side_rate >= comu_prob[i-1]:
            low_bin = bin_edges[i-1]
        if comu_prob[i] < (rate + one_side_rate) <= comu_prob[i+1]:
            high_bin = bin_edges[i+1]
    return_array = np.clip(array, low_bin, high_bin)
    # return_array[array < low_bin] = low_bin
    # return_array[array > high_bin] = high_bin
    # if True:
    #     print('low:{}, high:{}, zero:{}'.format(np.sum(array < low_bin),
    #                                             np.sum(array > high_bin), np.sum(array == 0)))
    if side_bin:
        return (low_bin, high_bin), return_array
    else:
        return return_array


def cut_off_outliers(data, percentile_lower=0.2, percentile_upper=99.8, per_channel=False):
    # data: CDHW
    if not per_channel:
        cut_off_lower = np.percentile(data, percentile_lower)
        cut_off_upper = np.percentile(data, percentile_upper)
        data[data < cut_off_lower] = cut_off_lower
        data[data > cut_off_upper] = cut_off_upper
    else:
        for c in range(data.shape[0]):
            cut_off_lower = np.percentile(data[c], percentile_lower)
            cut_off_upper = np.percentile(data[c], percentile_upper)
            data[c][data[c] < cut_off_lower] = cut_off_lower
            data[c][data[c] > cut_off_upper] = cut_off_upper

    return data


def image_histogram_equalization(image, number_bins=256, with_cdf=False):
    # get image histogram
    hist, bins_edges = np.histogram(image.flatten(), number_bins)
    cdf = hist.cumsum()  # cumulative distribution function
    # cdf_normalized = cdf / cdf.max()  # normalize
    # image_equalized = cdf_normalized[image] * number_bins

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * number_bins / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype(np.uint16)
    image_equalized = cdf[image]

    # image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    # from skimage import exposure
    # res = exposure.equalize_hist(image)
    # import cv2
    # res1 = cv2.equalizeHist(image)
    if with_cdf:
        return image_equalized.reshape(image.shape), cdf
    else:
        return image_equalized.reshape(image.shape)


def convert_str_to_list(str_seq, split=',', aim_type=int, condition=None):
    assert isinstance(str_seq, str)
    str_datas = str_seq.split(split)
    if condition:
        return [aim_type(str_data) for str_data in str_datas if condition(aim_type(str_data))]
    else:
        return [aim_type(str_data) for str_data in str_datas]


def get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


# 去黑边
def slim_array(array, dims=None, number=100):
    # print('before slim: {}'.format(array.shape))
    array = np.array(array)
    ndim = array.ndim
    return_array = np.copy(array)
    ind_non_zero = array != 0
    for dim in range(ndim):
        if dims is not None:
            if dim not in dims:
                continue
        del_list = []
        tmp_non_zero = np.swapaxes(ind_non_zero, 0, dim)
        for i in range(tmp_non_zero.shape[0]):
            if np.sum(tmp_non_zero[i, ...]) < number:
                del_list.append(i)
        # print(del_list)
        return_array = np.delete(return_array, del_list, axis=dim)
    # print('after slim: {}'.format(return_array.shape))
    return return_array


# 获取前景大小
def get_foreground_shape(mask, dims=None, number=50, background_value=0):
    '''
    :param mask:
    :param dims: 只计算特定dim的shape， dims是元组
    :param number: 最少多少个非零才算前景
    :param background_value:　背景的值
    :return:
    '''
    dims_shape = []

    mask = np.array(mask)
    ndim = mask.ndim
    ind_non_zero = mask != background_value
    for dim in range(ndim):
        if dims is not None:
            if dim not in dims:
                continue
        dim_num = ind_non_zero.shape[dim]   # 第dim维大小
        non_zero_num = np.sum(np.reshape(np.swapaxes(ind_non_zero, 0, dim), (dim_num, -1)), -1)     # 每一行的前景数

        total = non_zero_num.size
        start = 0
        end = total
        for i in range(total):
            if start > end:
                break
            else:
                if non_zero_num[i] < number and i == start:
                    start += 1
                if non_zero_num[-1-i] < number and total-i == end:
                    end -= 1
        # print((start, end))
        dims_shape.append((start, end-1))
    return dims_shape


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def get_gauusian_kernel(shape):
    if len(shape) == 2:
        X, Y = shape
        target = np.zeros(shape, dtype=np.float32)
        for x in range(X):
            x_i = x if x <= (X - 1)/2 else (X - 1) - x
            for y in range(Y):
                y_i = y if y <= (Y - 1)/2 else (Y - 1) - y
                target[x, y] = x_i + y_i
        # t_max = (X - 1)//2 + (Y - 1)//2
        # assert t_max == target.max()
        # target = target / t_max
        return target


def get_gauusian_kernel_v2(shape):
    dims = len(shape)
    target = np.zeros(shape, dtype=np.float32)
    # assert dims in (1, 2, 3, 4)
    for axis in range(dims):
        shp = shape[axis]

        target_tmp = target if axis == 0 else np.swapaxes(target, 0, axis)

        for i in range(shp):
            val = i if i <= (shp - 1)/2 else (shp - 1) - i
            target_tmp[i, ...] += val

        target = target_tmp if axis == 0 else np.swapaxes(target, 0, axis)
    return target


# 用于训练时存储测试结果。存储最好的n个指标，以及数据的原始索引
class DataPool(object):
    def __init__(self, poolsize, min_data=0.5):
        self.poolsize = poolsize
        self.pure_data = set()
        self.complete_data = []
        self.init_min_data = min_data

    def update(self, ind, data):
        if len(self.pure_data) < self.poolsize:
            if data > self.init_min_data:
                self.pure_data.add(data)
                self.complete_data.append({'ind': ind, 'data': data})
                return True
            else:
                return False
        else:
            min_data = min(self.pure_data)
            if data < min_data:
                return False
            elif data == min_data:
                self.pure_data.add(data)
                self.complete_data.append({'ind': ind, 'data': data})
                return True
            else:
                self.pure_data.add(data)
                self.complete_data.append({'ind': ind, 'data': data})
                self.pure_data.remove(min_data)
                self.remove_data(min_data)
                return True

    def remove_data(self, min_data):
        if min_data in self.pure_data:
            self.pure_data.remove(min_data)
        for item in self.complete_data:
            if item['data'] == min_data:
                self.complete_data.remove(item)

    def get_pure_data(self):
        return tuple(self.pure_data)

    def get_complete_data(self):
        return self.complete_data

    def get_best_data(self, last=True):
        max_data = max(self.pure_data)
        max_item = []
        for item in self.complete_data:
            if item['data'] == max_data:
                max_item.append(item)
        if last:
            return max_item[-1]['ind'], max_item[-1]['data']
        else:
            return [a for a in map(lambda x:x['ind'], max_item)], max_item[-1]['data']

    def reset(self):
        self.pure_data.clear()
        self.complete_data.clear()


# --------------------------from https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/cli.py

def cast_list(el):
    return el if isinstance(el, list) else [el]


def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'
