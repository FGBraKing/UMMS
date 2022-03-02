# encoding: utf-8
import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from itertools import combinations
from scipy.signal.signaltools import convolve
from scipy.ndimage.interpolation import map_coordinates, zoom, rotate, shift, affine_transform
from scipy.ndimage.filters import gaussian_filter, convolve, median_filter
from skimage.transform import resize, rescale
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from batchgenerators.augmentations.spatial_transformations import augment_channel_translation
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from batchgenerators.augmentations.noise_augmentations import augment_blank_square_noise, augment_gaussian_blur, \
    augment_gaussian_noise, augment_rician_noise
from batchgenerators.augmentations.utils import get_range_val
from typing import Union, Tuple, Callable, List

from data.transforms.transformOnSample import ResizeTransform, ZoomTransform, RandomScaleTransform, \
    RandomCropTransform, CenterCropTransform, Rot90Transform, RandomRotateTransform, \
    MirrorTransform, TransposeAxesTransform, RandomShiftTransform, ElasticDeformTransform
from data.transforms.transforms import Compose, ComposeForSample


from data.transforms.transformOnSample import RandomFlip, RandomCrop, RandomRotate, CenterCrop, \
    RandomRotate90, RandomScale, resize_3d


# ----------------------------------------------function----------------------------------------------
# --------------------------------custom
def standardize(array, mean, std, eps=1e-7):
    return (array - mean) / (std + eps)


def normalize(array, min_value=0., max_value=1.):
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + 1e-6)
    return (max_value - min_value) * normalized + min_value


# --------------------------------copy from internet
# Function to distort image  alpha = im_merge.shape[1]*2、sigma=im_merge.shape[1]*0.08、alpha_affine=sigma
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]   #(512,512)表示图像的尺寸
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1为变换前的坐标，pts2为变换后的坐标，范围为什么是center_square+-square_size？
    # 其中center_square是图像的中心，square_size=512//3=170
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # Mat getAffineTransform(InputArray src, InputArray dst)  src表示输入的三个点，dst表示输出的三个点，获取变换矩阵M
    M = cv2.getAffineTransform(pts1, pts2)  #获取变换矩阵
    #默认使用 双线性插值，
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # # random_state.rand(*shape) 会产生一个和 shape 一样打的服从[0,1]均匀分布的矩阵
    # * 2 - 1 是为了将分布平移到 [-1, 1] 的区间
    # 对random_state.rand(*shape)做高斯卷积，没有对图像做高斯卷积，为什么？因为论文上这样操作的
    # 高斯卷积原理可参考：https://blog.csdn.net/sunmc1204953974/article/details/50634652
    # 实际上 dx 和 dy 就是在计算论文中弹性变换的那三步：产生一个随机的位移，将卷积核作用在上面，用 alpha 决定尺度的大小
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)  #构造一个尺寸与dx相同的O矩阵
    # np.meshgrid 生成网格点坐标矩阵，并在生成的网格点坐标矩阵上加上刚刚的到的dx dy
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))  #网格采样点函数
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def hist_match(source, template):
    # from https://github.com/faustomilletari/VNet/blob/master/utilities.py
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    # interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


# -----------------------------------------copy from reg_3d---------------------------------------------------

def intensity_shift3d(volumes, factor):
    """channel first"""
    n = len(volumes)
    if isinstance(factor, int):
        factor = [factor] * n
    trans_volumes = []
    for i, dx in enumerate(factor):
        volume = volumes[i] + dx
        volume = np.clip(volume, 0, 1.0)
        trans_volumes.append(volume)

    return trans_volumes


def intensity_shift3d1(volumes, factor):
    """channel first"""
    volume = volumes + factor
    volume = np.clip(volume, 0, 1.0)

    return volume


# ----------------------------------------- CUSTOM TRANSFORM ------------------------------------------------------
class ChannelTranslation:
    """Simulates badly aligned color channels/modalities by shifting them against each other

    Args:
        const_channel: Which color channel is constant? The others are shifted

        max_shifts (dict {'x':2, 'y':2, 'z':2}): How many pixels should be shifted for each channel?

    """

    def __init__(self, const_channel=0, max_shifts=None, with_channel=False):
        self.max_shift = max_shifts
        self.const_channel = const_channel
        self.with_channel = with_channel

    def __call__(self, data, *args, **kwargs):
        ret_val = augment_channel_translation(data=data, const_channel=self.const_channel, max_shifts=self.max_shift)

        data = ret_val[0]

        return data


class NormalizeRange:
    """Normalize a ndarray  with max-min normalize.
    .. note::
        None

    Args:
        None

    """

    def __init__(self, min_value=0., max_value=1., dtype=np.float32):
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = dtype

    def __call__(self, array_in):
        """
        Args:
            array (array): array image of size (H, W, C) to be normalized.

        Returns:
            array: Normalized array image.
        """
        arr_min = np.min(array_in)
        arr_max = np.max(array_in)
        normalized = (array_in - arr_min) / (arr_max - arr_min + 1e-10)
        return ((self.max_value - self.min_value) * normalized + self.min_value).astype(self.dtype)

    def __repr__(self):
        return self.__class__.__name__ + '(min_value={0}, max_value={1})'.format(self.max_value, self.min_value)


class RandomGaussianNoise:
    def __init__(self, random_state, sigma_range=(0., 0.1), alpha_range=0., execution_probability=0.15, ):
        self.random_state = random_state
        if isinstance(alpha_range, (int, float)):
            self.alpha_min = self.alpha_max = alpha_range
        elif isinstance(alpha_range, (list, tuple)):
            self.alpha_min, self.alpha_max = alpha_range
        else:
            raise Exception('the type of alpha_range must be one of (int, float, list, tuple)')
        if isinstance(sigma_range, (int, float)):
            self.sigma_min = self.sigma_max = sigma_range
        elif isinstance(sigma_range, (list, tuple)):
            self.sigma_min, self.sigma_max = sigma_range
        else:
            raise Exception('the type of sigma_range must be one of (int, float, list, tuple)')
        self.execution_probability = execution_probability

    def __call__(self, m, *args, **kwargs):
        if self.random_state.random() < self.execution_probability:  # random()/sample()/rand()/random_sample()
            alpha = self.random_state.uniform(self.alpha_min, self.alpha_max, 1)
            sigma = self.random_state.uniform(self.sigma_min, self.sigma_max, 1)
            # noise = self.random_state.standard_normal(m.shape) * sigma + alpha
            noise = self.random_state.normal(loc=alpha, scale=sigma, size=m.shape)
            # np.clip(m+noise, 0, 1)
            return m+noise
        else:
            return m


class CutOffOutliersTransform:
    """ Removes outliers from data

    Args:
        percentile_lower (float between 0 and 100): Lower cutoff percentile

        percentile_upper (float between 0 and 100): Upper cutoff percentile

        per_channel (bool): determines whether percentiles are computed for each color channel separately
    """

    def __init__(self, percentile_lower=0.2, percentile_upper=99.8, per_channel=False, with_channel=False):
        self.per_channel = per_channel
        self.percentile_upper = percentile_upper
        self.percentile_lower = percentile_lower
        self.with_channel = with_channel

    def __call__(self, m):
        if self.with_channel and self.per_channel:
            for c in range(len(m)):
                cut_off_lower = np.percentile(m[c], self.percentile_lower)
                cut_off_upper = np.percentile(m[c], self.percentile_upper)
                m[c][m[c] < cut_off_lower] = cut_off_lower
                m[c][m[c] > cut_off_upper] = cut_off_upper
        else:
            cut_off_lower = np.percentile(m, self.percentile_lower)
            cut_off_upper = np.percentile(m, self.percentile_upper)
            m[m < cut_off_lower] = cut_off_lower
            m[m > cut_off_upper] = cut_off_upper

        return m


class ZeroMeanUnitVarianceTransform:
    """ Zero mean unit variance transform

    Args:
        per_channel (bool): determines whether mean and std are computed for and applied to each color channel
        separately

        epsilon (float): prevent nan if std is zero, keep at 1e-7
    """

    def __init__(self, per_channel=True, epsilon=1e-7, with_channel=False):
        self.epsilon = epsilon
        self.per_channel = per_channel
        self.with_channel = with_channel

    def __call__(self, m):
        data_normalized = np.zeros(m.shape, dtype=m.dtype)
        if self.with_channel and self.per_channel:
            for c in range(len(m)):
                mean = m[c].mean()
                std = m[c].std() + self.epsilon
                data_normalized[c] = (m[c] - mean) / std
        else:
            mean = m.mean()
            std = m.std() + self.epsilon
            data_normalized = (m - mean) / std

        return data_normalized


class MeanStdNormalizationTransform:
    """ Zero mean unit variance transform

    Args:
        per_channel (bool): determines whether mean and std are computed for and applied to each color channel
        separately

        epsilon (float): prevent nan if std is zero, keep at 1e-7
    """

    def __init__(self, mean, std, per_channel=True, with_channel=False):
        self.std = std
        self.mean = mean
        self.per_channel = per_channel
        self.with_channel = with_channel

    def __call__(self, data):
        data_normalized = np.zeros(data.shape, dtype=data.dtype)

        if isinstance(data, np.ndarray):
            data_shape = tuple(list(data.shape))
        elif isinstance(data, (list, tuple)):
            assert len(data) > 0 and isinstance(data[0], np.ndarray)
            data_shape = [len(data)] + list(data[0].shape)
        else:
            raise TypeError("Data has to be either a numpy array or a list")

        if self.per_channel and self.with_channel:
            if isinstance(self.mean, float) and isinstance(self.std, float):
                self.mean = [self.mean] * data_shape[0]
                self.std = [self.std] * data_shape[0]
            elif isinstance(self.mean, (tuple, list, np.ndarray)):
                assert len(self.mean) == data_shape[0]
            elif isinstance(self.std, (tuple, list, np.ndarray)):
                assert len(self.std) == data_shape[0]

            for c in range(data_shape[0]):
                data_normalized[c] = (data[c] - self.mean[c]) / self.std[c]
        else:
            data_normalized = (data - self.mean) / self.std
        return data_normalized


# ------------------------------------- resample transforms
class SimulateLowResolutionTransform:
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1.,
                 channels=None, order_downsample=1, order_upsample=0, p_per_sample=1.,
                 ignore_axes=None, with_channel=False):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes
        self.with_channel = with_channel

    def __call__(self, m):
        if np.random.uniform() < self.p_per_sample:
            if not self.with_channel:
                m = np.expand_dims(m, axis=0)
            m = augment_linear_downsampling_scipy(data_sample=m,
                                                  zoom_range=self.zoom_range,
                                                  per_channel=self.per_channel,
                                                  p_per_channel=self.p_per_channel,
                                                  channels=self.channels,
                                                  order_downsample=self.order_downsample,
                                                  order_upsample=self.order_upsample,
                                                  ignore_axes=self.ignore_axes)
            if not self.with_channel:
                m = np.squeeze(m, axis=0)
        return m


# ------------------------------------- noise transforms
class RicianNoiseTransform:
    def __init__(self, random_state, noise_variance=(0, 0.1), p_per_sample=1, with_channel=False):
        self.p_per_sample = p_per_sample
        self.noise_variance = noise_variance
        self.with_channel = with_channel
        self.random_state = random_state

    def __call__(self, m):
        if self.random_state.uniform() < self.p_per_sample:
            variance = self.random_state.uniform(self.noise_variance[0], self.noise_variance[1])
            m = np.sqrt((m + self.random_state.normal(0.0, variance, size=m.shape)) ** 2 +
                        self.random_state.normal(0.0, variance, size=m.shape) ** 2) * np.sign(m)

        return m


class GaussianNoiseTransform:
    def __init__(self, random_state, noise_variance=(0, 0.1), p_per_sample=1., with_channel=False):
        self.p_per_sample = p_per_sample
        self.noise_variance = noise_variance
        self.with_channel = with_channel
        self.random_state = random_state

    def __call__(self, m):
        if self.random_state.uniform() < self.p_per_sample:
            if self.noise_variance[0] == self.noise_variance[1]:
                variance = self.noise_variance[0]
            else:
                variance = self.random_state.uniform(self.noise_variance[0], self.noise_variance[1])

            m = m + self.random_state.normal(0.0, variance, size=m.shape)
        return m


class GaussianBlurTransform:
    def __init__(self, random_state, blur_sigma=(1, 5), different_sigma_per_channel=True,
                 p_per_channel=1., p_per_sample=1., with_channel=False):
        self.random_state = random_state
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.blur_sigma = blur_sigma
        self.with_channel = with_channel

    def __call__(self, m):
        if self.random_state.uniform() < self.p_per_sample:
            if self.with_channel:
                sigma = get_range_val(self.blur_sigma)
                for c in range(len(m)):
                    if self.random_state.uniform() <= self.p_per_channel:
                        if self.different_sigma_per_channel:
                            sigma = get_range_val(self.blur_sigma)
                        m[c] = gaussian_filter(m, sigma, order=0)
            else:
                sigma = get_range_val(self.blur_sigma)
                m = gaussian_filter(m, sigma, order=0)

        return m


class BlankSquareNoiseTransform:
    def __init__(self, random_state, squre_size=20, n_squres=1, noise_val=(0, 0),
                 channel_wise_n_val=False, square_pos=None,
                 p_per_sample=1, with_channel=False):
        '''
        :param squre_size:
        :param n_squres:
        :param noise_val:
        :param channel_wise_n_val:
        :param square_pos:
        :param p_per_sample:
        :param with_channel:
        '''

        self.p_per_sample = p_per_sample
        self.noise_val = noise_val
        self.n_squres = n_squres
        self.squre_size = squre_size
        self.channel_wise_n_val = channel_wise_n_val
        self.square_pos = square_pos
        self.with_channel = with_channel
        self.random_state = random_state

        if not self.with_channel:
            self.channel_wise_n_val = False

    def __call__(self, data):
        '''
        :param data: only support 2<=ndim<=4,
                     when channel_wise is True, support {HW/CHW/NCHW}
                     when channel_sise is False, support {...HW}
        :return:
        '''

        if self.random_state.uniform() < self.p_per_sample:
            data = augment_blank_square_noise(data, self.squre_size,
                                              self.n_squres, self.noise_val,
                                              self.channel_wise_n_val, self.square_pos)
        return data


class BlankRectangleTransform:
    def __init__(self, random_state, rectangle_size, rectangle_value, num_rectangles, force_square=False,
                 p_per_sample=0.5, p_per_channel=0.5,
                 with_channel=False):
        """
        Currently under development. This will replace BlankSquareNoiseTransform soon

        Overwrites areas in tensors specified by apply_to_keys with rectangles of some intensity

        This transform supports nD data.

        Note that we say square/rectangle here but we really mean line/square/rectangle/cube/whatevs.

        :param rectangle_size: rectangle size range. Can be either
            - int: creates only squares with edge length rectangle_size
            - tuple/list of int: constant size for rectangles is used. List/Tuple must have the same length as the
              data has dimensions (so len=3 for 3D images)
            - tuple/list of tuple/list: must have the same length as the data has dimensions. internal tuple/list
            specify a range from wich rectangle size will be sampled uniformly, for example: ((5, 10), (7, 12)) will
            generate rectangles between edge length between 5 and 10 for x and 7 and 12 for the y axis.
            - IMPORTANT: if force_square=True then only the first entry of the list will be used. So in the previous
            example rectangle_size=((5, 10), (7, 12)) the (7, 12) entry will be ignored and only squares between edge
            length (5, 10) in all dimensions will be produced
        :param rectangle_value: Intensity value to overwrite the voxels within the square with. Can be int, tuple,
        string, or callable:
            - int: always use the value specified by  rectangle_value
            - tuple: for example (0, 10) uniformly samples intensity values from the given interval. Note that the
            first entry must be smaller than the second! (10, 0) is not valid.
            - callable: we call rectangle_value(x) for each rectangle and you decide what happens (where x is the
            patch to be replaced)
        :param num_rectangles: Specifies the number of rectangles produced per selected image (depends on p_per_sample
        and p_per_channel). Canbe either int or tuple (for example (1, 5)) specifying a range form which the number
        of rectangles is uniformly sampled (note that we use np.random.random_integers, so the upper value is never
        selected (5 in this case). You can give 5.1 or so to make sure 5 gets selected as well)
        :param force_square: If True, only produces squares. In that case, all but the first entry of rectangle_size
        is discarded (also see doc for rectangle_size)
        :param p_per_sample:
        :param p_per_image:
        """
        self.rectangle_size = rectangle_size
        self.num_rectangles = num_rectangles
        self.force_square = force_square
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.with_channel = with_channel
        self.random_state = random_state

        # intensity value
        if np.isscalar(rectangle_value):
            self.color_fn = lambda x: rectangle_value
        elif callable(rectangle_value):
            self.color_fn = lambda x: rectangle_value(x)
        elif isinstance(rectangle_value, (tuple, list)):
            self.color_fn = lambda x: np.random.uniform(*rectangle_value)
        else:
            raise RuntimeError("unrecognized format for rectangle_value")

    def __call__(self, data):
        workon = data
        img_shape = workon.shape[1:]
        img_dim = len(img_shape)
        if self.random_state.uniform(0, 1) < self.p_per_sample:
            for c in range(workon.shape[0]):
                if self.random_state.uniform(0, 1) < self.p_per_channel:
                    # number of rectangles
                    n_rect = self.num_rectangles if isinstance(self.num_rectangles, int) else \
                        self.random_state.random_integers(*self.num_rectangles)
                    for rect_id in range(n_rect):
                        if isinstance(self.rectangle_size, int):
                            rectangle_size = [self.rectangle_size for d in img_shape]
                        elif isinstance(self.rectangle_size, (tuple, list)) and \
                                all([isinstance(i, int) for i in self.rectangle_size]):
                            rectangle_size = self.rectangle_size
                        elif isinstance(self.rectangle_size, (tuple, list)) and \
                                all([isinstance(i, (tuple, list)) for i in self.rectangle_size]):
                            if self.force_square:
                                rectangle_size = [self.random_state.random_integers(*self.rectangle_size[0])] * img_dim
                            else:
                                rectangle_size = [self.random_state.random_integers(*self.rectangle_size[d])
                                                  for d in range(img_dim)]
                        else:
                            raise RuntimeError("unrecognized format for rectangle_size")

                        lb = [self.random_state.random_integers(img_shape[i] - rectangle_size[i]) for i in range(img_dim)]
                        ub = [i + j for i, j in zip(lb, rectangle_size)]

                        my_slice = tuple([c, *[slice(i, j) for i, j in zip(lb, ub)]])

                        # figure out intensity value
                        intensity = self.color_fn(workon[my_slice])

                        workon[my_slice] = intensity
        return workon


class MedianFilterTransform:
    def __init__(self,
                 random_state,
                 filter_size: Union[int, Tuple[int, int]],
                 same_for_each_channel: bool = False,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 with_channel: bool = False
                 ):
        """
        :param filter_size:
        :param same_for_each_channel:
        :param p_per_sample:
        :param p_per_channel:
        """
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.filter_size = filter_size
        self.same_for_each_channel = same_for_each_channel
        self.with_channel = with_channel
        self.random_state = random_state

    def __call__(self, data):
        if self.random_state.uniform() < self.p_per_sample:
            if not self.with_channel:
                filter_size = self.filter_size if isinstance(self.filter_size, int) else self.random_state.randint(*self.filter_size)
                data = median_filter(data, filter_size)
            elif self.same_for_each_channel:
                filter_size = self.filter_size if isinstance(self.filter_size, int) else self.random_state.randint(*self.filter_size)
                for c in range(data.shape[0]):
                    if self.random_state.uniform() < self.p_per_channel:
                        data[c] = median_filter(data[c], filter_size)
            else:
                for c in range(len(data)):
                    if np.random.uniform() < self.p_per_channel:
                        filter_size = self.filter_size if isinstance(self.filter_size, int) else self.random_state.randint(*self.filter_size)
                        data[c] = median_filter(data[c], filter_size)
        return data


class SharpeningTransform:
    filter_2d = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])
    filter_3d = np.array([[[0, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]],
                          [[0, -1, 0],
                           [-1, 6, -1],
                           [0, -1, 0]],
                          [[0, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]],
                          ])

    def __init__(self,
                 strength: Union[float, Tuple[float, float]] = 0.2,
                 same_for_each_channel: bool = False,
                 p_per_sample: float = 1.,
                 p_per_channel: float = 1.,
                 with_channel: bool = False
                 ):
        """
        :param strength:
        :param same_for_each_channel:
        :param p_per_sample:
        :param p_per_channel:
        """
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.strength = strength
        self.same_for_each_channel = same_for_each_channel
        self.with_channel = with_channel

    def __call__(self, data):
        if np.random.uniform() < self.p_per_sample:
            flag_2d = self._is_2d(len(data.shape))
            if self.same_for_each_channel:
                mn, mx = data.min(), data.max()
                strength_here = self.strength if isinstance(self.strength, float) else np.random.uniform(*self.strength)
                filter_here = self._get_fileter_here(strength_here, flag_2d)
                for c in range(data.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        data[c] = convolve(data[c], filter_here, mode='same')
                        data[c] = np.clip(data[c], mn, mx)
            else:
                for c in range(data.shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        mn, mx = data[c].min(), data[c].max()
                        strength_here = self.strength if isinstance(self.strength, float) else np.random.uniform(*self.strength)
                        filter_here = self._get_fileter_here(strength_here, flag_2d)
                        data[c] = convolve(data[c], filter_here, mode='same')
                        data[c] = np.clip(data[c], mn, mx)
        return data

    def _get_fileter_here(self, strength_here, flag_2d):
        if flag_2d:
            filter_here = self.filter_2d * strength_here
            filter_here[1, 1] += 1
        else:
            filter_here = self.filter_3d * strength_here
            filter_here[1, 1, 1] += 1
        return filter_here

    def _is_2d(self, ndim):
        assert ndim in {2, 3, 4}
        if (self.with_channel and ndim == 4) or (not self.with_channel and ndim == 3):
            return False
        elif (self.with_channel and ndim == 2) or (not self.with_channel and ndim == 4):
            raise IndexError('your ndim and with_channel is not match')
        else:
            return True


# --------------------------------------- color transforms
class ContrastAugmentationTransform:
    def __init__(self,
                 random_state,
                 contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True,
                 per_channel: bool = True,
                 p_per_sample: float = 1,
                 p_per_channel: float = 1,
                 with_channel: bool = False):
        """
        Augments the contrast of data
        :param contrast_range:
            (float, float): range from which to sample a random contrast that is applied to the data. If
                            one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                            and the other half <1 (in the inverval that was specified)
            callable      : must be contrast_range() -> float
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param p_per_sample:
        """
        self.random_state = random_state
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.with_channel = with_channel

    def __call__(self, data):
        if self.random_state.uniform() < self.p_per_sample:
            if self.with_channel:
                cal_axis = tuple(range(1, len(data.shape)))
                do_contrast_channel = self.random_state.uniform(size=len(data)) < self.p_per_channel
                mean = data.mean(axis=cal_axis, keepdims=True)

                if self.per_channel:
                    # factor = self._get_factor(size=len(data))
                    factor = np.array([self._get_factor(size=1) for _ in range(len(data))])
                else:
                    factor = np.array([self._get_factor(size=1)] * len(data))
                factor = np.expand_dims(factor, axis=cal_axis)

                if self.preserve_range:
                    minm = data.min(axis=cal_axis, keepdims=True)
                    maxm = data.max(axis=cal_axis, keepdims=True)
                    data[do_contrast_channel] = mean[do_contrast_channel] + factor[do_contrast_channel] * (
                            data[do_contrast_channel] - mean[do_contrast_channel])

                    data = np.clip(data, minm, maxm)
                else:
                    data[do_contrast_channel] = mean[do_contrast_channel] + factor[do_contrast_channel] * (
                            data[do_contrast_channel] - mean[do_contrast_channel])
            else:
                factor = self._get_factor(size=1)
                mean = data.mean()
                if self.preserve_range:
                    minm = data.min()
                    maxm = data.max()
                    data = np.clip(mean + factor * (data - mean), minm, maxm)
                else:
                    data = mean + factor * (data - mean)

        return data

    def _get_factor(self, size: int = 1):
        if callable(self.contrast_range):
            factor = np.array([self.contrast_range() for _ in range(size)])
        else:
            if self.random_state.random() < 0.5 and self.contrast_range[0] < 1:
                factor = np.random.uniform(self.contrast_range[0], 1, size=size)
            else:
                factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1], size=size)
        return factor


class BrightnessTransform:
    def __init__(self, random_state, mu, sigma, per_channel=True, p_per_sample=1., p_per_channel=1., with_channel=False):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param p_per_sample:
        """
        self.random_state = random_state
        self.p_per_sample = p_per_sample
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.with_channel = with_channel

    def __call__(self, data):
        if self.random_state.uniform() < self.p_per_sample:
            if self.with_channel:
                do_contrast_channel = self.random_state.uniform(size=len(data)) < self.p_per_channel
                if self.per_channel:
                    rnb_nb = self.random_state.normal(self.mu, self.sigma, size=len(data))
                else:
                    rnb_nb = np.array([self.random_state.normal(self.mu, self.sigma)] * len(data))
                rnb_nb = np.expand_dims(rnb_nb, axis=tuple(range(1, data.ndim)))
                data[do_contrast_channel] = data[do_contrast_channel] + rnb_nb[do_contrast_channel]
            else:
                rnd_nb = self.random_state.normal(self.mu, self.sigma)
                data += rnd_nb
        return data


class BrightnessMultiplicativeTransform:
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, p_per_sample=1., with_channel=False):
        self.p_per_sample = p_per_sample
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
        self.with_channel = with_channel

    def __call__(self, data):
        if np.random.uniform() < self.p_per_sample:
            if self.with_channel and self.per_channel:
                multiplier = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1], size=len(data))
                multiplier = np.expand_dims(multiplier, axis=tuple(range(1, data.ndim)))
                data = data * multiplier
            else:
                multiplier = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1])
                data *= multiplier
        return data


class GammaTransform:
    def __init__(self, random_state, gamma_range=(0.5, 2), invert_image=False, per_channel=False, retain_stats=False,
                 p_per_sample=1., with_channel=False, epsilon=1e-7):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.random_state = random_state
        self.with_channel = with_channel
        self.epsilon = epsilon

    def _get_gamma(self):
        if self.random_state.random() < 0.5 and self.gamma_range[0] < 1:
            gamma = self.random_state.uniform(self.gamma_range[0], 1)
        else:
            gamma = self.random_state.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
        return gamma

    def __call__(self, data):
        if self.random_state.uniform() < self.p_per_sample:
            if self.invert_image:
                data = - data

            if self.with_channel and self.per_channel:
                cal_axis = tuple(range(1, len(data.shape)))
                if self.retain_stats:
                    mn = np.mean(data, axis=cal_axis, keepdims=True)
                    sd = np.std(data, axis=cal_axis, keepdims=True)
                gamma = np.array([self._get_gamma() for _ in range(len(data))])
                gamma = np.expand_dims(gamma, axis=tuple(range(1, data.ndim)))
                minm = np.min(data, axis=cal_axis, keepdims=True)
                maxm = np.max(data, axis=cal_axis, keepdims=True)
                data = np.power((data - minm) / (maxm - minm + self.epsilon), gamma) * (maxm - minm + self.epsilon) + minm

                if self.retain_stats:
                    data = data - np.mean(data, axis=cal_axis, keepdims=True)
                    data = data / (np.std(data, axis=cal_axis, keepdims=True) + self.epsilon) * sd
                    data = data + mn
            else:
                if self.retain_stats:
                    mn = data.mean()
                    sd = data.std()
                gamma = self._get_gamma()
                minm = data.min()
                rnge = data.max() - minm
                data = np.power(((data - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
                if self.retain_stats:
                    data = data - data.mean()
                    data = data / (data.std() + 1e-8) * sd
                    data = data + mn

            if self.invert_image:
                data = - data
        return data


# -----------------------------------------copy from reg_3d and CHAOS----------------------------------------------

class RandomContrast:
    """
       Adjust the contrast of an image by a random factor inside a the contrast_range
    """
    def __init__(self, random_state, contrast_range=(0.25, 0.75), execution_probability=0.2, **kwargs):
        assert len(contrast_range) == 2
        self.min_factor, self.max_factor = contrast_range
        self.random_state = random_state
        self.execution_probability = execution_probability

    def __call__(self, m):
        factor = self.random_state.uniform(self.min_factor, self.max_factor)
        if self.random_state.uniform() < self.execution_probability:
            if m.ndim == 3:
                # take the mean intensity of the entire patch
                mean_intensity = np.mean(m)
            else:
                # if 4D then compute per channel mean intensity (assuming: CZYX axis order)
                mean_intensity = np.mean(m, axis=(1, 2, 3), keepdims=True)
            return np.clip(mean_intensity + factor * (m - mean_intensity), 0, 1)

        return m


class RandomBrightness:
    """
        Adjust the brightness of an image by a random factor inside a the brightness_range
        Brightness range: tuple,float. If it's a tuple a random factor will be taken from (brightness_range[0], brightness_range[1])
        If it's float then the random factor will be taken from (-brightness_range,brightness_range).
        The intervals must be included in [-1,1]. If not, they would be clipped to [-1,1]
    """
    def __init__(self, random_state, brightness_range=0.1, **kwargs):
        if isinstance(brightness_range, tuple):
            assert len(brightness_range) == 2
            self.brightness_min, self.brightness_max = np.clip(brightness_range, -1., 1.)
        else:
            self.brightness_min, self.brightness_max = np.clip([-brightness_range, brightness_range], -1., 1.)
        self.random_state = random_state

    def __call__(self, m):
        brightness = self.random_state.uniform(self.brightness_min, self.brightness_max)
        return np.clip(m + brightness, 0, 1)


class RandomBrightnessContrast:
    """
        Apply RandomBrightness and RandomContrast interchangeably
    """

    def __init__(self, random_state, brightness_range=0.1,
                 contrast_range=(0.25, 0.75), execution_probability=0.2, **kwargs):
        self.rand_contrast = RandomContrast(random_state, contrast_range, execution_probability)
        self.rand_brightness = RandomBrightness(random_state, brightness_range)
        self.random_state = random_state

    def __call__(self, m):
        if self.random_state.uniform() < 0.5:  # Alternates order of the Brightness and Contrast transforms
            m = self.rand_brightness(m)
            return self.rand_contrast(m)
        else:
            m = self.rand_contrast(m)
            return self.rand_brightness(m)


class ElasticDeform:
    """
    Apply elasitc deformations, it's relatively slow
    Args:
        order: int, the order of spline interpolation
        alpha: float, scaling factor for deformations
        sigma: float, smothing factor for Gaussian filter
    """
    def __init__(self, random_state, order=1, alpha=15, sigma=3):
        self.random_state = random_state
        self.order = order
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, m):
        assert m.ndim == 3
        dz = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha
        dx = gaussian_filter(self.random_state.randn(*m.shape), self.sigma, mode="constant", cval=0) * self.alpha

        z_dim, y_dim, x_dim = m.shape
        z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
        indices = z + dz, y + dy, x + dx
        return map_coordinates(m, indices, order=self.order, mode='reflect')


class AbstractLabelToBoundary:
    AXES_TRANSPOSE = [
        (0, 1, 2),  # X
        (0, 2, 1),  # Y
        (2, 0, 1)  # Z
    ]

    def __init__(self, ignore_index=None, aggregate_affinities=False, append_label=False, **kwargs):
        """
        :param ignore_index: label to be ignored in the output, i.e. after computing the boundary the label ignore_index
            will be restored where is was in the patch originally
        :param aggregate_affinities: aggregate affinities with the same offset across Z,Y,X axes
        :param append_label: if True append the orignal ground truth labels to the last channel
        """
        self.ignore_index = ignore_index
        self.aggregate_affinities = aggregate_affinities
        self.append_label = append_label

    def __call__(self, m):
        """
        Extract boundaries from a given 3D label tensor.
        :param m: input 3D tensor
        :return: binary mask, with 1-label corresponding to the boundary and 0-label corresponding to the background
        """
        assert m.ndim == 3

        kernels = self.get_kernels()
        channels = np.stack([np.where(np.abs(convolve(m, kernel)) > 0, 1, 0) for kernel in kernels])
        results = []
        if self.aggregate_affinities:
            assert len(kernels) % 3 == 0, "Number of kernels must be divided by 3 (one kernel per offset per Z,Y,X axes"
            # aggregate affinities with the same offset
            for i in range(0, len(kernels), 3):
                # merge across X,Y,Z axes (logical OR)
                xyz_aggregated_affinities = np.logical_or.reduce(channels[i:i + 3, ...]).astype(np.int)
                # recover ignore index
                xyz_aggregated_affinities = _recover_ignore_index(xyz_aggregated_affinities, m, self.ignore_index)
                results.append(xyz_aggregated_affinities)
        else:
            results = [_recover_ignore_index(channels[i], m, self.ignore_index) for i in range(channels.shape[0])]

        if self.append_label:
            # append original input data
            results.append(m)

        # stack across channel dim
        return np.stack(results, axis=0)

    @staticmethod
    def create_kernel(axis, offset):
        # create conv kernel
        k_size = offset + 1
        k = np.zeros((1, 1, k_size), dtype=np.int)
        k[0, 0, 0] = 1
        k[0, 0, offset] = -1
        return np.transpose(k, axis)

    def get_kernels(self):
        raise NotImplementedError


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input


class StandardLabelToBoundary:
    def __init__(self, ignore_index=None, append_label=False, blur=False, **kwargs):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.blur = blur

    def __call__(self, m):
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2)
        if self.blur:
            boundaries = gaussian(boundaries, sigma=1)
            boundaries[boundaries >= 0.5] = 1
            boundaries[boundaries < 0.5] = 0

        results = [_recover_ignore_index(boundaries, m, self.ignore_index)]

        if self.append_label:
            # append original input data
            results.append(m)

        return np.stack(results, axis=0)


class RandomLabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the max_offset (thickness) of the border. Then the offset is picked at random every time you call
    the transformer (offset is picked form the range 1:max_offset) for each axis and the boundary computed.
    One may use this scheme in order to make the network more robust against various thickness of borders in the ground
    truth  (think of it as a boundary denoising scheme).
    """
    def __init__(self, random_state, max_offset=8, ignore_index=None, append_label=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label, aggregate_affinities=False)
        self.random_state = random_state
        self.offsets = tuple(range(1, max_offset + 1))

    def get_kernels(self):
        rand_offset = self.random_state.choice(self.offsets)
        axis_ind = self.random_state.randint(3)
        rand_axis = self.AXES_TRANSPOSE[axis_ind]
        # return a single kernel
        return [self.create_kernel(rand_axis, rand_offset)]


class LabelToBoundary(AbstractLabelToBoundary):
    """
    Converts a given volumetric label array to binary mask corresponding to borders between labels.
    One specify the offsets (thickness) of the border. The boundary will be computed via the convolution operator.
    """
    def __init__(self, offsets, ignore_index=None, append_label=False, aggregate_affinities=False, **kwargs):
        super().__init__(ignore_index=ignore_index, append_label=append_label,
                         aggregate_affinities=aggregate_affinities)
        if isinstance(offsets, int):
            assert offsets > 0, "'offset' must be positive"
            offsets = [offsets]
        elif isinstance(offsets, list) or isinstance(offsets, tuple):
            assert all(a > 0 for a in offsets), "'offset' must be positive"
            assert len(set(offsets)) == len(offsets), "'offsets' must be unique"
        else:
            raise ValueError(f"Unsupported 'offsets' type {type(offsets)}")

        self.kernels = []
        # create kernel for every axis-offset pair
        for offset in offsets:
            for axis in self.AXES_TRANSPOSE:
                # create kernels for a given offset in every direction
                self.kernels.append(self.create_kernel(axis, offset))

    def get_kernels(self):
        return self.kernels


class Normalize:
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """
    def __init__(self, mean, std, eps=1e-6, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        return (m - self.mean) / (self.std + self.eps)


class Identity:
    def __call__(self, m):
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


# --------------------------------transform api


class Transformer:
    '''
    It is used to transform on numpy array
    '''

    def __init__(self, opt):
        self.opt = opt
        self.preprocess = opt.preprocess
        if getattr(opt, 'random_state', None) is not None:
            self.random_state = opt.random_state
        elif getattr(opt, 'seed', None) is not None:
            self.random_state = np.random.RandomState(seed=opt.seed)

    def standard_transform(self):
        '''
        'centercrop_rot90_mirror______________'
        :return:
        '''
        opt = self.opt
        trans_list = []
        trans_list.append(ElasticDeformTransform(self.random_state,
                                                 order_data=opt.order_data, order_seg=opt.order_seg,
                                                 alpha=(0., 900.), sigma=(9., 13.),
                                                 p_el_per_sample=0.2,
                                                 with_channel=False))
        trans_list.append(RandomScaleTransform(self.random_state,
                                               order_data=opt.order_data, order_seg=opt.order_seg,
                                               scale=opt.scale_range, p_scale_per_sample=0.2,
                                               p_independent_scale_per_axis=1, independent_scale_for_each_axis=False,
                                               with_channel=False))
        trans_list.append(RandomRotateTransform(angle_spectrum=[(-opt.rot_angle_spectrum, opt.rot_angle_spectrum)],
                                                axes=list(combinations(np.unique(opt.rot_axes), 2)),
                                                p_per_sample=0.2, p_rot_per_axis=1, with_channel=False))
        trans_list.append(CenterCropTransform(opt.crop_size[::-1], with_channel=False))
        trans_list.append(Rot90Transform(num_rot=(1, 2, 3), axes=np.unique(opt.rot_axes),
                                         p_per_sample=0.5, with_channel=False))
        trans_list.append(MirrorTransform(axes=opt.mirror_axes, p_per_sample=1, with_channel=False))
        return ComposeForSample(trans_list)

    def custom_transform(self):
        '''
        'elastic_resize_zoom_randomscale_randomcrop_ranomrotate_centercrop_transposeaxes_randomshift_rot90_mirror'
        :return:
        '''
        opt = self.opt
        trans_list = []
        preprocess = self.preprocess.split('_')
        if 'elastic' in preprocess:
            trans_list.append(ElasticDeformTransform(self.random_state,
                                                     order_data=opt.order_data, order_seg=opt.order_seg,
                                                     alpha=opt.elastic_alpha, sigma=opt.elastic_sigma,
                                                     p_el_per_sample=0.2,
                                                     with_channel=False))
        if 'resize' in preprocess:
            trans_list.append(ResizeTransform(target_size=opt.target_size[::-1],
                                              order=opt.order_data, order_seg=opt.order_seg,
                                              with_channel=False))
        if 'zoom' in preprocess:
            trans_list.append(ZoomTransform(zoom_factors=opt.scale[::-1],
                                            order=opt.order_data, order_seg=opt.order_seg,
                                            with_channel=False))
        if 'randomscale' in preprocess:
            trans_list.append(RandomScaleTransform(self.random_state,
                                                   order_data=opt.order_data, order_seg=opt.order_seg,
                                                   scale=opt.scale_range, p_scale_per_sample=0.2,
                                                   p_independent_scale_per_axis=1, independent_scale_for_each_axis=False,
                                                   with_channel=False))
        if 'randomcrop' in preprocess:
            # # crop_size = opt.crop_size
            # # x*cos(t)+y*sin(t)
            # # int(np.ceil(np.sqrt(2)*crop_size[2]))

            # crop_size_refine = int(np.ceil(np.sqrt(np.sum(np.power(opt.crop_size, 2)))))
            # if opt.crop_size[0]*len(opt.crop_size) != np.sum(opt.crop_size):
            #     crop_size_refine = [opt.crop_size[-1]] + [crop_size_refine]*(len(opt.crop_size)-1)
            crop_size_refine = [a+10 for a in opt.crop_size]
            crop_size_refine = crop_size_refine[::-1]
            trans_list.append(RandomCropTransform(crop_size=crop_size_refine, with_channel=False))
        if 'ranomrotate' in preprocess:
            # [(-15, 15), (-15, 15), (-15, 15)]
            trans_list.append(RandomRotateTransform(angle_spectrum=[(-opt.rot_angle_spectrum, opt.rot_angle_spectrum)],
                                                    axes=list(combinations(np.unique(opt.rot_axes), 2)),
                                                    p_per_sample=0.2, p_rot_per_axis=1, with_channel=False))
        if 'centercrop' in preprocess:
            trans_list.append(CenterCropTransform(opt.crop_size[::-1], with_channel=False))
        if 'transposeaxes' in preprocess:
            trans_list.append(TransposeAxesTransform(transpose_any_of_these=(0, 1, 2), p_per_sample=0.8, with_channel=False))
        if 'randomshift' in preprocess:
            # 暂时不用，mu和sigma作为高斯分布的参数，从该分布采样偏移值
            trans_list.append(RandomShiftTransform(shift_mu=opt.shift_mu, shift_sigma=opt.shift_sigma,
                                                   p_per_sample=1, p_per_channel=0.5,
                                                   border_value=0, with_channel=False))
        if 'rot90' in preprocess:
            trans_list.append(Rot90Transform(num_rot=(1, 2, 3), axes=np.unique(opt.rot_axes),
                                             p_per_sample=0.5, with_channel=False))
        if 'mirror' in preprocess:
            trans_list.append(MirrorTransform(axes=opt.mirror_axes, p_per_sample=1, with_channel=False))
        return ComposeForSample(trans_list)

    def standard_transform_old(self):
        trans_list = []
        # trans_list.append(RandomCrop(self.random_state, 256))
        trans_list.append(RandomFlip(self.random_state, axes=(1, 2)))
        # trans_list.append(RandomRotate(self.random_state, angle_spectrum=5, axes=[(2, 1)]))
        trans_list.append(RandomRotate90(self.random_state))
        trans_list.append(ToTensor(expand_dims=False))
        return transforms.Compose(trans_list)

    def custom_transform_old(self):
        trans_list = []
        preprocess = self.preprocess.split('_')
        if 'resize' in preprocess:
            osize = self.opt.load_size
            order = self.opt.order
            trans_list.append(transforms.Lambda(lambda img: resize_3d(img, newSize=osize, order=order)))
        if 'scale' in preprocess:
            scale = self.opt.scale
            trans_list.append(RandomScale(self.random_state, scale=scale, order=1, execution_probability=0.2))
        if 'randomcrop' in preprocess and 'centercrop' in preprocess:
            crop_size = self.opt.crop_size
            # np.sqrt(2)*
            crop_size_refine = int(np.ceil(np.sqrt(2)*crop_size[0])), int(np.ceil(np.sqrt(2)*crop_size[1])), crop_size[2]
            trans_list.append(RandomCrop(self.random_state, crop_size_refine))
        if 'rotate' in preprocess:
            angle_spectrum = self.opt.angle_spectrum
            trans_list.append(RandomRotate(self.random_state, angle_spectrum=angle_spectrum, axes=[(1, 2)]))
        if 'randomcrop' in preprocess:
            crop_size = self.opt.crop_size
            if 'centercrop' in preprocess:
                trans_list.append(CenterCrop(crop_size))
            else:
                assert 'rotate' not in preprocess, 'it must not have the rotate when the centercrop not in preprocess'
                trans_list.append(RandomCrop(self.random_state, crop_size))
        if 'centercrop' in preprocess and 'randomcrop' not in preprocess:
            crop_size = self.opt.crop_size
            trans_list.append(CenterCrop(crop_size))

        if 'rot90' in preprocess:
            trans_list.append(RandomRotate90(self.random_state))
        if 'flip' in preprocess:
            trans_list.append(RandomFlip(self.random_state, axes=(1, 2)))

        trans_list.append(ToTensor(expand_dims=False))
        return transforms.Compose(trans_list)


def get_transform(opt):
    transfomer = Transformer(opt)
    if opt.custom:
        return transfomer.custom_transform()
    else:
        return transfomer.standard_transform()


def get_pre_transform(opt):
    transform_list = []

    # preprocess = opt.preprocess.split('_')
    # random_state = opt.random_state
    # transform_list.append(NormalizeRange(dtype=np.float32))
    # if 'BrightnessContrast' in preprocess:
    #     transform_list.append(RandomBrightnessContrast(random_state,
    #                                                    brightness_range=opt.brightness_range,
    #                                                    contrast_range=opt.contrast_range,
    #                                                    execution_probability=opt.execution_probability))
    # else:
    #     if 'brightness' in preprocess:
    #         transform_list.append(RandomBrightness(random_state, brightness_range=opt.brightness_range))
    #     if 'contrast' in preprocess:
    #         transform_list.append(RandomContrast(random_state,
    #                                              contrast_range=opt.contrast_range,
    #                                              execution_probability=opt.execution_probability))
    # if 'GaussianNoise' in preprocess:
    #     # print(type(opt.gaussian_sigma), opt.gaussian_sigma)
    #     transform_list.append(RandomGaussianNoise(random_state, sigma_range=opt.gaussian_sigma,
    #                                               alpha_range=0., execution_probability=0.15))
    return transforms.Compose(transform_list)


def get_post_transform(opt):
    '''
    'gaussianNoise_GaussianBlur_brightness_BrightnessMultiplicative_contrast_simulate_gammatransform'
    'gaussianNoise_GaussianBlur_BrightnessMultiplicative_contrast_simulate_gammatransform'
    :param opt:
    :return:
    '''
    transform_list = []
    preprocess = opt.preprocess.split('_')
    if 'gaussianNoise' in preprocess:
        transform_list.append(GaussianNoiseTransform(opt.random_state,
                                                     noise_variance=opt.g_noise_variance,
                                                     p_per_sample=0.15,
                                                     with_channel=False))
    if 'GaussianBlur' in preprocess:
        transform_list.append(GaussianBlurTransform(opt.random_state, blur_sigma=(0.5, 1.5),
                                                    different_sigma_per_channel=True,
                                                    p_per_sample=0.2, p_per_channel=0.5, with_channel=False))
    if 'brightness' in preprocess:
        transform_list.append(BrightnessTransform(opt.random_state,
                                                  mu=opt.bright_mu, sigma=opt.bright_sigma, per_channel=True,
                                                  p_per_sample=0.15, p_per_channel=0.5,
                                                  with_channel=False))
    if 'BrightnessMultiplicative' in preprocess:
        transform_list.append(BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3),
                                                                per_channel=True, p_per_sample=0.15,
                                                                with_channel=False))
    if 'contrast' in preprocess:
        transform_list.append(ContrastAugmentationTransform(opt.random_state,
                                                            contrast_range=(0.65, 1.5),
                                                            preserve_range=True, per_channel=True,
                                                            p_per_sample=0.15, p_per_channel=1., with_channel=False))
    if 'simulate' in preprocess:
        transform_list.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=False,
                                                             p_per_channel=0.5, p_per_sample=0.25, channels=None,
                                                             order_downsample=0, order_upsample=3,
                                                             ignore_axes=None, with_channel=False))
    if 'gammatransform' in preprocess:
        transform_list.append(GammaTransform(opt.random_state,
                                             gamma_range=(0.7, 1.5), invert_image=False,
                                             per_channel=False, retain_stats=True, p_per_sample=0.3,
                                             with_channel=False))

    # transform_list.append(ToTensor(expand_dims=True))
    return transforms.Compose(transform_list)
