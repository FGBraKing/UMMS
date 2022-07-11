# encoding: utf-8
import cv2
import torch
import random
import numpy as np

from scipy.ndimage.interpolation import map_coordinates, zoom, rotate, shift, affine_transform
from scipy.ndimage.filters import gaussian_filter, convolve
from skimage.transform import resize, rescale
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_spatial_2, \
    augment_mirroring, augment_transpose_axes, augment_zoom, augment_resize, augment_rot90
from batchgenerators.augmentations.utils import interpolate_img
from batchgenerators.augmentations.crop_and_pad_augmentations import pad_nd_image_and_seg, crop
from batchgenerators.transforms.spatial_transforms import augment_spatial


# ----------------------------------------- CUSTOM TRANSFORM ------------------------------------------------------
# --------------------spatial transforms
class ZoomTransform:
    def __init__(self, zoom_factors=1, order=3, order_seg=1, with_channel=False):
        """
        Zooms 'data' (and 'seg') by zoom_factors
        :param zoom_factors: int or list/tuple of int
        :param order: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        :param seg: can be None, if not None then it will also be zoomed by zoom_factors. Can also be list/tuple of
        np.ndarray (just like data). Must also be (b, c, x, y(, z))
        """
        # self.cval_seg = cval_seg
        self.order_seg = order_seg
        self.order = order
        self.zoom_factors = zoom_factors
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):
        '''
        :param data: c,d,h,w OR d,h,w
        :param seg:
        :param args:
        :param kwargs:
        :return:
        '''
        if self.with_channel:
            data, seg = augment_zoom(data, seg, self.zoom_factors, self.order, self.order_seg)
            # , self.cval_seg
        else:
            data = np.expand_dims(data, axis=0)
            seg = np.expand_dims(seg, axis=0) if seg is not None else seg

            data, seg = augment_zoom(data, seg, self.zoom_factors, self.order, self.order_seg)
            # , self.cval_seg

            data = np.squeeze(data, axis=0)
            seg = np.squeeze(seg, axis=0) if seg is not None else seg

        return data, seg


class Rot90Transform:
    def __init__(self, num_rot=(1, 2, 3), axes=(0, 1, 2), p_per_sample=0.3, with_channel=False):
        """
        :param num_rot: rotate by 90 degrees how often? must be tuple -> nom rot randomly chosen from that tuple
        :param axes: around which axes will the rotation take place? two axes are chosen randomly from axes.
        :param data_key:
        :param label_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.axes = axes
        self.num_rot = num_rot
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):

        if np.random.uniform() < self.p_per_sample:
            if self.with_channel:
                data, seg = augment_rot90(data, seg, self.num_rot, self.axes)
            else:
                data = np.expand_dims(data, axis=0)
                seg = np.expand_dims(seg, axis=0) if seg is not None else seg

                data, seg = augment_rot90(data, seg, self.num_rot, self.axes)

                data = np.squeeze(data, axis=0)
                seg = np.squeeze(seg, axis=0) if seg is not None else seg

        return data, seg


class ResizeTransform:

    def __init__(self, target_size, order=3, order_seg=1, with_channel=False):
        """
        Reshapes 'data' (and 'seg') to target_size
        :param target_size: int or list/tuple of int
        :param order: interpolation order for data (see skimage.transform.resize)
        :param order_seg: interpolation order for seg (see skimage.transform.resize)
        :param cval_seg: cval for segmentation (see skimage.transform.resize)
        :param seg: can be None, if not None then it will also be resampled to target_size. Can also be list/tuple of
        np.ndarray (just like data). Must also be (b, c, x, y(, z))

        """
        # self.cval_seg = cval_seg
        self.order_seg = order_seg
        self.order = order
        self.target_size = target_size
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):

        if self.with_channel:
            data, seg = augment_resize(data, seg, self.target_size, self.order, self.order_seg)
            # , self.cval_seg
        else:
            data = np.expand_dims(data, axis=0)
            seg = np.expand_dims(seg, axis=0) if seg is not None else seg

            data, seg = augment_resize(data, seg, self.target_size, self.order, self.order_seg)
            # , self.cval_seg

            data = np.squeeze(data, axis=0)
            seg = np.squeeze(seg, axis=0) if seg is not None else seg
        return data, seg


class MirrorTransform:
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), p_per_sample=1, with_channel=False):
        self.p_per_sample = p_per_sample
        self.axes = axes
        self.with_channel = with_channel
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, data, seg=None, *args, **kwargs):

        if np.random.uniform() < self.p_per_sample:

            if self.with_channel:
                data, seg = augment_mirroring(data, seg, axes=self.axes)
            else:
                data = np.expand_dims(data, axis=0)
                seg = np.expand_dims(seg, axis=0) if seg is not None else seg

                data, seg = augment_mirroring(data, seg, axes=self.axes)

                data = np.squeeze(data, axis=0)
                seg = np.squeeze(seg, axis=0) if seg is not None else seg
        return data, seg


class SpatialTransform:
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True,
                 p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1,
                 with_channel=False):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):

        if self.patch_size is None:
            if len(data.shape) == 3:
                patch_size = (data.shape[1], data.shape[2])
            elif len(data.shape) == 4:
                patch_size = (data.shape[1], data.shape[2], data.shape[3])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        if not self.with_channel:
            data = np.expand_dims(data, axis=0)
            seg = np.expand_dims(seg, axis=0) if seg is not None else seg

        data, seg = augment_spatial(data, seg, patch_size=patch_size,
                                    patch_center_dist_from_border=self.patch_center_dist_from_border,
                                    do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                    do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                    angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                    border_mode_data=self.border_mode_data,
                                    border_cval_data=self.border_cval_data, order_data=self.order_data,
                                    border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                    order_seg=self.order_seg, random_crop=self.random_crop,
                                    p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                    p_rot_per_sample=self.p_rot_per_sample,
                                    independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                    p_rot_per_axis=self.p_rot_per_axis,
                                    p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        if not self.with_channel:
            data = np.squeeze(data, axis=0)
            seg = np.squeeze(seg, axis=0) if seg is not None else seg

        return data, seg


class SpatialTransform_2:
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, deformation_scale=(0, 0.25),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True,
                 p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1, p_rot_per_axis: float = 1,
                 independent_scale_for_each_axis=False, p_independent_scale_per_axis: int = 1,
                 with_channel=False):
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.deformation_scale = deformation_scale
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_independent_scale_per_axis = p_independent_scale_per_axis
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_axis = p_rot_per_axis
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):
        if self.patch_size is None:
            if len(data.shape) == 3:
                patch_size = (data.shape[1], data.shape[2])
            elif len(data.shape) == 4:
                patch_size = (data.shape[1], data.shape[2], data.shape[3])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        if not self.with_channel:
            data = np.expand_dims(data, axis=0)
            seg = np.expand_dims(seg, axis=0) if seg is not None else seg

        data, seg = augment_spatial_2(data, seg, patch_size=patch_size,
                                      patch_center_dist_from_border=self.patch_center_dist_from_border,
                                      do_elastic_deform=self.do_elastic_deform, deformation_scale=self.deformation_scale,
                                      do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                      angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                      border_mode_data=self.border_mode_data,
                                      border_cval_data=self.border_cval_data, order_data=self.order_data,
                                      border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                      order_seg=self.order_seg, random_crop=self.random_crop,
                                      p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                      p_rot_per_sample=self.p_rot_per_sample,
                                      independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                      p_rot_per_axis=self.p_rot_per_axis,
                                      p_independent_scale_per_axis=self.p_independent_scale_per_axis)

        if not self.with_channel:
            data = np.squeeze(data, axis=0)
            seg = np.squeeze(seg, axis=0) if seg is not None else seg

        return data, seg


class TransposeAxesTransform:
    def __init__(self, transpose_any_of_these=(0, 1, 2), p_per_sample=1, with_channel=False):
        '''
        This transform will randomly shuffle the axes of transpose_any_of_these.
        Requires your patch size to have the same dimension in all axes specified in transpose_any_of_these. So if
        transpose_any_of_these=(0, 1, 2) the shape must be (128x128x128) and cannotbe, for example (128x128x96)
        (transpose_any_of_these=(0, 1) would be the correct one here)!
        :param transpose_any_of_these: spatial dimensions to transpose, 0=x, 1=y, 2=z. Must be a tuple/list of len>=2
        :param data_key:
        :param label_key:
        '''
        self.p_per_sample = p_per_sample
        self.transpose_any_of_these = transpose_any_of_these
        self.with_channel = with_channel
        if max(transpose_any_of_these) > 2:
            raise ValueError("TransposeAxesTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")
        assert isinstance(transpose_any_of_these, (list, tuple)), "transpose_any_of_these must be either list or tuple"
        assert len(
            transpose_any_of_these) >= 2, "len(transpose_any_of_these) must be >=2 -> we need at least 2 axes we " \
                                          "can transpose"

    def __call__(self, data, seg=None, *args, **kwargs):

        if np.random.uniform() < self.p_per_sample:
            if self.with_channel:
                data, seg = augment_transpose_axes(data, seg, self.transpose_any_of_these)
            else:
                data = np.expand_dims(data, axis=0)
                seg = np.expand_dims(seg, axis=0) if seg is not None else seg

                data, seg = augment_transpose_axes(data, seg, self.transpose_any_of_these)

                data = np.squeeze(data, axis=0)
                seg = np.squeeze(seg, axis=0) if seg is not None else seg

        return data, seg


class RandomRotateTransform:
    def __init__(self, angle_spectrum=((0, 360),), axes=None,
                 p_per_sample=1, p_rot_per_axis=0.3,
                 with_channel=False):
        self.angle_spectrum = angle_spectrum
        self.axes = axes

        self.p_rot = p_per_sample
        self.p_rot_per_axis = p_rot_per_axis

        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):
        if self.with_channel:  # CDHW OR CHW
            assert data.ndim in {3, 4}, "data have to be CDHW OR DHW"
            if data.ndim == 4:
                # is2d = False
                if self.axes is None:
                    self.axes = [(3, 2), (3, 1), (2, 1)]
            else:
                # is2d = True
                if self.axes is None:
                    self.axes = [(2, 1)]
        else:
            assert data.ndim in {2, 3}, "data have to be DHW OR HW"
            if data.ndim == 3:
                # is2d = False
                if self.axes is None:
                    self.axes = [(2, 1), (2, 0), (1, 0)]
            else:
                # is2d = True
                if self.axes is None:
                    self.axes = [(1, 0)]
        if len(self.angle_spectrum) == 1:
            self.angle_spectrum = self.angle_spectrum * len(self.axes)
        else:
            assert len(self.angle_spectrum) == len(self.axes), "angle_spectrum have to pair to axes"

        if np.random.uniform() < self.p_rot:
            for i_angle_spectrum, i_axes in zip(self.angle_spectrum, self.axes):
                angle = np.random.uniform(i_angle_spectrum[0], i_angle_spectrum[1])
                if np.random.uniform() < self.p_rot_per_axis:
                    data = rotate(data, angle, axes=i_axes, reshape=True, order=3, mode='constant', cval=0.0, prefilter=True)
                    if seg is not None:
                        seg = rotate(seg, angle, axes=i_axes, reshape=True, order=0, mode='constant', cval=-1, prefilter=False)

        return data, seg


class ElasticDeformTransform:
    def __init__(self, random_state, order_data=3, order_seg=0,
                 alpha=(0., 900.), sigma=(10., 13.), p_el_per_sample=1, with_channel=False):
        '''
        :param random_state:
        :param order_data:
        :param order_seg:
        :param alpha:
        :param sigma:
        :param p_el_per_sample:
        :param with_channel:
        '''
        self.random_state = random_state
        self.order_data = order_data
        self.order_seg = order_seg
        self.alpha = alpha
        self.sigma = sigma
        self.p_el_per_sample = p_el_per_sample
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):

        if self.with_channel:
            shape = data.shape[1:]
        else:
            shape = data.shape

        data_result = data  # np.zeros_like(data, dtype=np.float32)
        seg_result = seg

        if self.random_state.uniform() < self.p_el_per_sample:
            tmp = tuple([np.arange(i) for i in shape])
            coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)

            a = self.random_state.uniform(self.alpha[0], self.alpha[1])
            s = self.random_state.uniform(self.sigma[0], self.sigma[1])

            offsets = [gaussian_filter((np.random.random(coords.shape[1:]) * 2 - 1), s, mode="constant", cval=0) * a
                       for _ in range(len(coords))]
            indices = np.array(offsets) + coords

            if self.with_channel:
                for channel_id in range(len(data)):
                    data_result[channel_id, ...] = interpolate_img(data[channel_id], indices, self.order_data,
                                                                   mode='nearest', cval=0)
                if seg is not None:
                    seg_result = np.zeros_like(seg, dtype=np.float32)
                    for channel_id in range(len(seg)):
                        seg_result[channel_id, ...] = interpolate_img(seg[channel_id], indices, self.order_seg,
                                                                      mode='constant', cval=0, is_seg=True)
            else:
                data_result = interpolate_img(data, indices, self.order_data, mode='nearest', cval=0)
                if seg is not None:
                    seg_result = interpolate_img(seg, indices, self.order_seg, mode='constant', cval=0, is_seg=True)
        return data_result, seg_result


class RandomScaleTransform:
    def __init__(self,  random_state, order_data=3, order_seg=0, scale=(0.75, 1.25),
                 p_scale_per_sample=1, p_independent_scale_per_axis=1,
                 independent_scale_for_each_axis=False, with_channel=False):
        self.random_state = random_state
        self.order_data = order_data
        self.order_seg = order_seg
        # self.cval_seg = cval_seg
        self.p_scale_per_sample = p_scale_per_sample
        self.p_independent_scale_per_axis = p_independent_scale_per_axis
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.with_channel = with_channel
        if isinstance(scale, float):
            assert scale >= 0, 'scale should be greater than 0'
            self.scale = (1 - scale, 1 + scale)
        else:
            assert len(scale) == 2, 'You should give a range'
            self.scale = scale

    def __call__(self, data, seg=None, *args, **kwargs):
        if self.with_channel:
            ndim = len(data.shape) - 1
        else:
            ndim = len(data.shape)

        if self.random_state.uniform() < self.p_scale_per_sample:
            if self.independent_scale_for_each_axis and self.random_state.uniform() < self.p_independent_scale_per_axis:
                sc = []
                for _ in range(ndim):
                    # 保证放大和缩小的概率各半
                    if self.random_state.random() < 0.5 and self.scale[0] < 1:
                        sc.append(np.random.uniform(self.scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(self.scale[0], 1), self.scale[1]))
            else:
                if self.random_state.random() < 0.5 and self.scale[0] < 1:
                    sc = np.random.uniform(self.scale[0], 1)
                else:
                    sc = np.random.uniform(max(self.scale[0], 1), self.scale[1])

            if self.with_channel:
                data, seg = augment_zoom(data, seg, sc, self.order_data, self.order_seg)
                # , self.cval_seg
            else:
                data = np.expand_dims(data, axis=0)
                seg = np.expand_dims(seg, axis=0) if seg is not None else seg

                data, seg = augment_zoom(data, seg, sc, self.order_data, self.order_seg)
                # , self.cval_seg

                data = np.squeeze(data, axis=0)
                seg = np.squeeze(seg, axis=0) if seg is not None else seg

        return data, seg


class CenterCropTransform:
    def __init__(self, crop_size, with_channel=False):
        self.crop_size = crop_size
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):
        if self.with_channel:
            data = np.expand_dims(data, axis=0)
            seg = np.expand_dims(seg, axis=0) if seg is not None else seg

            data, seg = crop(data, seg, self.crop_size, 0, 'center')

            data = np.squeeze(data, axis=0)
            seg = np.squeeze(seg, axis=0) if seg is not None else seg
        else:
            data = np.expand_dims(data, axis=(0, 1))
            seg = np.expand_dims(seg, axis=(0, 1)) if seg is not None else seg

            data, seg = crop(data, seg, self.crop_size, 0, 'center')

            data = np.squeeze(data, axis=(0, 1))
            seg = np.squeeze(seg, axis=(0, 1)) if seg is not None else seg

        return data, seg


class RandomCropTransform:
    def __init__(self, crop_size=128, margins=(0, 0, 0), with_channel=False):
        self.crop_size = crop_size
        self.margins = margins
        self.with_channel = with_channel

    def __call__(self, data, seg=None, *args, **kwargs):
        if self.with_channel:
            data = np.expand_dims(data, axis=0)
            seg = np.expand_dims(seg, axis=0) if seg is not None else seg

            data, seg = crop(data, seg, self.crop_size, self.margins, 'random')

            data = np.squeeze(data, axis=0)
            seg = np.squeeze(seg, axis=0) if seg is not None else seg
        else:
            data = np.expand_dims(data, axis=(0, 1))
            seg = np.expand_dims(seg, axis=(0, 1)) if seg is not None else seg

            data, seg = crop(data, seg, self.crop_size, self.margins, 'random')

            data = np.squeeze(data, axis=(0, 1))
            seg = np.squeeze(seg, axis=(0, 1)) if seg is not None else seg

        return data, seg


class RandomCropWithStrideTransform:
    def __init__(self, crop_size=128, margins=(0, 0, 0), strides=1, with_channel=False):
        self.crop_size = crop_size
        self.margins = margins
        self.strides = strides
        self.with_channel = with_channel

    @staticmethod
    def get_lbs_for_random_crop_with_stride(crop_size, data_shape, margins, stride):
        lbs = []
        for i in range(len(data_shape)):
            if data_shape[i] - crop_size[i] - margins[i] > margins[i]:
                sample_points = list(range(margins[i], data_shape[i] - crop_size[i] - margins[i], stride[i]))
                lbs.append(np.random.choice(sample_points, size=1).item())
            else:
                lbs.append((data_shape[i] - crop_size[i]) // 2)
        return lbs

    def __call__(self, data, seg=None, *args, **kwargs):
        assert isinstance(data, np.ndarray), "data has to be a numpy array"

        data_shape = data.shape

        if seg is not None:
            assert isinstance(seg, np.ndarray), "seg has to be a numpy array"
            seg_shape = seg.shape
            assert data_shape == seg_shape,  "data and seg must have the same spatial dimensions. " \
                                             "Data: %s, seg: %s" % (str(data_shape), str(seg_shape))

        if self.with_channel:
            crop_data_dim = len(data_shape) - 1
            crop_data_shape = data_shape[1:]
        else:
            crop_data_dim = len(data_shape)
            crop_data_shape = data_shape

        if type(self.crop_size) not in (tuple, list, np.ndarray):
            self.crop_size = [self.crop_size] * crop_data_dim
        else:
            assert len(self.crop_size) == crop_data_dim, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"
        if not isinstance(self.margins, (np.ndarray, tuple, list)):
            self.margins = [self.margins] * crop_data_dim
        else:
            assert len(self.margins) == crop_data_dim, "If you provide a list/tuple as margins make sure it has the same dimension as your data (2d/3d)"
        if not isinstance(self.strides, (np.ndarray, tuple, list)):
            self.strides = [self.strides] * crop_data_dim
        else:
            assert len(self.strides) == crop_data_dim, "If you provide a list/tuple as strides make sure it has the same dimension as your data (2d/3d)"

        lbs = self.get_lbs_for_random_crop_with_stride(self.crop_size, crop_data_shape, self.margins, self.strides)
        need_to_pad = [[abs(min(0, lbs[d])), abs(min(0, crop_data_shape[d] - (lbs[d] + self.crop_size[d])))]
                       for d in range(crop_data_dim)]
        ubs = [min(lbs[d] + self.crop_size[d], crop_data_shape[d]) for d in range(crop_data_dim)]
        lbs = [max(0, lbs[d]) for d in range(crop_data_dim)]

        if self.with_channel:
            slicer_data = [slice(0, data_shape[1])] + [slice(lbs[d], ubs[d]) for d in range(crop_data_dim)]
        else:
            slicer_data = [slice(lbs[d], ubs[d]) for d in range(crop_data_dim)]

        pad_mode = 'constant'
        pad_kwargs = {'constant_values': 0}
        pad_mode_seg = 'constant'
        pad_kwargs_seg = {'constant_values': 0}

        data = data[tuple(slicer_data)]
        if any([i > 0 for j in need_to_pad for i in j]):
            data = np.pad(data, need_to_pad, pad_mode, **pad_kwargs)

        if seg is not None:
            seg = seg[tuple(slicer_data)]
            if any([i > 0 for j in need_to_pad for i in j]):
                seg = np.pad(seg, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        return data, seg


class RandomShiftTransform:
    def __init__(self, shift_mu, shift_sigma, p_per_sample=1, p_per_channel=0.5, border_value=0, with_channel=False):
        """
        randomly shifts the data by some amount. Equivalent to pad -> random crop but with (probably) less
        computational requirements

        shift_mu gives the mean value of the shift, 0 is recommended
        shift_sigma gives the standard deviation of the shift

        shift will ne drawn from a Gaussian distribution with mean shift_mu and variance shift_sigma

        shift_mu and shift_sigma can either be float values OR tuples of float values. If they are tuples they will
        be interpreted as separate mean and std for each dimension

        TODO separate per channel or not?

        :param shift_mu:
        :param shift_sigma:
        :param p_per_sample:
        :param p_per_channel:
        """
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.shift_sigma = shift_sigma
        self.shift_mu = shift_mu
        self.border_value = border_value
        self.with_channel = with_channel

    def __call__(self, *data_list):
        #
        result_list = []
        for workon in data_list:
            if np.random.uniform(0, 1) < self.p_per_sample:
                if not self.with_channel:
                    shift_here = []
                    for d in range(len(workon.shape)):
                        shift_here.append(int(np.round(np.random.normal(
                            self.shift_mu[d] if isinstance(self.shift_mu, (list, tuple)) else self.shift_mu,
                            self.shift_sigma[d] if isinstance(self.shift_sigma, (list, tuple)) else self.shift_sigma,
                            size=1))))
                    data_copy = np.ones_like(workon) * self.border_value
                    lb_x = max(shift_here[0], 0)
                    ub_x = max(0, min(workon.shape[0], workon.shape[0] + shift_here[0]))
                    lb_y = max(shift_here[1], 0)
                    ub_y = max(0, min(workon.shape[1], workon.shape[1] + shift_here[1]))

                    t_lb_x = max(-shift_here[0], 0)
                    t_ub_x = max(0, min(workon.shape[0], workon.shape[0] - shift_here[0]))
                    t_lb_y = max(-shift_here[1], 0)
                    t_ub_y = max(0, min(workon.shape[1], workon.shape[1] - shift_here[1]))

                    if len(shift_here) == 2:
                        data_copy[t_lb_x:t_ub_x, t_lb_y:t_ub_y] = workon[lb_x:ub_x, lb_y:ub_y]
                    elif len(shift_here) == 3:
                        lb_z = max(shift_here[2], 0)
                        ub_z = max(0, min(workon.shape[2], workon.shape[2] + shift_here[2]))
                        t_lb_z = max(-shift_here[2], 0)
                        t_ub_z = max(0, min(workon.shape[2], workon.shape[2] - shift_here[2]))
                        data_copy[t_lb_x:t_ub_x, t_lb_y:t_ub_y, t_lb_z:t_ub_z] = workon[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z]
                    workon = data_copy

                for c in range(workon.shape[0]):
                    if np.random.uniform(0, 1) < self.p_per_channel:
                        shift_here = []
                        for d in range(len(workon.shape) - 1):
                            shift_here.append(int(np.round(np.random.normal(
                                self.shift_mu[d] if isinstance(self.shift_mu, (list, tuple)) else self.shift_mu,
                                self.shift_sigma[d] if isinstance(self.shift_sigma, (list, tuple)) else self.shift_sigma,
                                size=1))))
                        data_copy = np.ones_like(workon[c]) * self.border_value
                        lb_x = max(shift_here[0], 0)
                        ub_x = max(0, min(workon.shape[1], workon.shape[1] + shift_here[0]))
                        lb_y = max(shift_here[1], 0)
                        ub_y = max(0, min(workon.shape[2], workon.shape[2] + shift_here[1]))

                        t_lb_x = max(-shift_here[0], 0)
                        t_ub_x = max(0, min(workon.shape[1], workon.shape[1] - shift_here[0]))
                        t_lb_y = max(-shift_here[1], 0)
                        t_ub_y = max(0, min(workon.shape[2], workon.shape[2] - shift_here[1]))

                        if len(shift_here) == 2:
                            data_copy[t_lb_x:t_ub_x, t_lb_y:t_ub_y] = workon[c, lb_x:ub_x, lb_y:ub_y]
                        elif len(shift_here) == 3:
                            lb_z = max(shift_here[2], 0)
                            ub_z = max(0, min(workon.shape[3], workon.shape[3] + shift_here[2]))
                            t_lb_z = max(-shift_here[2], 0)
                            t_ub_z = max(0, min(workon.shape[3], workon.shape[3] - shift_here[2]))
                            data_copy[t_lb_x:t_ub_x, t_lb_y:t_ub_y, t_lb_z:t_ub_z] = workon[c, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z]
                        workon[c] = data_copy

            result_list.append(workon)

        return result_list


# ------------------------------------OLD IMPLEMENT--------------------------------------------
# ------------------------------old class-------------------

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """
    def __init__(self, random_state, angle_spectrum=10, axes=None, mode='constant', execution_probability=0.2, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.execution_probability = execution_probability

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)
        if self.random_state.random_sample() < self.execution_probability:
            if m.ndim == 3:
                m = rotate(m, angle, axes=axis, reshape=True, order=0, mode=self.mode, cval=0)
            else:
                channels = [rotate(m[c], angle, axes=axis, reshape=True, order=0, mode=self.mode, cval=-1) for c in
                            range(m.shape[0])]
                m = np.stack(channels, axis=0)
        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state):
        self.random_state = random_state

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, (1, 2))
        else:
            channels = [np.rot90(m[c], k, (1, 2)) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomCrop:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """
    def __init__(self, random_state, crop_size=256):        # crop_size: W H D, xyz
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.crop_size = crop_size

    def __call__(self, m):
        assert m.ndim in [2, 3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        if m.ndim == 2:
            h, w = m.shape
            if isinstance(self.crop_size, int):
                self.crop_size = (self.crop_size,)*2
            x = self.random_state.randint(0, np.maximum(0, w - self.crop_size[0]))
            y = self.random_state.randint(0, np.maximum(0, h - self.crop_size[1]))
            return m[y:y + self.crop_size[1], x:x + self.crop_size[0]]
        elif m.ndim == 3:
            d, h, w = m.shape
            if isinstance(self.crop_size, int):
                self.crop_size = (self.crop_size,)*3  # HWD
            x = self.random_state.randint(0, np.maximum(1, w - self.crop_size[0]))
            y = self.random_state.randint(0, np.maximum(1, h - self.crop_size[1]))
            z = self.random_state.randint(0, np.maximum(1, d - self.crop_size[2]))
            return m[z:z + self.crop_size[2], y:y + self.crop_size[1], x:x + self.crop_size[0]]
        else:
            c, d, h, w = m.shape
            if isinstance(self.crop_size, int):
                self.crop_size = (self.crop_size,)*3
            # print(self.crop_size)
            # print(m.shape)
            x = self.random_state.randint(0, np.maximum(1, w - self.crop_size[0]))
            y = self.random_state.randint(0, np.maximum(1, h - self.crop_size[1]))
            z = self.random_state.randint(0, np.maximum(1, d - self.crop_size[2]))
            return m[:, z:z + self.crop_size[2], y:y + self.crop_size[1], x:x + self.crop_size[0]]


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """
    def __init__(self, random_state, axes=(0, 1, 2), execution_probability=0.2):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = axes
        self.execution_probability = execution_probability

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        for axis in self.axes:
            if self.random_state.uniform() < self.execution_probability:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)
        return m


class CenterCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, m, *args, **kwargs):
        assert m.ndim in [2, 3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        if m.ndim == 2:
            h, w = m.shape
            if isinstance(self.crop_size, int):
                self.crop_size = (self.crop_size,)*2
            x = (w - self.crop_size[0]) // 2
            y = (h - self.crop_size[1]) // 2
            return m[y:y + self.crop_size[1], x:x + self.crop_size[0]]
        elif m.ndim == 3:
            d, h, w = m.shape
            if isinstance(self.crop_size, int):
                self.crop_size = (self.crop_size,)*3  # HWD
            x = (w - self.crop_size[0]) // 2
            y = (h - self.crop_size[1]) // 2
            z = (d - self.crop_size[2]) // 2
            return m[z:z + self.crop_size[2], y:y + self.crop_size[1], x:x + self.crop_size[0]]
        else:
            c, d, h, w = m.shape
            if isinstance(self.crop_size, int):
                self.crop_size = (self.crop_size,)*3
            x = (w - self.crop_size[0]) // 2
            y = (h - self.crop_size[1]) // 2
            z = (d - self.crop_size[2]) // 2
            return m[:, z:z + self.crop_size[2], y:y + self.crop_size[1], x:x + self.crop_size[0]]


class RandomScale:
    def __init__(self, random_state, scale, order, order_seg, execution_probability=1.):
        self.random_state = random_state
        self.scale = scale
        if isinstance(scale, float):
            assert scale >= 0, 'scale should be greater than 0'
            self.scale_range = (1 - scale, 1 + scale)
        else:
            self.scale_range = None
        self.order = order
        # if order>1, prefilter should be True, the output will be slightly blurred
        if self.order > 1:
            self.prefilter = True
        else:
            self.prefilter = False
        self.execution_probability = execution_probability

    def __call__(self, m, *args, **kwargs):
        if self.random_state.random() < self.execution_probability:  # random()/sample()/rand()/random_sample()
            if self.scale_range:
                scale = self.random_state.uniform(self.scale_range[0], self.scale_range[1], 1).item()
            else:
                scale = self.scale

            return zoom(m, scale, order=self.order, mode='constant', cval=0.0, prefilter=self.prefilter)
        else:
            return m


# -----------------------------old function---------------------------------------
def random_scale(volume, label, scale_in, scale_range=None, execution_probability=1):
    ''' volume's ndim should > 1'''
    if isinstance(scale_in, float):
        assert scale_in >= 0, 'scale should be greater than 0'
        scale_range = (1 - scale_in, 1 + scale_in)
    else:
        assert len(scale_in) == volume.ndim, 'need: len(scale_in) == volume.ndim'

    if np.random.random() < execution_probability:  # random()/sample()/rand()/random_sample()
        if scale_range:
            scale = np.random.uniform(scale_range[0], scale_range[1], 1).item()
        else:
            scale = scale_in
        volume_out = zoom(volume, scale, order=3, mode='constant', cval=0.0, prefilter=False)
        label_out = zoom(label, scale, order=0, mode='constant', cval=0.0, prefilter=False)
        return volume_out, label_out
    else:
        return volume, label


def agent_resize(volume, outputsize, order=1, **kwargs):
    origin_shape = np.array(volume.shape)
    new_shape = np.array(outputsize)
    scale = new_shape.astype(np.float) / origin_shape
    return zoom(volume, scale, order=order, **kwargs)


def resize_3d(array, newSize=None, scale=None, order=3):
    oldSize =np.shape(array)
    if scale != None:
        scale = np.array(scale, float)
        newSize = oldSize * scale
    elif newSize != None:
        newSize = np.array(newSize, float)
        scale = newSize / oldSize
    newSize = newSize.astype(np.int)
    out_array = zoom(array, scale, order=order, mode='constant', cval=0.0)
    return out_array


def rescale3d(volumes, scale_range, order=1):
    """3d 尺度变换"""
    if not isinstance(volumes, list):
        volumes = [volumes]
        return_list = False
    else:
        return_list = True
    x_scale, y_scale, z_scale = list(scale_range)
    org_x_len, org_y_len, org_z_len = volumes[0].shape
    x_len = np.round(org_x_len * x_scale)
    y_len = np.round(org_y_len * y_scale)
    z_len = np.round(org_z_len * z_scale)
    output_shape = (x_len, y_len, z_len)
    trans_volumes = []
    for volume in volumes:
        trans_volumes.append(resize(volume, output_shape, order=order, preserve_range=True))
    if return_list:
        return trans_volumes
    else:
        return trans_volumes[0]


def random_padding_crop(volume, label, output_size, mode='constant'):
    len_x, len_y, len_z = volume.shape
    len_x_o, len_y_o, len_z_o = output_size
    if len_x < len_x_o:
        # need pad
        pad_len = len_x_o - len_x
        pad_before = np.random.randint(0, pad_len + 1)
        pad_after = pad_len - pad_before
        volume = np.pad(volume, [[pad_before, pad_after], [0, 0], [0, 0]], mode=mode)
        label = np.pad(label, [[pad_before, pad_after], [0, 0], [0, 0]], mode=mode)
    elif len_x > len_x_o:
        clip_x = len_x - len_x_o
        clip_x = np.random.randint(0, clip_x)
        volume = volume[clip_x:clip_x + len_x_o, :, :]
        label = label[clip_x:clip_x + len_x_o, :, :]
    if len_y < len_y_o:
        # need pad
        pad_len = len_y_o - len_y
        pad_before = np.random.randint(0, pad_len + 1)
        pad_after = pad_len - pad_before
        volume = np.pad(volume, [[0, 0], [pad_before, pad_after], [0, 0]], mode=mode)
        label = np.pad(label, [[0, 0], [pad_before, pad_after], [0, 0]], mode=mode)
    elif len_y > len_y_o:
        clip_y = len_y - len_y_o
        clip_y = np.random.randint(0, clip_y)
        volume = volume[:, clip_y:clip_y + len_y_o, :]
        label = label[:, clip_y:clip_y + len_y_o, :]
    if len_z < len_z_o:
        # need pad
        pad_len = len_z_o - len_z
        pad_before = np.random.randint(0, pad_len + 1)
        pad_after = pad_len - pad_before
        volume = np.pad(volume, [[0, 0], [0, 0], [pad_before, pad_after]], mode=mode)
        label = np.pad(label, [[0, 0], [0, 0], [pad_before, pad_after]], mode=mode)
    elif len_z > len_z_o:
        clip_z = len_z - len_z_o
        clip_z = np.random.randint(0, clip_z)
        volume = volume[:, :, clip_z:clip_z + len_z_o]
        label = label[:, :, clip_z:clip_z + len_z_o]

    return volume, label


def random_rescale3d(volume, label, scale_range=0.1):
    """3d 尺度变换"""
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range, 3)
    x_scale, y_scale, z_scale = scale_factor
    org_x_len, org_y_len, org_z_len = volume.shape
    x_len = np.round(org_x_len * x_scale)
    y_len = np.round(org_y_len * y_scale)
    z_len = np.round(org_z_len * z_scale)
    output_shape = (x_len, y_len, z_len)
    trans_volume = resize(volume, output_shape, order=2, preserve_range=True)
    trans_label = resize(label, output_shape, order=0, preserve_range=True)

    return trans_volume, trans_label


def random_rotate3d(volume, label, rotate_range=15):
    angles = np.random.uniform(0, rotate_range, 3)
    for axes in [(0, 1), (1, 2), (0, 2)]:
        if np.random.random() < 0.5 and angles[0] != 0:
            volume = rotate(volume, angle=angles[0], axes=axes, order=2, reshape=False)
            label = rotate(label, angle=angles[0], axes=axes, order=0, reshape=False)

    return volume, label


def ramdom_flip(volume1, label1, volume2, label2):
    for ax in range(3):
        if random.random() < 0.5:
            volume1 = np.flip(volume1, axis=ax)
            label1 = np.flip(label1, axis=ax)
            volume2 = np.flip(volume2, axis=ax)
            label2 = np.flip(label2, axis=ax)

    return volume1, label1, volume2, label2


