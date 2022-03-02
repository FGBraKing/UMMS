import os
import torch
import warnings
import numpy as np
import SimpleITK as sitk
from typing import List
try:
    from nipype.interfaces.ants import N4BiasFieldCorrection
except ModuleNotFoundError:
    pass


class ToArray:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, dtype=np.float32, normalize=True, **kwargs):
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, m, eps=1e-10):
        m_arr = np.array(m, dtype=self.dtype)
        if self.normalize:
            m_arr = (m_arr - np.min(m_arr)) / (np.max(m_arr) - np.min(m_arr) + eps)

        return m_arr


class Identity:
    def __call__(self, m):
        return m


class OneOfTransform:
    def __init__(self, list_of_transforms: List):
        """
        Randomly selects one of the transforms given in list_of_transforms and applies it with each call. Remember that
        probabilities of the individual transforms for being applied still exist and apply!
        :param list_of_transforms:
        """
        self.list_of_transforms = list_of_transforms

    def __call__(self, m):
        i = np.random.choice(len(self.list_of_transforms))
        return self.list_of_transforms[i](m)


# transform on the class SimpleITK.SimpleITK.Image


class ResizeItk:
    ''' resample on the class of SimpleITK.SimpleITK.Image'''
    # sitk.sitkNearestNeighbor
    # sitk.sitkLinear

    def __init__(self, reference_image=None,
                 new_size=None, new_spacing=None,
                 new_orgin=None, new_direction=None,
                 resamplemethod=sitk.sitkNearestNeighbor):
        '''you have to set the size at lease'''
        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetNumberOfThreads(8)
        if reference_image:
            self.resampler.SetReferenceImage(reference_image)
        if new_size:
            self.resampler.SetSize(new_size)
        if new_spacing:
            self.resampler.SetOutputSpacing(new_spacing)
        if new_orgin:
            self.resampler.SetOutputOrigin(new_orgin)
        if new_direction:
            self.resampler.SetOutputDirection(new_direction)
        self.resampler.SetInterpolator(resamplemethod)

    def SetSize(self, outsize):
        self.resampler.SetSize(outsize)

    def __call__(self, *args, **kwargs):
        output = []
        for arg in args:
            if isinstance(arg, sitk.SimpleITK.Image):
                itk_img_resized = self.resampler.Execute(arg)
                output.append(itk_img_resized)
        return output


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    # Example:
    #     >>> transforms.Compose([
    #     >>>     transforms.CenterCrop(10),
    #     >>>     transforms.ToTensor(),
    #     >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposeForSample:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, label):
        for t in self.transforms:
            data, label = t(data, label)
        return data, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# 二维：cv2.resize()，np.resize()
# 三维
# 1. scipy.ndimage.interpolation.zoom()
# 2. torch.nn.functional.interpolate()
def resize_image_itk(itkimage, newSize=None, newSpacing=None, newOrigin=None, newDirection=None,
                     N4BiasCorrect=False, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSize: such as [128,128,64]
    :param newSpacing:such as [1,1,1]
    :param resamplemethod: sitk.sitkBSpline;sitk.sitkNearestNeighbor;
    :return:
    """
    originSize = itkimage.GetSize()
    originSpcaing = itkimage.GetSpacing()

    if newSpacing != None:
        newSpacing = np.array(newSpacing, float)
        newSize = originSpcaing / newSpacing * originSize
        newSize = np.round(newSize)
        newSpacing = originSpcaing / newSize * originSize

    elif newSize != None:
        newSize = np.array(newSize, float)
        newSpacing = originSize / newSize * originSpcaing

    newSize = newSize.astype(np.int)
    # 运行有问题，弃用
    if N4BiasCorrect:
        mask_image = sitk.OtsuThreshold(itkimage, 0, 1, 200)
        itkimage = sitk.Cast(itkimage, sitk.sitkFloat32)  # 数据类型转换
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        itkimage = corrector.Execute(itkimage, mask_image)  # N4 错误
        # log_bias_field = corrector.GetBiasFieldFullWidthAtHalfMaximum()
        itkimage = sitk.Cast(itkimage, sitk.sitkInt16)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itkimage)  # 将输出的大小、原点、间距和方向设置为itkimage
    resampler.SetSize(newSize.tolist())  # 设置输出图像大小
    resampler.SetOutputSpacing(newSpacing.tolist())  # 设置输出图像间距
    if newOrigin:
        resampler.SetOutputOrigin(newOrigin)
    if newDirection:
        resampler.SetOutputDirection(newDirection)
    # if resamplemethod == sitk.sitkNearestNeighbor:
    #     resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    # else:
    #     resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)

    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    # 使用N4BiasFieldCorrection校正MRI图像的偏置场
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)



