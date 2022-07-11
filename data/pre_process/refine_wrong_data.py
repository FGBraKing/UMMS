import os
import re
import random
import shutil
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from utils.others.img_io import show_volume_label, show_image_label
from data.utils_data import nii_loader, save_nii, print_data_describe


def main():
    src_mr_image_path = r'/home/lf/data_fong/DATA/MR-US20/P105/P105_MR.nii'
    src_mr_label_path = r'/home/lf/data_fong/DATA/MR-US20/P105/P105_MR_Prostate.nii'
    src_us_image_path = r'/home/lf/data_fong/DATA/MR-US20/P105/P105_US.nii'
    src_us_label_path = r'/home/lf/data_fong/DATA/MR-US20/P105/P105_US_Prostate.nii'

    substitute_us_image_path = r'/home/lf/data_fong/DATA/MR-US20/P105/P084_image.nii'
    substitute_us_label_path = r'/home/lf/data_fong/DATA/MR-US20/P105/P084_label.nii'

    sub_us_image = sitk.ReadImage(substitute_us_image_path)
    sub_us_image.CopyInformation(sitk.ReadImage(src_us_image_path))
    sitk.WriteImage(sub_us_image, substitute_us_image_path)

    sub_us_label = sitk.ReadImage(substitute_us_label_path)
    sub_us_label.CopyInformation(sitk.ReadImage(src_us_label_path))
    sitk.WriteImage(sub_us_label, substitute_us_label_path)

    # s_us_image = nii_loader(src_us_image_path)      # DHW,RAI
    # s_us_label = nii_loader(src_us_label_path)
    #
    # t_us_image = nii_loader(substitute_us_image_path)   # DHW, I
    # t_us_label = nii_loader(substitute_us_label_path)
    #
    # # t_us_image = np.flip(t_us_image, axis=(1,))
    # # t_us_label = np.flip(t_us_label, axis=(1,))
    # show_image_label(t_us_image[175, :, :], t_us_label[175, :, :], title='sub-P084 z')
    # show_image_label(t_us_image[:, 224, :], t_us_label[:, 224, :], title='sub-P084 y')
    # show_image_label(t_us_image[:, :, 224], t_us_label[:, :, 224], title='sub-P084 x')
    #
    # show_image_label(s_us_image[175, :, :], s_us_label[175, :, :], title='sub-P105 z')
    # show_image_label(s_us_image[:, 224, :], s_us_label[:, 224, :], title='sub-P105 y')
    # show_image_label(s_us_image[:, :, 224], s_us_label[:, :, 224], title='sub-P105 x')


if __name__ == "__main__":
    main()












