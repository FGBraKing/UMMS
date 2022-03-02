#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from multiprocessing import Pool

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def verify_all_same_orientation(folder):
    """
    This should run after cropping
    :param folder:
    :return:
    """
    nii_files = subfiles(folder, suffix=".nii.gz", join=True)
    orientations = []
    for n in nii_files:
        img = nib.load(n)
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        orientations.append(orientation)
    # now we need to check whether they are all the same
    orientations = np.array(orientations)
    unique_orientations = np.unique(orientations, axis=0)
    all_same = len(unique_orientations) == 1
    return all_same, unique_orientations


def verify_same_geometry(img_1: sitk.Image, img_2: sitk.Image):
    ori1, spacing1, direction1, size1 = img_1.GetOrigin(), img_1.GetSpacing(), img_1.GetDirection(), img_1.GetSize()
    ori2, spacing2, direction2, size2 = img_2.GetOrigin(), img_2.GetSpacing(), img_2.GetDirection(), img_2.GetSize()

    same_ori = np.all(np.isclose(ori1, ori2))
    if not same_ori:
        print("the origin does not match between the images:")
        print(ori1)
        print(ori2)

    same_spac = np.all(np.isclose(spacing1, spacing2))
    if not same_spac:
        print("the spacing does not match between the images")
        print(spacing1)
        print(spacing2)

    same_dir = np.all(np.isclose(direction1, direction2))
    if not same_dir:
        print("the direction does not match between the images")
        print(direction1)
        print(direction2)

    same_size = np.all(np.isclose(size1, size2))
    if not same_size:
        print("the size does not match between the images")
        print(size1)
        print(size2)

    if same_ori and same_spac and same_dir and same_size:
        return True
    else:
        return False


def verify_contains_only_expected_labels(itk_img: str, valid_labels: (tuple, list)):
    img_npy = sitk.GetArrayFromImage(sitk.ReadImage(itk_img))
    uniques = np.unique(img_npy)
    invalid_uniques = [i for i in uniques if i not in valid_labels]
    if len(invalid_uniques) == 0:
        r = True
    else:
        r = False
    return r, invalid_uniques


def reorient_to_RAS(img_fname: str, output_fname: str = None):
    img = nib.load(img_fname)
    canonical_img = nib.as_closest_canonical(img)
    if output_fname is None:
        output_fname = img_fname
    nib.save(canonical_img, output_fname)


if __name__ == "__main__":
    # investigate geometry issues
    import SimpleITK as sitk

    # load image
    gt_itk = sitk.ReadImage(
        "/media/fabian/Results/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/case_00085.nii.gz")

    # get numpy array
    pred_npy = sitk.GetArrayFromImage(gt_itk)

    # create new image from numpy array
    prek_itk_new = sitk.GetImageFromArray(pred_npy)
    # copy geometry
    prek_itk_new.CopyInformation(gt_itk)
    # prek_itk_new = copy_geometry(prek_itk_new, gt_itk)

    # save
    sitk.WriteImage(prek_itk_new, "test.mnc")

    # load images in nib
    gt = nib.load(
        "/media/fabian/Results/nnUNet/3d_fullres/Task064_KiTS_labelsFixed/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/case_00085.nii.gz")
    pred_nib = nib.load("test.mnc")

    new_img_sitk = sitk.ReadImage("test.mnc")

    np1 = sitk.GetArrayFromImage(gt_itk)
    np2 = sitk.GetArrayFromImage(prek_itk_new)
