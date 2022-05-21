import os
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from data.utils_data import h5_loader
from utils.others.utils import print_numpy
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict, show_image
from data.transforms.transformOnArray import normalize, NormalizeRange
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


to_std_image_uint8 = NormalizeRange(0, 255, np.uint8)
to_std_image_float32 = NormalizeRange(0, 255, np.float32)


def draw_mask_edge_on_image_skimage(image, mask, color=(0, 0, 255), save_path=None, title='image'):
    image = to_std_image_float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # image = np.stack([image, image, image], axis=2)

    contours = measure.find_contours(mask)  # , level=0.5

    for c in contours:
        c = np.around(c).astype(np.int)
        image[c[:, 0], c[:, 1]] = np.array(color)
    image = image.astype(np.uint8)   # 或者 image = image/255
    if save_path is not None and isinstance(save_path, str):
        cv2.imwrite(save_path, image)
    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    show_image(image, cmap=None, title=title)


def draw_mask_predict_edge_on_image_skimage(image, label, segment, label_color=(0, 0, 255), seg_color=(255, 0, 0),
                                            save_path=None, title='image'):
    # blue and red
    image = to_std_image_float32(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # image = np.stack([image, image, image], axis=2)

    contours_label = measure.find_contours(label)   # , level=0.5
    contours_segment = measure.find_contours(segment)

    for c in contours_label:
        c = np.around(c).astype(np.int)
        image[c[:, 0], c[:, 1]] = np.array(label_color)

    for c in contours_segment:
        c = np.around(c).astype(np.int)
        image[c[:, 0], c[:, 1]] = np.array(seg_color)

    image = image.astype(np.uint8)   # 或者 image = image/255
    if save_path is not None and isinstance(save_path, str):
        cv2.imwrite(save_path, image)
    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    show_image(image, cmap=None, title=title)


def visulaizing_training_result(data_path, key_volume, key_label, key_segment, number=None, interval=2):
    def inner_show_result(i_vol, i_lab, i_seg, title):
        show_volume_label_predict(i_vol, i_lab, i_seg,
                                  interval=interval, add_line=True, normalize_per=False,
                                  row=3, col=2, title=title)
        i = 0
        for img_sub, lab_sub, seg_sub in zip(i_vol[::interval, ...], i_lab[::interval, ...], i_seg[::interval, ...]):
            i += 1
            draw_mask_predict_edge_on_image_skimage(img_sub, lab_sub, seg_sub, title=f'test edge{i}')

    data_name = os.path.basename(data_path).split('.')[0]
    volume, label, segment = h5_loader(data_path, key_volume, key_label, key_segment)
    if volume.ndim == 3:
        inner_show_result(volume, label, segment, data_name)
    elif number is not None:
        vol = volume[number, 0, ...]
        lab = label[number, 0, ...]
        seg = segment[number, 0, ...]
        inner_show_result(vol, lab, seg, data_name)
    else:
        for vol, seg, lab in zip(volume, segment, label):
            vol = vol[0, ...]
            lab = lab[0, ...]
            seg = seg[0, ...]
            inner_show_result(vol, lab, seg, data_name)


if __name__ == "__main__":
    logs_dir = r'/home/lf/raid_lf/PROJECT/UMMS/traces/logs'
    exp_name = r'mrusmr128_fold0_patch_bs8_unet3d_ch16_combo_1_1_1.5_adam_2e-4_poly_3x300_0.6'
    #
    train_train_data = os.path.join(logs_dir, exp_name, 'visuals', 'latestvisuals.h5')
    train_test_data = os.path.join(logs_dir, exp_name, 'visuals',  'latest-testvisuals.h5')
    slide_test_data = os.path.join(logs_dir, exp_name, 'visuals',  'latest-slidevisuals.h5')

    visulaizing_training_result(slide_test_data, 'volume', 'label', 'predict', number=0, interval=2)

    # print_numpy(volume, shp=True, percentile=True)
    # # shape, (6, 1, 32, 96, 96)
    # # mean = 0.394, min = -1.922, max = 7.853, median = 0.077, std=1.003
    # # percentile_99_5 = 4.307, percentile_00_5 = -0.899
    # print_numpy(segment, shp=True, percentile=True)
    # # shape, (6, 1, 32, 96, 96)
    # # mean = 0.280, min = 0.005, max = 1.000, median = 0.016, std=0.433
    # # percentile_99_5 = 1.000, percentile_00_5 = 0.009
    # print_numpy(label, shp=True, percentile=True)
    # # shape, (6, 1, 32, 96, 96)
    # # mean = 0.266, min = 0.000, max = 1.000, median = 0.000, std=0.442
    # # percentile_99_5 = 1.000, percentile_00_5 = 0.000
    # print_numpy(testvolume, shp=True, percentile=True)
    # # shape, (4, 1, 32, 96, 96)
    # # mean = 0.577, min = -1.440, max = 9.907, median = 0.256, std=1.098
    # # percentile_99_5 = 4.822, percentile_00_5 = -0.788
    # print_numpy(testsegment, shp=True, percentile=True)
    # # shape, (4, 1, 32, 96, 96)
    # # mean = 0.296, min = 0.007, max = 1.000, median = 0.021, std=0.434
    # # percentile_99_5 = 1.000, percentile_00_5 = 0.012
    # print_numpy(testlabel, shp=True, percentile=True)
    # # shape, (4, 1, 32, 96, 96)
    # # mean = 0.306, min = 0.000, max = 1.000, median = 0.000, std=0.461
    # # percentile_99_5 = 1.000, percentile_00_5 = 0.000
