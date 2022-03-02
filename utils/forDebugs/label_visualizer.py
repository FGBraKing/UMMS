import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from data.utils_data import h5_loader
from utils.others.utils import print_numpy
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict, show_image
from data.transforms.transformOnArray import normalize, NormalizeRange
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


to_std_image_uint8 = NormalizeRange(0, 255, np.uint8)
to_std_image_float32 = NormalizeRange(0, 255, np.float32)


def find_contours(gt_path, out_path):
    gt = cv2.imread(gt_path)
    output = cv2.imread(out_path)

    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gt_gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

    print('number of the contours found = ' + str(len(contours)))
    cv2.imshow('Contours', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_outline(img, mask, color=(0, 0, 255)):
    coef = 255 if np.max(img) < 3 else 1
    image = (img * coef).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.RETR_EXTERNAL表示图像的外轮廓
    # binary, contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 1)
    # cv2.imwrite('test.png', image)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


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


def show_data_in_h5(data_path, key_volume, key_segment, key_label, interval=2):
    data_name = os.path.basename(data_path).split('.')[0]
    volume, segment, label = h5_loader(data_path, key_volume, key_segment, key_label)
    # volume, segment, label = volume.transpose(1, 0, 2), segment.transpose(1,0,2), label.transpose(1,0,2)
    show_volume_label_predict(volume, segment, label,
                              interval=interval, add_line=True, normalize_per=False,
                              row=3, col=2, title=data_name)
    # .transpose(1,0,2)
    # volume = to_std_image_float32(volume)
    # verts, faces, normals, values = measure.marching_cubes(volume, mask=segment.astype(np.bool))
    i = 0
    for img, seg, lab in zip(volume[::interval, ...], segment[::interval, ...], label[::interval, ...]):
        i += 1
        draw_mask_predict_edge_on_image_skimage(img, seg, lab, title=f'test edge{i}')


def visulaizing_training_result(data_path):
    pass


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


if __name__ == "__main__":

    result_dir = r'/home/lf/data_fong/CODE/PycharmProject/DLForPytorch/traces/results'
    exp_name = r'mrusmr_unet3dV1_969632_bs6_ch32_kaiming_combo_1.0_1.0_adam_2e-4_cosine_1.0_0.3_2x500_warmup_10_1e-5'
    phase_name = r'mrval'
    process_name = r'crop969632_slide24248_nopad_noaug'
    data_name = r'579_neP069_MR_image.h5'

    d_path = os.path.join(result_dir, exp_name, phase_name, process_name, data_name)
    # # 'origin_volume', 'segment', 'label'
    # 'label', 'origin_volume', 'segment'
    # it seems that label is the predicted
    show_data_in_h5(d_path, 'origin_volume', 'segment', 'label', interval=2)
    # print_numpy(volume,shp=True, percentile=True)
    # shape, (175, 224, 224)
    # mean = -0.000, min = -0.904, max = 3.678, median = -0.506, std=1.000
    # percentile_99_5 = 3.315, percentile_00_5 = -0.904
    # print_numpy(segment,shp=True, percentile=True)
    # shape, (175, 224, 224)
    # mean = 0.149, min = 0.000, max = 1.000, median = 0.000, std=0.356
    # percentile_99_5 = 1.000, percentile_00_5 = 0.000
    # print_numpy(label,shp=True, percentile=True)
    # shape, (175, 224, 224)
    # mean = 0.156, min = 0.000, max = 1.000, median = 0.000, std=0.363
    # percentile_99_5 = 1.000, percentile_00_5 = 0.000
