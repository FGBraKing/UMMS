import cv2
import numpy as np
import nibabel as nib

from PIL import Image
from math import ceil
from skimage import measure
from matplotlib import pyplot as plt
from nibabel.viewers import OrthoSlicer3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from data.transforms.transformOnArray import normalize, NormalizeRange

to_std_image_uint8 = NormalizeRange(0, 255, np.uint8)
to_std_image_float32 = NormalizeRange(0, 1, np.float32)


# fig = plt.figure()  # an empty figure with no Axes
# fig, ax = plt.subplots()  # a figure with a single Axes
# fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
def read_nii(img_path):
    img = nib.load(img_path)
    img_array = img.get_data()  # W H D
    return img_array


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def show_niiimg(path):
    volume_nii = nib.load(path)
    weight, height, queue = volume_nii.dataobj.shape
    OrthoSlicer3D(volume_nii.dataobj).show()
    fig = plt.figure()
    x = int((queue / 10) ** 0.5) + 1
    num = 1
    for i in range(0, queue, 10):
        volume_arr = volume_nii.dataobj[:, :, i]
        ax = plt.subplot(x, x, num)
        ax.imshow(volume_arr)
        num += 1
    plt.show()


#  figure axes axis tick
def show_image(image, num=None, figsize=None, cmap='gray', title=None):
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    fig = plt.figure(num=num, figsize=figsize)
    # fig.suptitle('test')
    # ax = fig.add_axes([0, 0, 1, 1])
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.show()
    # plt.figure()
    # if title:
    #     plt.title(title)
    # plt.imshow(img, cmap=cmap)
    # plt.show()


def show_image_label(image, label, num=None, figsize=None, cmap='gray', title=None, concat=True):
    assert label.ndim == 2, "the dim of the gray image must be 2"
    assert image.shape == label.shape
    if concat:
        label = label.astype(image.dtype)
        # np.max(image)<0.5 的情况要注意
        index = label > 0.5
        label[index] = np.max(image)
        label[~index] = np.min(image)
        image_label = np.concatenate([image, label], axis=1)
        show_image(image_label, num=num, figsize=figsize, cmap=cmap, title=title)
    else:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title)
        ax[0].imshow(image, cmap=cmap)
        ax[0].set_title('image', fontsize=4, color='r')
        ax[0].axis('off')
        ax[1].imshow(label, cmap=cmap)
        ax[1].set_title('label', fontsize=4, color='r')
        ax[1].axis('off')
        plt.show()


def show_volume_label(volume, label, interval=1, add_line=False, **kwargs):
    # kwargs: max_num\fix_num\normalize_per\title\col\row
    assert label.ndim == 3, "the dim of the gray volume must be 3 of D H W"
    assert volume.shape == label.shape
    label = label.astype(volume.dtype)
    # np.max(image)<0.5 的情况要注意
    index = label > 0.5
    label[index] = np.max(volume)
    label[~index] = np.min(volume)
    if add_line:
        med_line = np.ones(shape=label.shape[:-1], dtype=volume.dtype) * np.max(volume)
        med_line = med_line[..., np.newaxis]
        volume_label = np.concatenate([volume, med_line, label], axis=-1)
    else:
        volume_label = np.concatenate([volume, label], axis=-1)

    show_array_3d(volume_label[::interval, ...], **kwargs)


def show_volume_label_predict(volume, label, predict, interval=1, add_line=False, **kwargs):
    assert label.ndim == 3, "the dim of the gray volume must be 3 of D H W"
    assert volume.shape == label.shape == predict.shape

    volume = to_std_image_float32(volume)
    label = to_std_image_float32(label)
    predict = to_std_image_float32(predict)
    # label = label.astype(volume.dtype)
    # predict = predict.astype(volume.dtype)
    #
    # label = np.where(label > 0.5, np.max(volume), np.min(volume))
    # predict = np.where(predict > 0.5, np.max(volume), np.min(volume))

    if add_line:
        med_line = np.ones(shape=label.shape[:-1], dtype=volume.dtype) * np.max(volume)
        med_line = med_line[..., np.newaxis]
        volume_label_predict = np.concatenate([volume, med_line, label, med_line, predict], axis=-1)
    else:
        volume_label_predict = np.concatenate([volume, label, predict], axis=-1)
    show_array_3d(volume_label_predict[::interval, ...], **kwargs)


def show_array_3d(array, row=5, col=5, title='number', normalize_per=False, fix_num=False, max_num=40, fig_list=[]):
    assert array.ndim == 3, "the array'dim is not 3"
    channel = array.shape[0]   # D H W
    total = row * col
    # current_num = len(plt.get_fignums())

    for i in range(ceil(channel/total)):
        if total == 1:
            show_image(array[i, :, :], num=i, title=f'{i}')
            return

        if fix_num:
            fig = fig_list[max_num - i - 1]
            ax = fig.subplots(nrows=row, ncols=col)
        else:
            fig, ax = plt.subplots(row, col)

        # fig, ax = plt.subplots(row, col, num=num)
        fig.suptitle(f'{title}:{i+1:2d}')
        for j in range(row):
            for k in range(col):
                if i * total + j * col + k < channel:
                    data = array[i * total + j * col + k, :, :]
                    if normalize_per:
                        # from data.transforms.transformOnArray import normalize
                        data = normalize(data)
                    ax[j][k].imshow(data, cmap='gray')  # f'{i * 25 + j * 5 + k}'
                    ax[j][k].set_title(f'{i * total + j * col + k + 1}', fontsize=5, color='r')
                    ax[j][k].axis('off')
                    # ax[j][k].xlabel('x')
                    ax[j][k].set(xlabel='x', ylabel='y')
        fig.show()
    # plt.show()


def plot_2d(x, y, fig_title=None, ax_title=None, x_label=None, y_label=None, *args, **kwargs):
    # 主要是要设置线、坐标轴、刻度、注释
    # color = ['b','g','r','c','m','y','k','w']
    # linestyle = ['-','--','-.',':']
    # marker=['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_','.',',']
    # linewidth=2

    # color="blue",linewidth=20,marker="o",markersize=50,
    # markerfacecolor="red",markeredgewidth=6,markeredgecolor="grey"

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    # 640x480
    if fig_title:
        fig.suptitle(fig_title, fontsize=14, fontweight='bold')
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


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)  #将扫描件竖直放置
    verts, faces = measure.marching_cubes(p, threshold) #Liner推进立方体算法来查找3D体积数据中的曲面。
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)  #创建3Dpoly
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)  #设置颜色
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def show_array_histogram(array, bins=1000, bin_low=None, bin_high=None, title='gray histogram'):
    if bin_low is None:
        bin_low = np.min(array)
    if bin_high is None:
        bin_high = np.max(array)
    # hist, bins_edges = np.histogram(array, bins=bins, range=(bin_low, bin_high))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    ax.hist(array.ravel(), bins=bins, range=(bin_low, bin_high), density=False, color="blue")
    plt.show()


def show_paired_image(image1, image2, title1='image1', title2='image2', cmap1='gray', cmap2='gray'):
    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    # fig.subplots_adjust(hspace=0.3, wspace=0.3)
    # for ax in axes.flat
    # ax.set_xlabel('gfdg')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # 1*2或2*1的索引都是1维的
    fig.suptitle('paired_image')
    ax[0].imshow(image1, cmap=cmap1)
    ax[0].set_title(title1, fontsize=4, color='r')
    ax[0].axis('off')
    ax[1].imshow(image2, cmap=cmap2)
    ax[1].set_title(title2, fontsize=4, color='r')
    ax[1].axis('off')
    plt.show()


def show_pired_histogram(image1, image2, bins=1000, bin_low=None, bin_high=None, title='gray histogram pired'):
    if isinstance(bins, int):
        bins = (bins, bins)
    if bin_low is None:
        bin_low = (np.min(image1), np.min(image2))
    if bin_high is None:
        bin_high = (np.max(image1), np.max(image2))
    title_list = ['image1', 'image2']
    fig, axs = plt.subplots(1, 2, figsize=(6, 4))
    fig.suptitle(title)
    try:
        for ax, image, bin, b_low, b_high, title_image in zip(axs, (image1, image2), bins, bin_low, bin_high, title_list):
            ax.hist(image.ravel(), bins=bin, range=(b_low, b_high), density=False, color="blue")
            ax.set_title(title_image)
    except Exception as e:
        print('something wrong happened on iteration:', e)
    finally:
        plt.show()


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam/cam_{}.jpg".format(name), np.uint8(255 * cam))


# https://github.com/faustomilletari/VNet/blob/master/utilities.py
def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    for k in range(0, nda.shape[2]):
        print("printing slice "+str(k))
        ax.imshow(np.squeeze(nda[:,:,k]),extent=extent,interpolation=None)
        plt.draw()
        plt.pause(0.1)
        # plt.waitforbuttonpress()

