import os
import h5py
import pickle
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image


# dataloader
def npy_loader(path, *args, **kwargs):
    img = np.load(path)
    # img = normalize(img, 0, 255)
    # return Image.fromarray(img.astype('u1'), mode='L')
    return img


def h5_loader(path, *args, **kwargs):
    output = []
    with h5py.File(path, mode='r') as fd_read:
        print(fd_read.keys())
        for name in args:
            if isinstance(name, str):
                try:
                    data = fd_read.get(name=name)[:]
                except Exception as e:
                    print('something wrong:%s' % e)
                    print(f'there is no {name} on file')
                else:
                    output.append(data)
    return output


def img_loader(path, *args, **kwargs):
    return Image.open(path, mode='r')


def nii_loader(path, *args, **kwargs):
    '''
    :param path:
    :param num: int=-1
    :return:
    '''
    if 'num' in kwargs:
        num = kwargs['num']
    else:
        num = -1
    # 读取nii的某一层或全部数据
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))  # D H W
    except Exception as e:
        print('some wrong hanpped on SimpleItk:{}. then will try to use nibabel'.format(e))
        img = nib.load(path).get_data()  # W H D
        img = np.transpose(img, axes=(2, 1, 0))     # D H W
    if num != -1 and num < img.shape[-1]:
        return img[num, ...]
    else:
        return img


# save data
def save_npy(data, save_dir, save_name):
    path = os.path.join(save_dir, save_name+'.npy')
    np.save(path, data)


def save_nii(save_path, img_data, origin=None, direction=None, spacing=None):
    image = sitk.GetImageFromArray(img_data)
    if spacing:
        image.SetSpacing(spacing)
    if origin:
        image.SetOrigin(origin)
    if direction:
        image.SetDirection(direction)
    sitk.WriteImage(image, save_path)


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)


# for slide windows and combine windows
def get_full_length(w, c, s):
    # simple claculate: (w-c)+s-(w-c)%s
    ''' get a length which more than w, and reduce c divide by s == 0 , (l-c)%s==0, l>=w
    :param w: length
    :param c: crop_size
    :param s: stride
    :return:
    '''
    i = 0
    while (w+i-c) % s != 0:
        i += 1
    return w+i


def get_pad_image(m, crop_size, stride, common_order=True, mode='edge', **kwargs):   # minimum
    '''
    :param m:              c, d, h,w
    :param crop_size:            w,h
    :param stride:
    :param common_order:
    :param mode:   using in np.pad
    :param kwargs: using in np.pad
    :return:      when image is 4D, the channels is not padded defaultly
    '''
    assert m.ndim in [2, 3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
    ndim = m.ndim
    if isinstance(crop_size, int):
        crop_size = (crop_size,) * ndim
    if isinstance(stride, int):
        stride = (stride,) * ndim
    assert len(crop_size) == ndim and len(stride) == ndim
    if common_order:
        crop_size = crop_size[::-1]
        stride = stride[::-1]
    if ndim == 2:
        h, w = m.shape
        c_h, c_w = crop_size
        s_h, s_w = stride
        h_pad = get_full_length(h, c_h, s_h) - h
        w_pad = get_full_length(w, c_w, s_w) - w
        m_pad = np.pad(m, pad_width=[[int(np.floor(h_pad/2)), int(np.ceil(h_pad/2))],
                                     [int(np.floor(w_pad/2)), int(np.ceil(w_pad/2))]], mode=mode, **kwargs)
    elif ndim == 3:
        d, h, w = m.shape
        c_d, c_h, c_w = crop_size
        s_d, s_h, s_w = stride
        d_pad = get_full_length(d, c_d, s_d) - d
        h_pad = get_full_length(h, c_h, s_h) - h
        w_pad = get_full_length(w, c_w, s_w) - w
        m_pad = np.pad(m, pad_width=[[int(np.floor(d_pad/2)), int(np.ceil(d_pad/2))],
                                     [int(np.floor(h_pad/2)), int(np.ceil(h_pad/2))],
                                     [int(np.floor(w_pad/2)), int(np.ceil(w_pad/2))]], mode=mode, **kwargs)
    else:
        _, d, h, w = m.shape
        _, c_d, c_h, c_w = crop_size
        _, s_d, s_h, s_w = stride
        d_pad = get_full_length(d, c_d, s_d) - d
        h_pad = get_full_length(h, c_h, s_h) - h
        w_pad = get_full_length(w, c_w, s_w) - w
        m_pad = np.pad(m, pad_width=[[0, 0],
                                     [int(np.floor(d_pad/2)), int(np.ceil(d_pad/2))],
                                     [int(np.floor(h_pad/2)), int(np.ceil(h_pad/2))],
                                     [int(np.floor(w_pad/2)), int(np.ceil(w_pad/2))]], mode=mode, **kwargs)
    return m_pad


def slide_crop(m, crop_size, stride, common_order=True, mode='minimum', **kwargs):
    '''
    :param m: map
    :param crop_size:
    :param stride:
    :param common_order:
    :param mode:
    Traversal priority:
    :return:
    '''
    assert m.ndim in [2, 3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
    ndim = m.ndim
    if isinstance(crop_size, int):
        crop_size = (crop_size,) * ndim
    if isinstance(stride, int):
        stride = (stride,) * ndim
    assert len(crop_size) == ndim and len(stride) == ndim
    if common_order:
        crop_size = crop_size[::-1]
        stride = stride[::-1]
    if ndim == 2:
        h, w = m.shape
        c_h, c_w = crop_size
        s_h, s_w = stride
        h_pad = get_full_length(h, c_h, s_h) - h
        w_pad = get_full_length(w, c_w, s_w) - w
        m_pad = np.pad(m, pad_width=[[int(np.floor(h_pad/2)), int(np.ceil(h_pad/2))],
                                     [int(np.floor(w_pad/2)), int(np.ceil(w_pad/2))]], mode=mode, **kwargs)
        h_new, w_new = m_pad.shape
        assert h_new == h + h_pad
        assert w_new == w + w_pad
        return_list = [m_pad[y:y+c_h, x:x+c_w]
                       for x in range(0, w_new - c_w, s_w)
                       for y in range(0, h_new - c_h, s_h)]
        return np.expand_dims(np.stack(return_list, axis=0), axis=1)  # NCHW,C=1
    elif ndim == 3:
        d, h, w = m.shape
        c_d, c_h, c_w = crop_size
        s_d, s_h, s_w = stride
        d_pad = get_full_length(d, c_d, s_d) - d
        h_pad = get_full_length(h, c_h, s_h) - h
        w_pad = get_full_length(w, c_w, s_w) - w
        m_pad = np.pad(m, pad_width=[[int(np.floor(d_pad/2)), int(np.ceil(d_pad/2))],
                                     [int(np.floor(h_pad/2)), int(np.ceil(h_pad/2))],
                                     [int(np.floor(w_pad/2)), int(np.ceil(w_pad/2))]], mode=mode, **kwargs)
        # , constant_values=0
        d_new, h_new, w_new = m_pad.shape
        assert d_new == d + d_pad
        assert h_new == h + h_pad
        assert w_new == w + w_pad
        return_list = [m_pad[z:z+c_d, y:y+c_h, x:x+c_w]
                       for x in range(0, w_new - c_w, s_w)
                       for y in range(0, h_new - c_h, s_h)
                       for z in range(0, d_new - c_d, s_d)]
        return np.expand_dims(np.stack(return_list, axis=0), axis=1)  # NCDHW,C=1
    else:
        c, d, h, w = m.shape
        _, c_d, c_h, c_w = crop_size
        _, s_d, s_h, s_w = stride
        d_pad = get_full_length(d, c_d, s_d) - d
        h_pad = get_full_length(h, c_h, s_h) - h
        w_pad = get_full_length(w, c_w, s_w) - w
        m_pad = np.pad(m, pad_width=[[0, 0],
                                     [int(np.floor(d_pad/2)), int(np.ceil(d_pad/2))],
                                     [int(np.floor(h_pad/2)), int(np.ceil(h_pad/2))],
                                     [int(np.floor(w_pad/2)), int(np.ceil(w_pad/2))]], mode=mode, **kwargs)
        c_new, d_new, h_new, w_new = m_pad.shape
        assert c_new == c
        assert d_new == d + d_pad
        assert h_new == h + h_pad
        assert w_new == w + w_pad
        return_list = []
        for c in range(c_new):
            arr_list = [m_pad[c, z:z+c_d, y:y+c_h, x:x+c_w]
                        for x in range(0, w_new - c_w, s_w)
                        for y in range(0, h_new - c_h, s_h)
                        for z in range(0, d_new - c_d, s_d)]
            return_list.append(np.stack(arr_list, axis=0))  # [NDHW]
        return np.expand_dims(np.stack(return_list, axis=0), axis=2)  # CN1DHW


def get_flip_volumes(volume, axises=((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))):
    volume_list_origin = [volume]
    volume_list_aug = [np.flip(volume, axis) for axis in axises]
    volume_list = volume_list_origin + volume_list_aug
    return np.stack(volume_list, axis=0)


def combine_filp_mask(masks, axises=((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))):
    '''
    :param masks: list or ndarray, n dhw
    :param axises:
    :return:
    '''
    assert len(masks) == len(axises) + 1
    # power_mask = np.zeros_like(masks[0], dtype=np.int8)
    power_mask = masks[0].astype(np.int8)   # origin mask
    for mask, axis in zip(masks[1:], axises):
        std_mask = np.flip(mask, axis)
        power_mask = power_mask + std_mask.astype(np.int8)
    threshold = len(masks)/2
    power_mask[power_mask <= threshold] = 0
    power_mask[power_mask > threshold] = 1
    return power_mask


def combine_slid_mask(masks, aim_shape, stride=(32, 32, 8)):
    '''
    :param masks:masks to be combine.  masks's order:w, h, d, c
    :param aim_shape:
    :param stride:
    :param combine_rule:
    :return:
    '''
    sub_d, sub_h, sub_w = masks[0].shape
    o_d, o_h, o_w = aim_shape
    s_w, s_h, s_d = stride

    aim_masks = np.zeros((len(masks),)+tuple(aim_shape))
    threshold = np.zeros(aim_shape, dtype=np.float)
    std_ones = np.ones_like(masks[0])

    ind = 0
    for z in range(0, o_d-sub_d, s_d):
        for y in range(0, o_h-sub_h, s_h):
            for x in range(0, o_w - sub_w, s_w):
                aim_masks[ind, z:z+sub_d, y:y+sub_h, x:x+sub_w] = masks[ind]
                threshold[z:z+sub_d, y:y+sub_h, x:x+sub_w] += std_ones
                ind += 1

    threshold = threshold / 2
    tmp_mask = np.sum(aim_masks, axis=0)
    useful_mask = np.where(tmp_mask > threshold, 1, 0)
    return useful_mask


def combine_all_masks(predict, aim_shape, stride=(32, 32, 8), axises=((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))):
    '''
    :param predict:  NCDHW
    :param aim_shape:
    :param stride:
    :param combine_rule:
    :param axises
    :return:
    '''
    n, c, d, h, w = predict.shape
    mask_n_list = []
    for i in range(predict.shape[0]):
        flip_mask = combine_filp_mask(predict[i, ...],
                                      axises=axises)
        mask_n_list.append(flip_mask)
    out_mask = combine_slid_mask(mask_n_list, aim_shape, stride)
    return out_mask  # dhw


def get_unpad_image(now_shape, origin_shape, *data_pad):
    '''
    only support 3 dims's ndarray
    :param now_shape:
    :param origin_shape:
    :param data_pad:
    :return:
    '''

    if len(data_pad) == 1:
        # tmp_mask = np.expand_dims(tmp_mask, axis=0)
        tmp_mask = np.expand_dims(data_pad[0], axis=0)
    else:
        # 似乎跟numpy版本有关，当data_pad长度是1时，有些版本不会添加新的维度
        tmp_mask = np.stack(data_pad, axis=0)     # n d h w

    left_gap_list = [int(np.floor((i-j)/2)) for (i, j) in zip(now_shape, origin_shape)]
    right_gap_list = [int(np.ceil((i-j)/2)) for (i, j) in zip(now_shape, origin_shape)]

    i_axis = 0
    for l_gap, r_gap in zip(left_gap_list, right_gap_list):
        if l_gap > 0 or r_gap > 0:
            del_list = [i for i in range(l_gap)] + [now_shape[i_axis] - j - 1 for j in range(r_gap)]
            tmp_mask = np.delete(tmp_mask, del_list, axis=i_axis+1)
        i_axis += 1
    if len(data_pad) == 1:
        return tmp_mask[0]
    else:
        out_list = list(tmp_mask)
        return out_list


def get_rotate_axes(axes):
    from itertools import combinations, permutations
    axes = np.unique(axes)
    # number = len(axes)
    return list(combinations(axes, 2))



