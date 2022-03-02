import torch
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod
from ..utils_data import get_pad_image, get_flip_volumes
from data.utils_data import nii_loader, h5_loader


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


class CustomDataset(BaseDataset):

    def __init__(self, opt):
        super(CustomDataset, self).__init__(opt)
        self.paths = []  # should be [{'volume':volume,'label':label}, ...]
        self.data_size = len(self.paths)
        self.loader = None

        self.pre_transform = None
        self.transform = None
        self.post_transform = None

    def __getitem__(self, index):
        pass

    def _get_volume_label_array(self, index_used):
        pass

    def __len__(self):
        """Return the total number of images."""
        return self.data_size

    def _get_used_index(self, index):
        if self.opt.serial_batches:  # make sure index is within then range
            index_used = index % self.data_size
        else:
            index_used = self.opt.random_state.randint(0, self.data_size - 1)
        return index_used

    # 进行形状变换前的对volume进行的一些特殊处理,目前为空
    def _apply_pre_transform(self, volume):
        if self.pre_transform:
            volume = self.pre_transform(volume)
        return volume

    # 同时对volume和label进行的一些处理，主要包括，旋转、放缩、剪切，镜像，通道变换等
    def _apply_transform(self, volume, label):
        if self.transform:
            # # print(volume.shape)
            # # print(volume_path)
            # # print(label.shape)
            # # print(label_path)
            # volume_label = np.stack([volume, label], axis=0)    # array
            # volume_label = self.transform(volume_label)         # tensor
            # volume, label = volume_label[:-1, ...], volume_label[-1:, ...]
            # # label = torch.unsqueeze(label, dim=0)
            volume, label = self.transform(volume, label)
        return volume, label

    # 单独对volume做的一些处理，主要包括亮度、对比度、噪声变换等
    def _apply_post_transform(self, volume):
        if self.post_transform:
            volume = self.post_transform(volume)
        return volume


class NIIDataset(CustomDataset):
    def __init__(self, opt):
        super(NIIDataset, self).__init__(opt)
        self.loader = nii_loader


class H5Dataset(CustomDataset):
    def __init__(self, opt):
        super(H5Dataset, self).__init__(opt)
        self.loader = h5_loader


class TestOnePatientDataset(data.Dataset):
    axis_database = ((0,),
                     ((0,), (1,)),
                     ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)),
                     )

    def __init__(self, origin_volume, opt, **pad_kwargs):
        '''
        :param origin_volume:  CDHW\DHW\HW
        :param opt:
            crop_size: WHD
            stride: WHD
        :param pad_kwargs:
        '''
        super(TestOnePatientDataset, self).__init__()

        # Check whether it can be executed
        assert origin_volume.ndim in [2, 3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        ndim = origin_volume.ndim
        crop_size = opt.crop_size
        stride = opt.stride
        if isinstance(crop_size, int):
            crop_size = (crop_size,) * ndim
        if isinstance(stride, int):
            stride = (stride,) * ndim
        assert len(crop_size) == ndim and len(stride) == ndim

        # save parameters
        self.opt = opt
        self.origin_volume = origin_volume
        self.origin_size = origin_volume.shape      # # DHW, zyx
        self.pad_volume = get_pad_image(origin_volume, opt.crop_size, opt.stride, mode='minimum', **pad_kwargs)
        self.padded_size = self.pad_volume.shape

        self.stride = stride[::-1]      # 这里注意要转化为实际顺序
        self.crop_size = crop_size[::-1]
        if ndim == 4:
            # # 4D数据，不pad通道维,需修正stride和crop_size，影响后续计算crop_num
            self.stride[0] = 1
            self.crop_size[0] = 1

        self.crop_num = self.get_crop_num()         # N

        if opt.no_augment:
            self.axis = ()                          # C
        else:
            self.axis = TestOnePatientDataset.axis_database[ndim-1]

        self.coordinate = [((slice(0, 0, None),) * ndim), ]*self.crop_num
        # save coordinate
        if ndim == 2:
            c_h, c_w = self.crop_size
            s_h, s_w = self.stride
            h_num, w_num = self.get_crop_num_list()
            assert h_num * w_num == self.crop_num
            for i in range(0, h_num):
                for j in range(0, w_num):
                    self.coordinate[i*w_num+j] = slice(i*s_h, i*s_h+c_h, None), \
                                                 slice(j*s_w, j*s_w+c_w, None),
        elif ndim == 3:
            c_d, c_h, c_w = self.crop_size
            s_d, s_h, s_w = self.stride
            d_num, h_num, w_num = self.get_crop_num_list()
            assert d_num*h_num*w_num == self.crop_num
            for i in range(d_num):
                for j in range(h_num):
                    for k in range(w_num):
                        self.coordinate[i*h_num*w_num+j*w_num+k] = slice(i*s_d, i*s_d+c_d, None), \
                                                                   slice(j*s_h, j*s_h+c_h, None), \
                                                                   slice(k*s_w, k*s_w+c_w, None),
        else:
            c_c, c_d, c_h, c_w = self.crop_size
            s_c, s_d, s_h, s_w = self.stride
            c_num, d_num, h_num, w_num = self.get_crop_num_list()
            assert c_num == self.padded_size[0]
            assert c_num*d_num*h_num*w_num == self.crop_num
            for c in range(c_num):
                for i in range(d_num):
                    for j in range(h_num):
                        for k in range(w_num):
                            self.coordinate[c*d_num*h_num*w_num+i*h_num*w_num+j*w_num+k] = slice(c*s_c, c*s_c+c_c),\
                                                                                            slice(i+s_d, i+s_d+c_d),\
                                                                                            slice(j+s_h, j+s_h+c_h),\
                                                                                            slice(k+s_w, k+s_w+c_w),

    def __getitem__(self, item):
        # 这里有个需要注意的地方，flip是对子块做的
        aug_volume = get_flip_volumes(self.pad_volume[self.coordinate[item]], self.axis)      # C DHW
        volume_out = torch.from_numpy(aug_volume.astype(dtype=np.float32))
        return volume_out

    def __len__(self):
        return self.crop_num

    def get_crop_num(self):
        num = 1
        for i, j, k in zip(self.padded_size, self.crop_size, self.stride):
            num *= ((i-j)/k + 1)
        # assert
        return int(num)

    def get_info(self):
        return {'crop_size': self.crop_size, 'stride': self.stride,
                'origin_shape': self.origin_size, 'pad_shape': self.padded_size}

    def get_volume(self):
        return {'origin_volume': self.origin_volume, 'pad_volume': self.pad_volume}

    def get_crop_num_list(self):
        return [int((i-j)/k + 1) for i, j, k in zip(self.padded_size, self.crop_size, self.stride)]

    def get_axis(self):
        return self.axis
