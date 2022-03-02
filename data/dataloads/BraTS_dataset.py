from data.dataloads.base_dataset import BaseDataset
import os
import random
import numpy as np

import torchvision.transforms as transforms
from data.transforms.transformOnArray import normalize, NormalizeRange, get_transform
from data.transforms.transforms import ToArray


def get_braTS_path(dataroot, data_phase):
    # if istrain:
    #     A_root = os.path.join(dataroot, 'trainA')
    #     B_root = os.path.join(dataroot, 'trainB')
    # else:
    #     A_root = os.path.join(dataroot, 'testA')
    #     B_root = os.path.join(dataroot, 'testB')
    A_root = os.path.join(dataroot, data_phase+'A')
    B_root = os.path.join(dataroot, data_phase+'B')
    A_paths = [os.path.join(A_root, path) for path in os.listdir(A_root) if path.endswith('npy')]
    B_paths = [os.path.join(B_root, path) for path in os.listdir(B_root) if path.endswith('npy')]
    return A_paths, B_paths


def npy_bra_loader(path, num=-1):
    # the last dim is channel
    img = np.load(path)
    if num != -1 and num < img.shape[-1]:
        return img[..., num]
    else:
        return img

# dataset-specific, you also can take it to loader
def get_pre_transform():
    transform_list = []
    transform_list.append(NormalizeRange(dtype=np.float32))
    transform_list.append(transforms.ToPILImage())
    return transforms.Compose(transform_list)


def get_post_transform():
    transform_list = []
    transform_list.append(ToArray(normalize=False))
    transform_list.append(NormalizeRange(dtype=np.float32))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)


class BratsDataset(BaseDataset):
    def __init__(self, opt, loader=npy_bra_loader):
        # save the option and dataset root
        super(BratsDataset, self).__init__(opt)
        # get the image paths of your dataset;
        self.A_paths, self.B_paths = get_braTS_path(opt.dataroot, opt.phase)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # adjust the true 2D=============================================
        self.A_paths_new = []
        self.B_paths_new = []
        for i in range(10):  # total 10 slice on each volume
            tmpA_paths = tuple(zip(self.A_paths, [i]*self.A_size))
            tmpB_paths = tuple(zip(self.B_paths, [i]*self.B_size))
            self.A_paths_new += tmpA_paths
            self.B_paths_new += tmpB_paths
        #  ============================================================
        self.A_paths = self.A_paths_new
        self.B_paths = self.B_paths_new
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.loader = loader
        self.pre_transform = get_pre_transform()
        self.transform = get_transform(opt)
        self.post_transform = get_post_transform()

    def __getitem__(self, index):
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index_B]
        A_img = self.loader(*A_path)
        B_img = self.loader(*B_path)
        if self.pre_transform:
            A_img = self.pre_transform(A_img)
            B_img = self.pre_transform(B_img)
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        if self.post_transform:
            A_img = self.post_transform(A_img)
            B_img = self.post_transform(B_img)
        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images."""
        return max(self.A_size, self.B_size)
