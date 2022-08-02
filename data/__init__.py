# Copyright (c) 2020, CoolFong. All rights reserved.
# @Time    : 2020/9/10
# @Author  : CoolFong
"""This is the data for project.
dataloads: dataload.
transforms: some function to transform
 """
import logging
import importlib
import torch.utils.data
import torch.distributed
from data.dataloads.base_dataset import BaseDataset
from collections import defaultdict
from types import SimpleNamespace

ddp_logger = logging.getLogger('ddp_logger')


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data.dataloads." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    return dataset


def create_dataset(opt, proxy_two=False):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    if proxy_two:
        data_loader = ProxyDataloader(opt)
    else:
        data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def create_test_dataset(opt):
    test_arg_dict = defaultdict()
    test_arg_dict['fake_shufflt'] = False
    test_arg_dict['custom'] = True
    test_arg_dict['serial_batches'] = True
    test_arg_dict['dataroot'] = opt.dataroot
    test_arg_dict['phase'] = opt.test_data_phase
    test_arg_dict['fold'] = opt.fold
    test_arg_dict['preprocess'] = opt.test_preprocess
    # test_arg_dict['scale'] = opt.test_scale
    test_arg_dict['target_size'] = opt.crop_size
    if getattr(opt, 'source_crop_size', None):
        test_arg_dict['source_crop_size'] = opt.source_crop_size
    if getattr(opt, 'target_crop_size', None):
        test_arg_dict['target_crop_size'] = opt.source_crop_size
    test_arg_dict['crop_size'] = opt.crop_size
    test_arg_dict['order_data'] = opt.order_data
    test_arg_dict['order_seg'] = opt.order_seg
    # ----------------------------------------------------------------
    # shift_mu shift_sigma elastic_sigma elastic_alpha bright_sigma bright_mu target_size
    # test_arg_dict['shift_mu'] = opt.shift_mu
    # test_arg_dict['shift_sigma'] = opt.shift_sigma
    # test_arg_dict['elastic_sigma'] = opt.elastic_sigma
    # test_arg_dict['elastic_alpha'] = opt.elastic_alpha
    # test_arg_dict['bright_sigma'] = opt.bright_sigma
    # test_arg_dict['bright_mu'] = opt.bright_mu

    dataset_filename = "data.dataloads." + opt.dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset_class = None
    target_dataset_name = 'predict' + opt.dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset_class = cls
    if dataset_class is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    # dataset_class = find_dataset_using_name(opt.dataset_name)
    dataset = dataset_class(SimpleNamespace(**test_arg_dict))
    ddp_logger.warning(" test dataset [%s] was created" % type(dataset).__name__)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=opt.world_size,
                                                              rank=opt.rank,
                                                              shuffle=False,
                                                              seed=0,
                                                              drop_last=False) if opt.use_distribute_sample else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.test_batchsize,
        shuffle=False,
        sampler=sampler,        #
        batch_sampler=None,     #
        num_workers=int(opt.num_threads),
        collate_fn=None,
        pin_memory=True,
        drop_last=False
    )
    return dataloader


def create_predict_dataset(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data.dataloads." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = 'predict' + dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    return dataset


def create_slide_test_dataset(opt):
    dataset_filename = "data.dataloads." + opt.dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = 'test' + opt.dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name "
                                  "that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    kwargs = {
        'dataroot': opt.dataroot,
        'phase': opt.test_data_phase,
        'fold': opt.fold
    }
    test_dataset = dataset(SimpleNamespace(**kwargs))
    ddp_logger.warning(" slide_test_dataset [%s] was created" % type(dataset).__name__)
    return test_dataset


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_name)
        self.dataset = dataset_class(opt)
        # print("dataset [%s] was created" % type(self.dataset).__name__)
        ddp_logger.warning("dataset [%s] was created" % type(self.dataset).__name__)
        sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                  num_replicas=opt.world_size,
                                                                  rank=opt.rank,
                                                                  shuffle=opt.data_shuffle,
                                                                  seed=0,
                                                                  drop_last=opt.drop_last) if opt.use_distribute_sample else None
        # torch.distributed.get_world_size()  torch.distributed.get_rank()
        # batch_sample = torch.utils.data.BatchSampler(sampler=sampler, batch_size=opt.batch_size, drop_last=False)
        # collate_fn = None

        self.sample = sampler
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=(sampler is None) and opt.data_shuffle,
            sampler=sampler,        #
            batch_sampler=None,     #
            num_workers=int(opt.num_threads),
            collate_fn=None,        #
            pin_memory=True,
            drop_last=opt.drop_last,        #
            prefetch_factor=2       #
        )

    def load_data(self):
        return self

    def get_true_loader(self):
        return self.dataloader

    def set_epoch(self, epoch):
        if self.sample:
            self.sample.set_epoch(epoch)

    def get_loader_size(self):
        return len(self.dataloader)

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                ddp_logger.warning('max_dataset_size:{}'.format(self.opt.max_dataset_size))
                break
            yield data


class ProxyDataloader(object):
    def __init__(self, opt):
        dataset_args = {
            "random_state": opt.random_state,
            "batch_size": opt.batch_size,
            "max_dataset_size": opt.max_dataset_size,
            "dataroot": opt.dataroot,
            "phase": opt.phase,
            "seed": opt.seed,
            "fold": opt.fold,
            "serial_batches": opt.serial_batches,
            "custom": opt.custom,
            "preprocess": opt.preprocess,
            "order_data": opt.order_data,
            "order_seg": opt.order_seg,
            "elastic_alpha": opt.elastic_alpha,
            "elastic_sigma": opt.elastic_sigma,
            "scale_range": opt.scale_range,
            "rot_axes": opt.rot_axes,
            "rot_angle_spectrum": opt.rot_angle_spectrum,
            "mirror_axes": opt.mirror_axes,
            "crop_size": opt.crop_size,
            "crop_stride": opt.crop_stride,
            "g_noise_variance": opt.g_noise_variance,
            "bright_multiplier_range": opt.bright_multiplier_range,
            "contrast_range": opt.contrast_range,
            "simulate_zoom_range": opt.simulate_zoom_range,
            "gamma_range": opt.gamma_range,
    }

        self.opt = SimpleNamespace(**dataset_args)
        mr_dataset_class = find_dataset_using_name("mrusmrplus")
        self.mr_dataset = mr_dataset_class(self.opt)
        ddp_logger.warning("dataset [%s] was created" % type(self.mr_dataset).__name__)
        us_dataset_class = find_dataset_using_name("mrususplus")
        self.us_dataset = us_dataset_class(self.opt)
        ddp_logger.warning("dataset [%s] was created" % type(self.us_dataset).__name__)
        mr_sampler = torch.utils.data.distributed.DistributedSampler(self.mr_dataset,
                                                                     num_replicas=opt.world_size,
                                                                     rank=opt.rank,
                                                                     shuffle=opt.data_shuffle,
                                                                     seed=0,
                                                                     drop_last=opt.drop_last) if opt.use_distribute_sample else None
        us_sampler = torch.utils.data.distributed.DistributedSampler(self.us_dataset,
                                                                     num_replicas=opt.world_size,
                                                                     rank=opt.rank,
                                                                     shuffle=opt.data_shuffle,
                                                                     seed=0,
                                                                     drop_last=opt.drop_last) if opt.use_distribute_sample else None

        self.mr_sampler = mr_sampler
        self.us_sampler = us_sampler
        self.mr_dataloader = torch.utils.data.DataLoader(
            self.mr_dataset,
            batch_size=opt.batch_size,
            shuffle=(self.mr_sampler is None) and opt.data_shuffle,
            sampler=self.mr_sampler,  #
            batch_sampler=None,  #
            num_workers=int(opt.num_threads),
            collate_fn=None,  #
            pin_memory=True,
            drop_last=opt.drop_last,  #
            prefetch_factor=2  #
        )
        self.us_dataloader = torch.utils.data.DataLoader(
            self.us_dataset,
            batch_size=opt.batch_size,
            shuffle=(self.us_sampler is None) and opt.data_shuffle,
            sampler=self.us_sampler,  #
            batch_sampler=None,  #
            num_workers=int(opt.num_threads),
            collate_fn=None,  #
            pin_memory=True,
            drop_last=opt.drop_last,  #
            prefetch_factor=2  #
        )

    def load_data(self):
        return self

    def set_epoch(self, epoch):
        if self.mr_sampler:
            self.mr_sampler.set_epoch(epoch)
        if self.us_sampler:
            self.us_sampler.set_epoch(epoch+4)

    def get_loader_size(self):
        return min(len(self.mr_dataloader), len(self.us_dataloader))

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.mr_dataset), len(self.us_dataset), self.opt.max_dataset_size)

    def __iter__(self):
        data = defaultdict()
        i = 0
        for mr_data, us_data in zip(self.mr_dataloader, self.us_dataloader):
            for key, value in mr_data.items():
                data["mr_" + key] = value
            for key, value in us_data.items():
                data["us_" + key] = value
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                ddp_logger.warning('max_dataset_size:{}'.format(self.opt.max_dataset_size))
                break
            yield data


if __name__ == "__main__":
    pass
