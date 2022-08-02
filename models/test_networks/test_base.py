import os
import logging
import warnings

import h5py
import torch
import numpy as np
import torch.distributed
import torch.nn as nn
from skimage.transform import resize, rescale
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from utils.others.distributed_utils import reduce_mean, torch_distributed_zero_first
from utils.others.metrics import BinaryMetrics, SoftMetrics

from data.connected_components import retain_the_largest_connected_component_binary


class TestBase(ABC):
    def __init__(self, opt):
        self.opt = opt
        # self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')

        self.phase = opt.phase
        self.test_name = opt.test_name

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.logs_dir = os.path.join(opt.logs_dir, opt.name)
        self.results_dir = os.path.join(opt.results_dir, opt.name)

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.model_names = []
        self.visual_names = []
        self.metric_names = []
        self.metric_dict = {}

        self.local_weight = None

    @abstractmethod
    def set_input(self, inputs):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            inputs (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
            self.compute_metrics()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def compute_metrics(self):
        pass

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_metrics(self):
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str) and name in self.metric_dict.keys():
                metrics_ret[name] = self.metric_dict[name]
        return metrics_ret

    def get_models(self):
        nets = []
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if isinstance(net, nn.Module):
                    if isinstance(net, torch.nn.parallel.DataParallel) \
                            or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        nets.append(net.module)
                    else:
                        nets.append(net)
        return nets

    def load_networks(self, load_path):
        self.local_weight = load_path
        if not os.path.exists(load_path):
            raise IOError(f"Checkpoint '{load_path}' does not exist")
        state_dict = torch.load(load_path, map_location=str(self.device))
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.parallel.DataParallel) \
                        or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                if name in state_dict.keys():
                    net_state_dict = state_dict[name]
                else:
                    net_state_dict = state_dict
                net.load_state_dict(net_state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))


class TestGeneric(TestBase):
    def __init__(self, opt):
        super(TestGeneric, self).__init__(opt)
        self.domains = ["source", "target"]

        self.model_names = ['umms']
        self.net_umms = nn.Identity()

        self.visual_names = opt.visual_names
        self.metric_names = opt.metric_names
        # ['DC', 'recall', 'precision', 'ravd', 'roisize']
        # 'hd' 'hd95' 'assd' 'asd'  'specificity', 'accuracy',

        self.get_metrics = BinaryMetrics()

        self.source_volume = None
        self.target_volume = None
        self.source_label = None
        self.target_label = None
        self.source_predict = None
        self.target_predict = None
        self.spacing = None

        self.metric_dict_source = None
        self.metric_dict_target = None

        self.source_volume_numpy = None
        self.target_volume_numpy = None
        self.source_label_numpy = None
        self.target_label_numpy = None
        self.source_predict_numpy = None
        self.target_predict_numpy = None

    def set_input(self, inputs):
        self.source_volume = inputs['mr_volume'].to(self.device)  # bs C D H W, C=1
        self.source_label = inputs['mr_label'].to(self.device)  # bs C D H W, C=1
        self.target_volume = inputs['us_volume'].to(self.device)  # bs C D H W, C=1
        self.target_label = inputs['us_label'].to(self.device)  # bs C D H W, C=1
        self.volume_path = {'source': inputs['mr_volume_path'], 'target': inputs['us_volume_path']}
        self.label_path = {'source': inputs['mr_label_path'], 'target': inputs['us_label_path']}
        self.spacing = {'source': inputs['mr_spacing'].mean(0).tolist(),
                        'target': inputs['us_spacing'].mean(0).tolist()}
        self.origin_shape = {'source': inputs['mr_origin_shape'], 'target': inputs['us_origin_shape']}
        self.now_shape = {'source': inputs['mr_now_shape'], 'target': inputs['us_now_shape']}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.source_predict = self.net_umms(self.source_volume, 'source')
        self.target_predict = self.net_umms(self.target_volume, 'target')

    def compute_visuals(self):
        pass

    def compute_metrics(self, *args, **kwargs):
        self.process_predict_result()

        self.metric_dict_source = self.compute_metrics_base(self.source_predict_numpy,
                                                            self.source_label_numpy, 'source')
        self.metric_dict_target = self.compute_metrics_base(self.target_predict_numpy,
                                                            self.target_label_numpy, 'target')

    def compute_metrics_base(self, predict, label, domain, *args, **kwargs):
        keys = tuple(self.metric_names) + args

        predict = predict > 0.5
        label = label > 0.5
        metrics = self.get_metrics(predict, label, *self.metric_names,
                                   *args, **kwargs, voxelspacing=self.spacing[domain])
        metric_dict = dict(zip(keys, metrics))
        return metric_dict

    def get_current_metrics(self):
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_ret['source'+name] = self.metric_dict_source[name]
                metrics_ret['target'+name] = self.metric_dict_target[name]
        return metrics_ret

    def get_current_metrics_by_domain(self, domain):
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_ret[name] = getattr(self, f"metric_dict_{domain}")[name]
        return metrics_ret

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret['source_'+name] = getattr(self, 'source_'+name)
                visual_ret['target_'+name] = getattr(self, 'target_'+name)
        return visual_ret

    def save_current_visuals(self, vis_name):
        save_name = os.path.join(self.results_dir, self.phase, self.test_name, vis_name + '.h5')
        with h5py.File(save_name, mode="w") as fw:
            for name in self.visual_names:
                if isinstance(name, str):
                    fw.create_dataset(name=f"source_{name}", data=getattr(self, f"source_{name}_numpy"))
                    fw.create_dataset(name=f"target_{name}", data=getattr(self, f"target_{name}_numpy"))

    def process_predict_result(self):
        for domain in ('source', 'target'):
            segment = getattr(self, domain+'_predict').clone().detach().cpu().numpy()[0, 0]
            segment = np.where(segment > 0.5, 1, 0)
            label = getattr(self, domain+'_label').clone().detach().cpu().numpy()[0, 0]
            volume = getattr(self, domain+'_volume').clone().detach().cpu().numpy()[0, 0]

            if self.opt.do_connected_component:
                volume_per_voxel = float(np.prod(self.spacing[domain], dtype=np.float64))
                segment, kept_size, largest_removed = \
                    retain_the_largest_connected_component_binary(segment, volume_per_voxel,
                                                                  self.opt.minimum_valid_object_size)
                print(f"kept_size: {kept_size:.2f}mm^3, largest_removed: {largest_removed}")
                # print(kept_size, largest_removed)

            if self.opt.revert:
                segment = resize(segment.astype(float), tuple(self.origin_shape[domain]),
                                 order=0, mode="constant", cval=0, clip=True, preserve_range=False, anti_aliasing=False)
                label = resize(label.astype(float), tuple(self.origin_shape[domain]),
                               order=0, mode="constant", cval=0, clip=True, preserve_range=False, anti_aliasing=False)
                volume = resize(volume.astype(float), tuple(self.origin_shape[domain]),
                                order=3, mode="constant", cval=0, clip=True, preserve_range=False, anti_aliasing=False)

            setattr(self, domain+"_predict_numpy", segment)
            setattr(self, domain+"_label_numpy", label)
            setattr(self, domain+"_volume_numpy", volume)







