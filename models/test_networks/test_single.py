from models.test_networks.test_base import TestBase

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


class TestSingle(TestBase):
    def __init__(self, opt):
        super(TestSingle, self).__init__(opt)
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

        s_sum = self.source_predict_numpy.sum()
        t_sum = self.target_predict_numpy.sum()
        self.metric_dict_source.update({'mravd': (s_sum-t_sum)/(float(t_sum)+1e-7)})
        self.metric_dict_target.update({'mravd': (t_sum-s_sum)/(float(s_sum)+1e-7)})

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
                try:
                    print(f"kept_size: {kept_size:.2f}mm^3, largest_removed: {largest_removed}")
                except TypeError:
                    print(f"kept_size: {kept_size}mm^3, largest_removed: {largest_removed}")
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











