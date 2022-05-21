# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import yaml
import h5py
import torch
import logging
import imageio
import argparse
import multiprocessing
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
from pprint import pprint
from multiprocessing import Process

import torch.nn.functional as F
import torch.optim
import torch.distributed
import torch.utils.data
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.simple_options import get_opt
from configs.utils_config import pretty_print_opt, get_config
from data import create_dataset, CustomDatasetDataLoader
from data.utils_data import h5_loader, nii_loader
from models import create_model
from models.loss import get_loss_criterion, BinaryDiceLoss
from models.optim import create_optimizer, create_optimizer_v2
from models.scheduler import create_scheduler
from models.auxiliary_funs import get_init_func, get_activation
from utils.forLogs import Visualizer, get_logger
from utils.others.metrics import BinaryMetrics
from utils.others.utils import init_seed, init_torch, Timer, convert_str_to_list
from utils.others.img_io import show_array_3d, show_volume_label, show_volume_label_predict


def debug():
    pass


def main():
    # test_val_dataset()
    pass


if __name__ == '__main__':
    pass
