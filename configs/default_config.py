'''
三级配置文件----第一级
把所有会用到的参数项都在这里提供，并提供一些不再改变的固定参数值
'''
from yacs.config import CfgNode as CN


_C = CN(new_allowed=True)  # init_dict=None, key_list=None, new_allowed=False


# ---------------------------------------------------------------------------- #
# basic
# ---------------------------------------------------------------------------- #
_C.name = None
_C.dataset_name = None
_C.model_name = None
_C.seed = 1008
_C.gpu_ids = None
_C.visible_gpu = None
_C.local_gpu = -1
_C.DEBUG = None
_C.deterministic = None
# ---------------------------------------------------------------------------- #
# tricks
# ---------------------------------------------------------------------------- #
_C.use_mixed_precision = None
# DDP
_C.DP = None
_C.DDP = None
_C.SyncBatchNorm = None
_C.world_size = None
_C.rank = None
_C.local_rank = None
_C.dist_url = None
_C.dist_backend = 'nccl'
# horovod
_C.HOROVOD = False
_C.HOROVOD_fp16 = False
_C.HOROVOD_backward_passes_per_step = False
_C.HOROVOD_use_adasum = False
_C.HOROVOD_gradient_predivide_factor = 1.0
# Apex
_C.APEX = False
_C.APEX_opt_level = 'O1'
# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
# dataset
_C.dataroot = None
_C.phase = None
_C.serial_batches = None
_C.data_shuffle = None
_C.custom = None
_C.preprocess = None
_C.crop_size = None
_C.target_size = None
_C.scale = None
_C.bright_mu = None
_C.bright_sigma = None
_C.elastic_alpha = None
_C.elastic_sigma = None
_C.shift_mu = None
_C.shift_sigma = None
_C.order_data = 3
_C.order_seg = 1
# dataloader
_C.num_threads = None
_C.batch_size = None
_C.drop_last = None
_C.max_dataset_size = float('inf')
# ---------------------------------------------------------------------------- #
# module
# ---------------------------------------------------------------------------- #
# model
_C.input_nc = None
_C.output_nc = None
_C.init_channel_number = None
_C.up_interpolate = None
_C.conv_order = None
# loss
_C.loss_name = 'combo'
_C.loss_alpha = None
_C.loss_beta = None
_C.loss_gamma = None
_C.loss_weight = None
_C.loss_eps = 1e-7
_C.loss_smooth = None
_C.reduction = None
_C.ignore_index = None
# initialization
_C.init_type = None
_C.init_gain = None
_C.init_std = None
# ---------------------------------------------------------------------------- #
# optimizer
# ---------------------------------------------------------------------------- #
# optimizer
_C.optimizer_name = None
_C.lr = None
_C.weight_decay = 0.0
_C.momentum = 0.9
_C.beta1 = 0.9
# scheduler
_C.lr_policy = None
_C.lr_noise = None
_C.lr_noise_pct = 0.67
_C.lr_noise_std = 1.
_C.warmup_epochs = None
_C.warmup_lr = None
_C.warmup_prefix = True
_C.lr_cycle_mul = None
_C.lr_cycle_decay = None
_C.lr_cycle_limit = None
_C.cooldown_epochs = 5
_C.min_lr = 1e-8
_C.decay_epochs = None
_C.decay_rate = None
_C.lr_k_decay = 1.0
_C.eval_metric = ''
_C.patience_epochs = 20
# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #
_C.logs_dir = './traces/logs'
_C.checkpoints_dir = './traces/checkpoints'
_C.weight_path = None
_C.optimizer_path = None
_C.apex_path = None
_C.continue_train = None
_C.verbose = None
# additional
_C.suffix = None
_C.epoch_start = None
_C.num_epochs = None

# ---------------------------------------------------------------------------- #
# train
# ---------------------------------------------------------------------------- #
_C.use_gradient_accumulation = False
_C.gradient_accumulation_k_step = None
# network saving and loading
_C.save_epoch_start = None
_C.save_epoch_freq = None
_C.save_iter_start = None
_C.save_iter_freq = None
_C.save_by_iter = None
# network print and showing
_C.display_freq = None
_C.print_freq = None
_C.plot_freq = None
_C.val_epoch_freq = None
_C.test_on_train = None
# visualizer
_C.with_html = False
_C.with_tensorboard = True
_C.with_visdom = False
_C.save_log = True
_C.save_visuals = False
_C.save_only_latest = True
_C.save_visuals_frep = 1

# visdom
_C.visdom_server = '172.21.16.17:25555'
_C.display_port = 6666
_C.display_env = 'main'
_C.display_id = 1
_C.display_ncols = 2
# html
_C.display_winsize = 128
# tensorboard and logging
_C.draw_model = False
_C.display_on_tensorboard = False
_C.display_histogram = False
_C.play_video = False

# ---------------------------------------------------------------------------- #
# test
# ---------------------------------------------------------------------------- #
_C.results_dir = './traces/results'
_C.eval = None




