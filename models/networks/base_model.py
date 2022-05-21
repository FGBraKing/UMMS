import os
import logging
import warnings

import torch
import torch.distributed
import torch.nn as nn
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
from utils.others.distributed_utils import reduce_mean, torch_distributed_zero_first
# from models.auxiliary_funs import get_scheduler

try:
    import apex.amp
    # import apex.optimizers
    # from apex.fp16_utils import *
    has_apex = True
except ImportError:
    has_apex = False


ddp_logger = logging.getLogger('ddp_logger')


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        # if opt.DDP:
        #     self.device = torch.device('cuda:{}'.format(opt.local_rank))  #
        # else:
        #     self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.device = torch.device('cuda:{}'.format(opt.local_gpu)) if opt.local_gpu >= 0 else torch.device('cpu')
        ddp_logger.warning(repr(self.device))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.logs_dir = os.path.join(opt.logs_dir, opt.name)
        # 多进程情况下，会有重复创建的报错
        with torch_distributed_zero_first(opt.local_rank):
            if not os.path.exists(self.save_dir):
                ddp_logger.warning('making a dir of {}'.format(self.save_dir))
                os.mkdir(self.save_dir)
            if not os.path.exists(self.logs_dir):
                ddp_logger.warning('making a dir of {}'.format(self.logs_dir))
                os.makedirs(self.logs_dir)

        # self.data_paths = []
        self.volume_path = []
        self.label_path = []
        self.model_names = []
        self.loss_names = []
        self.loss_item_dict = defaultdict()
        self.optimizers = []
        # define self.optimizers and self.loss_criterion
        # self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.schedulers = []
        self.visual_names = []
        self.metric_names = []
        self.metric_dict = {}
        self.lr_metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

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

    @abstractmethod
    def optimize_parameters(self, update=True):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if not self.isTrain or opt.continue_train:
            if opt.APEX and has_apex:
                self.load_for_apex(self.opt.weight_path)
            else:
                self.load_networks(self.opt.weight_path)
        self.print_networks(opt.verbose)

    def zero_grad_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

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

    def slide_test(self, patient_data):
        pass

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def compute_metrics(self):
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.volume_path

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step(epoch)
            # if self.opt.lr_policy == 'plateau':
            #     scheduler.step(self.lr_metric)
            # else:
            #     scheduler.step(epoch)

        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))
        ddp_logger.info('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_lrs(self):
        lrs = []
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                lrs.append(group['lr'])
        return lrs

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

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
                # float(...) works for both scalar tensor and float number

        if self.opt.DDP:
            # torch.distributed.barrier()
            for k, v in errors_ret.items():
                if isinstance(v, torch.Tensor):
                    errors_ret[k] = reduce_mean(v, torch.distributed.get_world_size())

        for key, item in self.loss_item_dict.items():
            errors_ret[key] = item
        return errors_ret

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

    def save_for_apex(self, epoch):
        # restoring the model using the same opt_level
        if self.opt.DDP and torch.distributed.get_rank() != 0:
            return
        state_dict = defaultdict()
        state_dict['amp'] = apex.amp.state_dict()
        for name in self.model_names:
            net = getattr(self, 'net_' + name)
            state_dict[name] = net.state_dict()
        for ind, optimizer in enumerate(self.optimizers):
            state_dict[ind] = optimizer.state_dict()

        save_filename = '%s_net_apex_%s.pth' % (epoch, self.opt.name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state_dict, save_path)

    def load_for_apex(self, load_path):
        # recommend calling the load_state_dict methods after amp.initialize
        if not os.path.exists(load_path):
            raise IOError(f"Checkpoint '{load_path}' does not exist")
        ddp_logger.info('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))

        apex.amp.load_state_dict(state_dict['amp'])
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.load_state_dict(state_dict[name])
        for ind, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state_dict[ind])

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if self.opt.DDP and torch.distributed.get_rank() != 0:
            return
        state_dict = defaultdict()
        # state_dict['lr'] = self.optimizers[0].param_groups[0]['lr']
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                # if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                if isinstance(net, torch.nn.parallel.DataParallel) \
                        or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    state_dict[name] = net.module.state_dict()
                else:
                    state_dict[name] = net.state_dict()
        save_filename = '%s_net_%s.pth' % (epoch, self.opt.name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state_dict, save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        # keys:[module parameter ]
        key = keys[i]   # 得到该parameter的module name
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, load_path):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not os.path.exists(load_path):
            raise IOError(f"Checkpoint '{load_path}' does not exist")
        ddp_logger.info('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.parallel.DataParallel) \
                        or isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    ddp_logger.warning('loading model of type torch.nn.parallel')
                    net = net.module
                if name in state_dict.keys():
                    net_state_dict = state_dict[name]
                else:
                    net_state_dict = state_dict
                if hasattr(net_state_dict, '_metadata'):
                    del net_state_dict._metadata
                for key in list(net_state_dict.keys()):
                    self.__patch_instance_norm_state_dict(net_state_dict, net, key.split('.'))
                net.load_state_dict(net_state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        ddp_logger.info('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    # print(net)
                    ddp_logger.info(repr(net))
                ddp_logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        ddp_logger.info('-----------------------------------------------')

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):  # self,
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_optimizer(self, epoch):
        state_dict = defaultdict()
        # state_dict['lr'] = self.optimizers[0].param_groups[0]['lr']
        for ind, optimizer in enumerate(self.optimizers):
            state_dict[ind] = optimizer.state_dict()
        save_filename = '%s_optimizer_%s.pth' % (epoch, self.opt.name)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(state_dict, save_path)

    def load_optimizer(self):
        load_path = self.opt.optim_path
        if load_path is None:
            warnings.warn('the path of optimizer is None', RuntimeWarning)
            return
        if not os.path.exists(load_path):
            raise IOError(f"Checkpoint '{load_path}' does not exist")
        ddp_logger.info('loading the optimizer from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        for ind, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(state_dict[ind])

    def get_optimizers(self):
        return self.optimizers

    def get_schedulers(self):
        return self.schedulers

    # optional
    def warp_horovod_optimizer(self):
        pass

    def broadcast_horovod_parameters(self):
        pass

    def optimize_parameters_with_apex(self):
        pass

# torch.cuda.amp
# ['GradScaler', 'autocast', 'autocast_mode', 'custom_bwd', 'custom_fwd', 'grad_scaler']
