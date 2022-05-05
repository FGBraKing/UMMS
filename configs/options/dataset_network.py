import os
import argparse
import numpy as np
from configs.utils_config import only_one_true
from utils.others.utils import mkdirs, convert_str_to_list


class ProjectOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.isTrain = False
        self.parser = None
        self.opt = None

    @staticmethod
    def data_initialize(parser):
        # dataset parameters
        parser.add_argument('--dataroot', type=str,
                            default='/data/project_data_lf/PROJECT/DLForPytorch/datasets/promise12',
                            help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='train')
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--custom', action='store_true', help='whether to use custom configure')
        parser.add_argument('--preprocess', type=str,
                            default='randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror_'
                                    'gaussianNoise_GaussianBlur_BrightnessMultiplicative_'
                                    'contrast_simulate_gammatransform',
                            help='scaling and cropping of images at load time ')
        parser.add_argument('--crop_size', type=str, default='128,128,32', help='the crop size of slide  windows')
        parser.add_argument('--target_size', type=str, default='128,128,128', help='the target size ')
        parser.add_argument('--scale', type=str, default='1.,1.,1.', help='the scale of target size')
        parser.add_argument('--scale_range', type=list, default=[0.85, 1.25])
        parser.add_argument('--rot_angle_spectrum', type=int, default=25)
        parser.add_argument('--rot_axes', type=str, default='0,1,2', help='the rot90 axes')
        parser.add_argument('--mirror_axes', type=str, default='0,1,2', help='the mirror axes')
        parser.add_argument('--g_noise_variance', type=list, default=[0.3, 0.7])
        parser.add_argument('--bright_mu', type=float, default=0.0, help='brightness')
        parser.add_argument('--bright_sigma', type=float, default=0.5, help='brightness')
        parser.add_argument('--elastic_alpha', type=str, default='0., 1000.', help='ElasticDeformTransform ')
        parser.add_argument('--elastic_sigma', type=str, default='10.,13.', help='ElasticDeformTransform ')
        parser.add_argument('--shift_mu', type=str, default='0., 1000.', help='RandomShiftTransform ')
        parser.add_argument('--shift_sigma', type=str, default='10., 13.', help='RandomShiftTransform ')
        parser.add_argument('--order_data', type=int, default=3, help='order_data ')
        parser.add_argument('--order_seg', type=int, default=0, help='order_seg ')

        # dataloader parameters
        parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
        parser.add_argument('--data_shuffle', action='store_true', help='data_shuffle')
        parser.add_argument('--num_threads', type=int, default=1, help='# threads for loading data')
        parser.add_argument('--drop_last', action='store_true', help='data_shuffle')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains'
                                 ' more than max_dataset_size, only a subset is loaded.')
        return parser

    @staticmethod
    def model_initialize(parser):
        # model parameters
        parser.add_argument('--input_nc', type=int, default=1, help='# of input volume channels')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels:')
        parser.add_argument('--init_channel_number', type=int, default=32, help='the init channel number of unet')
        parser.add_argument('--up_interpolate', action='store_true', help='upsample_interpolate')
        parser.add_argument('--conv_order', type=str, default='crb', help='# of the order of conv layer in the 3d-unet')
        # initialization parameters
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=np.sqrt(2),
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--init_std', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        return parser

    @staticmethod
    def optimizer_initialize(parser):
        # loss parameters
        parser.add_argument('--loss_name', type=str, default='combo', help='loss  name')
        parser.add_argument('--loss_alpha', type=float, default=1., help='loss function alpha')
        parser.add_argument('--loss_beta', type=float, default=1., help='loss function beta')
        parser.add_argument('--loss_gamma', type=float, default=1., help='loss function gamma')
        parser.add_argument('--loss_weight', type=float, default=1., help='loss function weight')
        parser.add_argument('--loss_eps', type=float, default=1e-7, help='loss function eps')
        parser.add_argument('--loss_smooth', type=float, default=1., help='loss function smooth')
        parser.add_argument('--reduction', type=str, default='mean', help='loss reduction')
        parser.add_argument('--ignore_index', type=str, default=None, help='which class should be ignore')
        # optimizer parameters
        parser.add_argument('--optimizer_name', type=str, default='adam', help='name of optimizer to create')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 penalty) (default: 0)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum for momentum based optimizers (others may use betas via kwargs)')
        parser.add_argument('--optim_beta', type=float, default=0.9, help='momentum term of adam')
        # scheduler parameters
        parser.add_argument('--lr_policy', type=str, default='step',
                            help='learning rate policy. [linear | step | plateau | cosine]')

        parser.add_argument('--lr_noise', type=float, default=None,
                            help='the range of epochs for applying noise to lr')
        parser.add_argument('--lr_noise_pct', type=float, default=0.67, help='lr_noise_pct')
        parser.add_argument('--lr_noise_std', type=float, default=1.0, help='lr_noise_std')

        parser.add_argument('--warmup_lr', type=float, default=1e-7, help='warmup_lr_init')
        parser.add_argument('--warmup_epochs', type=int, default=100, help='how many epoch to warmup')
        parser.add_argument('--warmup_prefix', action='store_true', help='warmup_prefix')

        parser.add_argument('--lr_cycle_mul', type=float, default=1.0, help='lr_cycle_mul')
        parser.add_argument('--lr_cycle_decay', type=float, default=1.0, help='lr_cycle_decay')
        parser.add_argument('--lr_cycle_limit', type=int, default=5, help='lr_cycle_limit')
        parser.add_argument('--cooldown_epochs', type=int, default=5, help='cooldown_epochs')

        parser.add_argument('--min_lr', type=float, default=1e-8, help='min_lr')
        parser.add_argument('--decay_epochs', type=int, default=100,
                            help='multiply by a gamma every decay_epochs ')
        parser.add_argument('--decay_rate', type=float, default=0.1, help='the base to decay')
        parser.add_argument('--lr_k_decay', type=float, default=1.0, help='lr_k_decay')
        parser.add_argument('--eval_metric', type=str, default='', help='eval_metric')
        parser.add_argument('--patience_epochs', type=int, default=20, help='hpatience_epochs')

        return parser

    @staticmethod
    def tricks_initialize(parser):
        # gradient_accumulation
        parser.add_argument('--use_gradient_accumulation', default=False, action='store_true',
                            help='whether to use use_gradient_accumulation to train')
        parser.add_argument('--gradient_accumulation_k_step', type=int, default=1, help='gradient_accumulation_k_step')
        # mixed_precision
        parser.add_argument('--use_mixed_precision', default=False, action='store_true',
                            help='whether to use mixed_precision to train')
        # DDP distribution parameters
        parser.add_argument('--DP', action='store_true', help='use torch.nn.DataParallel')
        parser.add_argument('--DDP', action='store_true', help='torch.nn.parallel.DistributedDataParallel')
        parser.add_argument('--SyncBatchNorm', action='store_true', help='DDP with SyncBatchNorm')
        parser.add_argument('--world_size', type=int, default=-1, help='number of distributed processes')
        parser.add_argument('--rank', type=int, default=-1, help='The first rank of the process on this node')
        parser.add_argument('--local_rank', type=int, default=-1,
                            help='local_rank of distributed processes. local_rank = gpu_ids[ind], -1 means cpu')
        parser.add_argument('--dist_url', type=str, default='env://',
                            help='url used to set up distributed training')
        parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
        # horovod distribution
        parser.add_argument('--HOROVOD', action='store_true', help='horovod.torch')
        parser.add_argument('--HOROVOD_fp16', action='store_true', help='horovod.torch use fp16?')
        parser.add_argument('--HOROVOD_backward_passes_per_step', action='store_true',
                            help='Number of expected backward passes to perform before calling step()/synchronize().'
                                 ' This allows accumulating gradients over multiple mini-batches before reducing '
                                 'and applying them')
        parser.add_argument('--HOROVOD_use_adasum', action='store_true', help='horovod.torch')
        parser.add_argument('--HOROVOD_gradient_predivide_factor', type=float, default=1.0,
                            help='If op == Average, gradient_predivide_factor splits the averaging before and after '
                                 'the sum. Gradients are scaled by 1.0 / gradient_predivide_factor before the sum '
                                 'andgradient_predivide_factor / size after the sum.')
        # apex distribution
        parser.add_argument('--APEX', action='store_true', help='APEX')
        parser.add_argument('--APEX_opt_level', type=str, default='O1',
                            help='Recognized opt_levels are "O0", "O1", "O2", and "O3".')
        return parser

    @staticmethod
    def auxiliary_initialize(parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='promise_unet_default',
                            help='name of the experiment option. It decides where to store samples and models')
        parser.add_argument('--dataset_name', type=str, default='promise12', help='chooses how datasets are loaded')
        parser.add_argument('--model_name', type=str, default='unet3d', help='chooses which model to use.')
        parser.add_argument('--seed', type=int, default=1008, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='1',
                            help='available gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU;'
                                 'when using DDP, it also means nproc on this node')
        parser.add_argument('--visible_gpu', type=str, default='0,1,2,3', help='visible gpu ids: e.g. 0  0,1,2, 0,2.')
        parser.add_argument('--local_gpu', type=int, default=None, help='default gpu')
        parser.add_argument('--DEBUG', action='store_false',
                            help='in the debug mode, print moreover info, but do not save any more')
        parser.add_argument('--deterministic', required=False, default=False, action="store_true",
                            help='Makes training deterministic, but reduces training speed substantially')
        # additional
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix:e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        # files path
        parser.add_argument('--logs_dir', type=str,
                            default='./traces/logs',
                            help='logs are saved here')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='./traces/checkpoints',
                            help='models are saved here')
        parser.add_argument('--weight_path', type=str, default='None', help='')
        parser.add_argument('--optimizer_path', type=str, default='None', help='')
        parser.add_argument('--apex_path', type=str, default='None', help='')
        # basic train
        parser.add_argument('--epoch_start', type=int, default=1, help='form which epoch to start')
        parser.add_argument('--num_epochs', type=int, default=1000, help='total epochs for training')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        # network saving and loading parameters
        parser.add_argument('--save_epoch_start', type=int, default=500,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_epoch_freq', type=int, default=50,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_iter_start', type=int, default=5000,
                            help='we save the model by <save_epoch_start>, <save_epoch_start>+<save_latest_freq>, ...')
        parser.add_argument('--save_iter_freq', type=int, default=500,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true',
                            help='whether saves model by iteration')
        # network print and showing parameters
        parser.add_argument('--display_freq', type=int, default=64,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of print training loss on console')
        parser.add_argument('--plot_freq', type=int, default=1,
                            help='frequency of plot training metrics on console')
        parser.add_argument('--val_epoch_freq', type=int, default=10,
                            help='frequency of test, when training')
        parser.add_argument('--test_on_train', action='store_true',
                            help='whether do_test, on training')
        # visualizer parameters
        parser.add_argument('--with_html', action='store_true',
                            help='whether save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--with_tensorboard', action='store_true', help='whether to use tensorboard')
        parser.add_argument('--with_visdom', action='store_true', help='whether to use visdom')
        parser.add_argument('--save_log', action='store_true', help='whether to save logging file')
        parser.add_argument('--save_visuals', action='store_true',
                            help='whether to save visuals')
        parser.add_argument('--save_only_latest', action='store_true',
                            help='whether to save_only_latest')
        parser.add_argument('--save_visuals_frep', type=int, default=1,
                            help='the frep to save visuals')
        # visdom  parameters
        parser.add_argument('--visdom_server', type=str, default="http://172.21.141.4",
                            help='visdom server of the web display')
        parser.add_argument('--visdom_port', type=int, default=30303,
                            help='visdom port of the web display')
        parser.add_argument('--visdom_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--visdom_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--visdom_ncols', type=int, default=0,
                            help='if positive, display all images in a single visdom web panel '
                                 'with certain number of images per row.')
        # html parameters
        parser.add_argument('--html_winsize', type=int, default=256, help='display windows size for html')

        # tensorboard and logging parameters
        parser.add_argument('--draw_model', action='store_true', help='whether to draw model on tensorboard')
        parser.add_argument('--display_histogram', action='store_true', help='whether display histogram ')
        parser.add_argument('--display_on_tensorboard', action='store_true', help='whether to display visuals ')
        parser.add_argument('--play_video', action='store_true', help='whether play volume as a video ')

        return parser

    @staticmethod
    def train_initialize(parser):

        return parser

    @staticmethod
    def test_initialize(parser):
        # basic parameters
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        # rest from the training program
        parser.add_argument('training_script_args', nargs=argparse.REMAINDER)
        return parser

    def initialize(self, parser, is_train):
        """Define the common options that are used in both training and test."""
        parser = self.data_initialize(parser)
        parser = self.model_initialize(parser)
        parser = self.optimizer_initialize(parser)
        parser = self.tricks_initialize(parser)
        parser = self.auxiliary_initialize(parser)

        if is_train:
            # TODO: 暂时不对训练参数做分离
            parser = self.train_initialize(parser)
        else:
            parser = self.test_initialize(parser)
        self.isTrain = is_train
        self.initialized = True
        return parser

    def gather_options(self, is_train, args=None):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                             description='PyTorch mr-trus options')
            parser = self.initialize(parser, is_train)
        else:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # # get the basic options
        # opt, _ = parser.parse_known_args()
        # modify options
        # # save and return the parser
        self.parser = parser
        return parser.parse_args(args=args, namespace=None)

    def print_options(self, opt, save_log=False):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if save_log and opt.save_log and not opt.DEBUG:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)  # opt.dataset_name + opt.model_name + opt.name
            print('expr_dir:{}'.format(expr_dir))
            mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    @staticmethod
    def repair_options(opt, is_train):
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if (opt.dist_url is None) or (opt.dist_url == "env://") or (opt.world_size == -1) or (opt.rank == -1):
            # 环境变量初始化
            opt.dist_url = 'env://'

        return opt

    @staticmethod
    def change_options(opt, is_train):
        opt.isTrain = is_train
        opt.random_state = np.random.RandomState(seed=opt.seed)

        opt.gpu_ids = convert_str_to_list(opt.gpu_ids, split=',', aim_type=int, condition=lambda x: x >= 0)
        opt.crop_size = convert_str_to_list(opt.crop_size, split=',', aim_type=int, condition=lambda x: x > 0)
        opt.target_size = convert_str_to_list(opt.target_size, split=',', aim_type=int, condition=lambda x: x > 0)
        opt.scale = convert_str_to_list(opt.scale, split=',', aim_type=float, condition=lambda x: x > 0)
        opt.elastic_alpha = convert_str_to_list(opt.elastic_alpha, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.elastic_sigma = convert_str_to_list(opt.elastic_sigma, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.shift_mu = convert_str_to_list(opt.shift_mu, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.shift_sigma = convert_str_to_list(opt.shift_sigma, split=',', aim_type=float, condition=lambda x: x >= 0)
        opt.rot_axes = convert_str_to_list(opt.rot_axes, split=',', aim_type=int, condition=lambda x: x >= 0)
        opt.mirror_axes = convert_str_to_list(opt.mirror_axes, split=',', aim_type=int, condition=lambda x: x >= 0)
        if opt.ignore_index is not None:
            opt.ignore_index = convert_str_to_list(opt.ignore_index, split=',', aim_type=int, condition=lambda x: x >= 0)

        opt.use_distribute_sample = opt.DDP or opt.HOROVOD
        return opt

    def parse(self, is_train=True, args=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(is_train, args=args)
        # assert not(opt.DP and opt.DDP)

        if not only_one_true(opt.DP, opt.DDP):
            print('you have not use parallel')

        opt = self.repair_options(opt, is_train)    # repair some value of args

        if ((not opt.DDP) or (opt.DDP and opt.rank == 0) or (opt.DDP and opt.rank == -1 and opt.local_rank == 0))\
                and opt.verbose:
            self.print_options(opt)
        if opt.DDP:
            assert opt.weight_path is not None or opt.epoch_start == 1

        opt = self.change_options(opt, is_train)    # change some type of args

        self.opt = opt
        return self.opt

    def __str__(self):
        str_list = ['{:>35}: {:<45}'.format(*item) for item in sorted(vars(self.opt).items())]
        str_list.insert(0, '{:*^80s}'.format('Custom config'))
        str_list.append('{:*^80s}'.format('End'))
        message = '\n'.join(str_list)
        return message


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # parser.add_argument('--dataroot', default='/test', required=False,
    #                     help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    # parser.add_argument('--name', type=str, default='experiment_name',
    #                     help='name of the experiment. It decides where to store samples and models')
    # parser.add_argument('--gpu_ids', type=str, default='0',
    #                     help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
    #                     help='models are saved here')
    # opt = parser.parse_args(args=[])
    option = ProjectOptions().parse(True, args=None)
    print('option get ready')
#  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids[0]
#  torch.cuda.set_device(opt.gpu_ids[0])
#  torch.device('cuda:{}'.format(self.gpu_ids[0]))
#  data.to(device)
#  model.to(device)
#  net.to(gpu_ids[0])
#  net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs



