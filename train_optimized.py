'''
使用DDP时，使用gpu_ids作为运行的GPU号码和数量
优化版本，可以完整使用gpu_ids.也可以从script直接运行DDP
'''
import os
import time
import torch.multiprocessing as mp

from train import do_train as do_train_single
from train_multi import do_train as do_train_multi
from configs.simple_options import get_opt
from utils.others.utils import init_seed, init_torch, get_device_name
# from configs.options.dataset_network import ProjectOptions
from argparse import ArgumentParser, REMAINDER, ZERO_OR_MORE, OPTIONAL


# 维护rank, local_rank, world_size, init_method, backend
def correct_dist_args(ind, args):
    # 当ind=-1，表示没有使用DDP训练，或者使用了DDP，但使用环境变量提供
    if ind == -1:
        # using environment variables to initialize or not in DDP
        args.rank = -1
        args.local_rank = -1
    elif args.dist_url == 'env://' and args.rank == -1:
        args.local_rank = ind
        os.environ["RANK"] = int(os.environ["RANK"]) + ind
    else:
        args.local_rank = ind
        args.rank = args.rank + ind

    return args


def setup_apex_env(opt):
    import os
    os.environ['RANK'] = str(opt.rank)
    os.environ['LOCAL_RANK'] = str(opt.local_rank)
    os.environ['WORLD_SIZE'] = str(opt.world_size)


def train_single(ind, *args):
    '''
    :param ind:process id, from 0 to len(opt.gpu_ids), when ind=-1, means not DDP
    :param args:
    :return:
    '''
    opt = args[0]

    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    device_name = get_device_name()
    opt.name = opt.name + '_' + device_name if device_name is not None else opt.name

    opt = correct_dist_args(ind, opt)

    if opt.APEX:
        setup_apex_env(opt)

    # if opt.DEBUG:
    #     from configs.utils_config import pretty_print_opt
    #     pretty_print_opt(opt)

    do_train_single(opt)


def train_multi(ind, *args):
    '''
    :param ind:process id, from 0 to len(opt.gpu_ids), when ind=-1, means not DDP
    :param args:
    :return:
    '''
    opt = args[0]

    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)
    device_name = get_device_name()
    opt.name = opt.name + '_' + device_name if device_name is not None else opt.name

    opt = correct_dist_args(ind, opt)

    if opt.APEX:
        setup_apex_env(opt)

    # if opt.DEBUG:
    #     from configs.utils_config import pretty_print_opt
    #     pretty_print_opt(opt)

    do_train_multi(opt)


def train_ddp(args, train_fn):
    if args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    assert args.world_size > 0, 'world_size{} have to > 0'.format(args.world_size)
    assert len(args.gpu_ids) > 0, 'gpu_ids{} have to specified'.format(args.gpu_ids)
    nprocs = min(args.world_size, len(args.gpu_ids))

    print(type(args))
    if len(args.gpu_ids) > 0:
        # 这个ConfigDict有点问题，不能被spawn传递.  会报 'NoneType' object is not callable
        # 现在用的是SimpleNamespace
        mp.spawn(fn=train_fn,
                 args=(args,),
                 nprocs=nprocs,
                 join=True,
                 daemon=False)
    else:
        raise ValueError('when use ddp, you must provide the correct gpu_ids,'
                         ' but got [] of {}'.format(repr(args.gpu_ids)))


def main(config_name, sleep_sec=10):
    # opt = ProjectOptions().parse(True)   # get training options
    # opt = get_opt(args=None)
    # opt = get_opt(args=['--config_path=configs/defaults/mrusmr_unet_train.yaml', '--use_config'])
    opt = get_opt(args=[f'--config_path=configs/defaults/{config_name}.yaml', '--use_config'])
    # opt = get_opt(args=['--config_path=configs/defaults/promise12_unet3d.yaml', '--use_config'])
    # opt = get_opt(args=['--config_path=configs/defaults/trus_unet3d.yaml', '--use_config'])
    print('option get ready')

    train = train_single if opt.single else train_multi

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    time.sleep(sleep_sec)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if opt.DDP:
        train_ddp(opt, train)
    else:
        train(-1, opt)


if __name__ == "__main__":
    parser = ArgumentParser(description="Project's useful tool to parse args")
    # rest from the training program
    # local_rank, is suitable to distrubute.launch
    parser.add_argument('--config_name', type=str, default='mrusus_unet_train', help='the name of config')
    parser.add_argument('second', type=int, default=10, help='wait some second and then run')
    parser.add_argument('training_script_args', nargs=REMAINDER, help='training_script_args')
    args = parser.parse_args()
    main(args.config_name, args.second)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # # using environment variables to initialize or not in DDP
    # warnings.warn('you are trying to use environment variables to initialize the DDP, please try to use the utils '
    #               'that torch.distributed.launch to run script. or simply run script on multi-shell')


