from argparse import ArgumentParser, REMAINDER, ZERO_OR_MORE, OPTIONAL
from configs.utils_config import ConfigDict, dict2obj, SimpleNamespace, get_pretty_opt


def parse_args(args=None):
    parser = ArgumentParser(description="Project's useful tool to parse args")
    # rest from the training program
    # local_rank, is suitable to distrubute.launch
    parser.add_argument('--config_path', type=str, default=None, help='the path of config')
    parser.add_argument('--use_config', default=False, action="store_true", help='whether to use config')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local_rank of distributed processes. local_rank = gpu_ids[ind], -1 means cpu')
    parser.add_argument('--use_current_local_rank', default=False, action="store_true",
                        help='whether to use local_rank in this args. only useful when using torch.disribute.launch')

    parser.add_argument('--option_name', type=str, default='ProjectOptions',
                        help='useless now, just a position flag')
    parser.add_argument('isTrain', type=str, default='train', nargs=OPTIONAL, help='whether is training')
    parser.add_argument('training_script_args', nargs=REMAINDER, help='training_script_args')
    # OPTIONAL = '?'
    # ZERO_OR_MORE = '*'
    # ONE_OR_MORE = '+'
    # REMAINDER = '...'
    return parser.parse_args(args=args)


# 默认使用ProjectOptions，train的默认option, opt = get_opt(args=None)
# 使用train的非默认option，  opt = get_opt(args=['train', ...])
# 使用ProjectOptions的test模式默认option， opt = get_opt(args=['test'])
# 使用test模式的非默认option，opt = get_opt(args=['test', ...])
# 自定义options类， opt = get_opt(args=['--option_name={}', 'train', ...]])
# 使用yaml文件参数输入， opt = get_opt(args=['--config_path={}', '--use_config'])
# 注意，并行时候要注意local_rank的赋值。所有参数都可通过命令行输入
def get_opt(args=None, save_log=True):
    args = parse_args(args=args)    #
    # print('args:', args)
    if args.use_config and args.config_path is not None:
        from configs.default_config import _C as cfg    # yacs.config.CfgNode, dict
        cfg.merge_from_file(args.config_path)           # dict
        if args.use_current_local_rank:
            cfg.local_rank = args.local_rank
        option = SimpleNamespace(**cfg)
        # option = ConfigDict(cfg)
        option.use_distribute_sample = option.DDP or option.HOROVOD

        return option
    else:
        import importlib
        option_lib = importlib.import_module("configs.options")
        try:
            option_class = getattr(option_lib, args.option_name)
        except AttributeError as e:
            print('some wrong of [{}] have been found, maybe the option name {} that you input can not be found, '
                  'using a default option with the name of {}'.format(e, args.option_name, 'ProjectOptions'))
            option_class = getattr(option_lib, 'ProjectOptions')
        option = option_class().parse(args.isTrain == 'train', args=args.training_script_args)
        return option


if __name__ == '__main__':
    # opt = parse_args(args=['fad', '--local_rank=5', '--config_path=2', '--dsf', 'haha', '--local_rank', '4'])
    # opt = parse_args(args=['fff', '--name=hello'])
    opt = get_opt(args=['fff', '--name=hello'])
    # print(type(opt))
    from utils_config import pretty_print_opt
    pretty_print_opt(opt)
    # print(opt)
    # print(vars(opt))



