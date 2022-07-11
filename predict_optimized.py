import torch.multiprocessing as mp
import time
from predict import do_predict as do_predict_single
from predict_multi import do_predict as do_predict_multi
from types import SimpleNamespace
from utils.others.utils import init_torch
from configs.utils_config import get_pretty_opt, get_config
from argparse import ArgumentParser, REMAINDER, ZERO_OR_MORE, OPTIONAL


def predict(ind, *args):
    opt = args[0]

    opt.local_gpu = opt.gpu_ids[ind]
    opt.phase = opt.phase_list[ind]

    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)

    do_predict = do_predict_single if opt.single else do_predict_multi

    do_predict(opt)


def main(config_name, sleep_sec=5):
    opt_dict = get_config(f'configs/defaults/{config_name}_predict.yaml')
    opt = SimpleNamespace(**opt_dict)
    print('option get ready')

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(opt.name)
    print('waiting: {} seconds'.format(sleep_sec))
    time.sleep(sleep_sec)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    nprocs = len(opt.gpu_ids)
    assert nprocs > 0, 'gpu_ids{} have to specified'.format(opt.gpu_ids)
    mp.spawn(fn=predict,
             args=(opt,),
             nprocs=nprocs,
             join=True,
             daemon=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Project's useful tool to parse args")
    parser.add_argument('--config_name', type=str, default='ummkd', help='the name of config')
    parser.add_argument('--second', type=int, default=5, help='wait some second and then run')
    parser.add_argument('training_script_args', nargs=REMAINDER, help='training_script_args')
    aux_args = parser.parse_args()
    main(aux_args.config_name, aux_args.second)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
