import torch.multiprocessing as mp

from predict import do_predict
from types import SimpleNamespace
from utils.others.utils import init_torch
from configs.utils_config import get_pretty_opt, get_config


def predict(ind, *args):
    opt = args[0]

    opt.local_gpu = opt.gpu_ids[ind]
    opt.phase = opt.phase_list[ind]

    init_torch(gpu_id=opt.visible_gpu, deterministic=opt.deterministic)

    do_predict(opt)


def main():
    opt_dict = get_config('configs/defaults/trus_unet3d_predict.yaml')
    opt = SimpleNamespace(**opt_dict)
    print('option get ready')

    nprocs = len(opt.gpu_ids)
    assert nprocs > 0, 'gpu_ids{} have to specified'.format(opt.gpu_ids)
    mp.spawn(fn=predict,
             args=(opt,),
             nprocs=nprocs,
             join=True,
             daemon=False)


if __name__ == "__main__":
    main()
