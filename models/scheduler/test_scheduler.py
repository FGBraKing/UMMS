import torch
from torch.optim import SGD
from utils.others.img_io import plot_2d
from models.scheduler import *
from types import SimpleNamespace

# CUDA_VISIBLE_DEVICES=1, nnUNet_train 3d_fullres nnUNetTrainerV2 600 0 --npz --continue_training

SUPPORT_SCHEDULERS = ['cosine', 'tanh', 'step', 'multistep', 'plateau', 'poly', 'linear']


#             [0.5, 1]   0.01
# noise_args: lr_noise lr_noise_pct lr_noise_std seed
# warmup_args: warmup_lr warmup_epochs warmup_prefix
# cycle_args: lr_cycle_mul lr_cycle_decay lr_cycle_limit
# other_args: num_epochs min_lr  lr_k_decay  cooldown_epochs decay_rate decay_epochs patience_epochs eval_metric

#    cosine:  num_epochs min_lr  lr_k_decay  cooldown_epochs
#      tanh:  num_epochs min_lr  lr_k_decay  cooldown_epochs
#      poly:  num_epochs min_lr  lr_k_decay  cooldown_epochs decay_rate
#    linear:             min_lr                                         decay_epochs
#      step:                                                 decay_rate decay_epochs
# multistep:                                                 decay_rate decay_epochs
#   plateau:             min_lr                              decay_rate              patience_epochs eval_metric

if __name__ == '__main__':

    test_tensor = torch.rand(3, 4, 5, requires_grad=True)
    test_opt = SGD([test_tensor], lr=1e-3)

    for sche in SUPPORT_SCHEDULERS:
        args = dict(
            lr_noise=None,
            lr_noise_pct=0.67,
            lr_noise_std=1.,
            seed=42,

            warmup_epochs=20,
            warmup_lr=1e-5,
            warmup_prefix=True,

            lr_cycle_mul=1.0,       # mul > 1 - t_init / t_max
            lr_cycle_decay=0.6,
            lr_cycle_limit=3,
            cooldown_epochs=10,

            min_lr=1e-7,
            num_epochs=300,
            decay_epochs=100,
            decay_rate=0.5,
            lr_k_decay=1.0,

            eval_metric='',
            patience_epochs=20
        )
        args['lr_policy'] = sche
        if sche == 'cosine' or sche == 'tanh' or sche == 'poly':
            args['num_epochs'] = 300
        elif sche == 'step':
            args['num_epochs'] = 1000
            args['decay_epochs'] = 100
            args['decay_rate'] = 0.6
        elif sche == 'multistep':
            args['num_epochs'] = 1000
            args['decay_epochs'] = [150, 350, 650, 1000]
            args['decay_rate'] = 0.6
        elif sche == 'linear':
            args['decay_epochs'] = 1000
        else:
            args['decay_epochs'] = 100

        used_sche, num_epochs = create_scheduler(SimpleNamespace(**args), test_opt)
        print(num_epochs)
        try:
            data_list = [used_sche._get_lr(t) for t in range(1, num_epochs+1)]
        except AttributeError as e:
            print(type(used_sche), sche)
            print(type(e), e)
            continue
        plot_2d(range(1, num_epochs+1), data_list, fig_title=sche)







