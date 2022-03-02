""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .linear_lr import LinearScheduler
from .cosine_lr import CosineLRScheduler
from .multistep_lr import MultiStepLRScheduler
from .plateau_lr import PlateauLRScheduler
from .poly_lr import PolyLRScheduler
from .step_lr import StepLRScheduler
from .tanh_lr import TanhLRScheduler

SUPPORT_SCHEDULERS = ['cosine', 'tanh', 'step', 'multistep', 'plateau', 'poly', 'linear']


#             [0.5, 1]   0.01
# noise_args: lr_noise lr_noise_pct lr_noise_std seed
# warmup_args: warmup_lr warmup_epochs warmup_prefix
# cycle_args: lr_cycle_mul lr_cycle_decay lr_cycle_limit
# other_args: num_epochs min_lr  lr_k_decay  cooldown_epochs decay_rate decay_epochs patience_epochs eval_metric

#    cosine:  num_epochs min_lr  lr_k_decay  cooldown_epochs
#      tanh:  num_epochs min_lr              cooldown_epochs
#      poly:  num_epochs min_lr  lr_k_decay  cooldown_epochs decay_rate
#    linear:             min_lr                                         decay_epochs
#      step:                                                 decay_rate decay_epochs
# multistep:                                                 decay_rate decay_epochs
#   plateau:             min_lr                              decay_rate              patience_epochs eval_metric


def create_scheduler(args, optimizer):
    assert args.lr_policy in SUPPORT_SCHEDULERS, f'Invalid loss: {args.lr_policy}'
    num_epochs = args.num_epochs

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=getattr(args, 'lr_noise_pct', 0.67),
        noise_std=getattr(args, 'lr_noise_std', 1.),        # useless
        noise_seed=getattr(args, 'seed', 42),
    )
    cycle_args = dict(
        cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(args, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(args, 'lr_cycle_limit', 1),
    )
    warmup_args = dict(
        warmup_t=getattr(args, 'warmup_epochs', 0),
        warmup_lr_init=getattr(args, 'warmup_lr', 0.),
        warmup_prefix=getattr(args, 'warmup_prefix', False),
    )

    # other_args = dict(
    #     t_initial=num_epochs,
    #     lr_min=args.min_lr,
    #     decay_t=args.decay_epochs,
    #     decay_rate=args.decay_rate,
    # )

    lr_scheduler = None
    if args.lr_policy == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=args.min_lr,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **warmup_args,
            **cycle_args,
            **noise_args
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.lr_policy == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=args.min_lr,
            t_in_epochs=True,
            **warmup_args,
            **cycle_args,
            **noise_args
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.lr_policy == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            **warmup_args,
            **noise_args
        )
    elif args.lr_policy == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            **warmup_args,
            **noise_args
        )
    elif args.lr_policy == 'plateau':
        mode = 'min' if 'loss' in getattr(args, 'eval_metric', '') else 'max'
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            lr_min=args.min_lr,
            mode=mode,
            cooldown_t=0,
            **warmup_args,
            **noise_args
        )
    elif args.lr_policy == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_epochs,
            lr_min=args.min_lr,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **warmup_args,
            **cycle_args,
            **noise_args
        )
        num_epochs = lr_scheduler.get_cycle_length() + args.cooldown_epochs
    elif args.lr_policy == 'linear':
        lr_scheduler = LinearScheduler(
            optimizer,
            decay_t=args.num_epochs,
            lr_min=args.min_lr,
            **warmup_args,
            **noise_args
        )

    return lr_scheduler, num_epochs


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

