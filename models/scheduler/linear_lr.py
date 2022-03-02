""" Step Scheduler

Basic step LR schedule with warmup, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import torch

from .scheduler import Scheduler


class LinearScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 lr_min: float = 1e-8,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=True,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        self.lr_min = lr_min

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            decay_t = float(self.decay_t)
            if self.warmup_prefix:
                t = t - self.warmup_t
                decay_t = decay_t - self.warmup_t
            gamma = 1.0 - t / decay_t
            lrs = [self.lr_min + (v-self.lr_min) * gamma for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def custom_test(self):
        lr_data = [self._get_lr(t)[0] for t in range(1000)]
        return lr_data


if __name__ == '__main__':
    from torch.optim import SGD, Optimizer
    from utils.others.img_io import plot_2d
    test_tensor = torch.rand(3, 4, 5, requires_grad=True)
    test_opt = SGD([test_tensor], lr=1)
    test_sch = LinearScheduler(test_opt, 1000)
    lr_all = test_sch.custom_test()
    plot_2d(range(1000), lr_all)
