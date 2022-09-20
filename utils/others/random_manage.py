import numpy as np
import torch
import random
import os


class RandomManage:
    @staticmethod
    def init_global_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed + 1)
        random.seed(seed + 2)
        np.random.seed(seed + 3)
        torch.random.manual_seed(seed + 4)  # 配置pytorch中默认的随机seed
        # torch.random.seed()     # 得到一个随机的seed
        torch.cuda.manual_seed(seed + 5)  # 当前GPU
        torch.cuda.manual_seed_all(seed + 6)  # 所有GPU

    def __init__(self, base_seed):
        # print('pytorch initial seed:', torch.random.initial_seed())
        self.base_seed = base_seed
        self.python_random = random.Random(base_seed)
        self.numpy_random = np.random.RandomState(base_seed)  # seed

    def set_base_seed(self, seed):
        self.base_seed = seed

    def set_torch_seed(self, seed):
        torch.random.manual_seed(seed+self.base_seed)

    def set_python_seed(self, seed):
        random.seed(seed+self.base_seed)

    def set_numpy_seed(self, seed):
        np.random.seed(seed+self.base_seed)

    def set_smart_python_random(self, seed):
        self.python_random.seed(seed+self.base_seed)

    def set_smart_numpy_random(self, seed):
        self.numpy_random.seed(seed+self.base_seed)

    def get_smart_python_random(self):
        return self.python_random

    def get_smart_numpy_random(self):
        return self.numpy_random


RANDOMMANAGE = RandomManage(1008)

