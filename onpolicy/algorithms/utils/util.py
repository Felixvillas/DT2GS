import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class Roll_Mean_Std:
    def __init__(self, init_mean=0, init_std=1, alpha=1e-2) -> None:
        self.mean = init_mean
        self.std = init_std
        self.alpha = alpha

    def get_mean_std(self):
        return self.mean, self.std

    def update(self, target_mean, target_std, step=1):
        for _ in range(step):
            self.mean += self.alpha * (target_mean - self.mean)
            self.std += self.alpha * (target_std - self.std)
            