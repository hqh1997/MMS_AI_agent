import h5py
import math
import torch
import torch.nn as nn


def l1_reg(model):
    reg = None
    for W in model.parameters():
        if reg is None:
            reg = torch.abs(W).sum()
        else:
            reg = reg + torch.abs(W).sum()
    return reg

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
