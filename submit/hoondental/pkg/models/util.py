import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def to_onehot(x, num_cls, dim=-1):
    shape = list(x.shape)
    if dim < 0:
        dim = dim + len(shape) + 1
    shape.insert(dim, num_cls)
    zeros = torch.zeros(*shape, device=x.device)
    return zeros.scatter_(dim, x.unsqueeze(dim), 1)