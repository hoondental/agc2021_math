import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import OrderedDict
from .convs import MaskedConv1d

from .config import Config, ConfigList, ConfigDict, configurable


@configurable()
class Embed(nn.Module):
    def __init__(self, num_symbols=512, embedding_dim=256, num_upsample=1, padding_idx=None):
        super().__init__()
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.num_upsample = num_upsample
        self.padding_idx = padding_idx
        self.embed = nn.Embedding(self.num_symbols, embedding_dim * num_upsample, padding_idx)


    def forward(self, x, x_len=None):
        batch_size, seq_len = x.shape
        x = self.embed(x).transpose(1, 2)
        xs = torch.chunk(x, self.num_upsample, 1)
        y = torch.stack(xs, dim=-1)
        y = y.reshape(batch_size, self.embedding_dim, self.num_upsample * seq_len)
        y_len = None if x_len is None else x_len * self.num_upsample
        if torch.is_anomaly_enabled():
            if torch.isnan(x).any():
                raise Exception(type(self), 'nan in x', x)
            if torch.isnan(y).any():
                raise Exception(type(self), 'nan in y', y)
        return y, y_len
    

        
       
        
        
@configurable()        
class Regressor(nn.Module):
    def __init__(self, num_symbols=44, embedding_dim=1024, hidden_dims=[], external_embed=None):
        super().__init__()
        self.num_symbols = num_symbols
        self.embedding = embedding_dim
        self.hidden_dims = hidden_dims
        
        _layers = []
        _in_dim = embedding_dim
        _out_dim = embedding_dim
        for _out_dim in hidden_dims:
            _layers.append(nn.Linear(_in_dim, _out_dim))
            _in_dim = _out_dim
        self.layers = nn.ModuleList(_layers)
        self.linear = nn.Linear(_out_dim, self.num_symbols)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.linear(x)
        return x
 


        
               
        
        
        
        
        
        
       
