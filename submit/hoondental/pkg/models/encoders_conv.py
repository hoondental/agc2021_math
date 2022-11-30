import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from .config import Config, ConfigList, ConfigDict, configurable

from .convs import MaskedConv1d, HighwayConv1d, ConvBlock, HighwayBlock



@configurable()
class ConvEncoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, kernel_size=3, stride=1, 
                 num_blocks=1, num_layers=3, dilation_base=1, dilation_power=1, 
                 dropout_rate=0.0, activation=nn.ReLU(), padding='same', groups=1, bias=True):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dilation_power = dilation_power
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        embedding_dim = out_dim
        super().__init__()
        
        self.conv_in = MaskedConv1d(in_dim, embedding_dim, kernel_size=1, stride=stride, dilation=1)     
        self.blocks = nn.ModuleList([ConvBlock(embedding_dim, kernel_size, num_layers, dilation_base, dilation_power, 
                                               dropout_rate, activation, padding, groups, bias) \
                                       for i in range(num_blocks)])
        self.conv_out = MaskedConv1d(embedding_dim, out_dim, kernel_size=1, dilation=1)  
        


    def forward(self, x, x_len=None):
        x, x_len= self.conv_in(x, x_len)
        for block in self.blocks:
            x, x_len = block(x, x_len)
        x, x_len = self.conv_out(x, x_len)
        return x, x_len
    


@configurable()
class HighwayEncoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=256, kernel_size=3, stride=1, 
                 num_blocks=1, num_layers=5, dilation_base=1, dilation_power=1, 
                 dropout_rate=0.0, padding='same', groups=1, bias=True, normalization='batch'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dilation_power = dilation_power
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.normalization = normalization
        embedding_dim = out_dim
        super().__init__()
        
        self.conv_in = MaskedConv1d(in_dim, embedding_dim, kernel_size=1, stride=stride, dilation=1)  
        self.blocks = nn.ModuleList([HighwayBlock(embedding_dim, kernel_size, num_layers, dilation_base, dilation_power,
                                                  dropout_rate, padding, groups, bias, normalization=normalization) \
                                       for i in range(num_blocks)])
        self.conv_out = MaskedConv1d(embedding_dim, out_dim, kernel_size=1, dilation=1) 


    def forward(self, x, x_len=None):
        x, x_len= self.conv_in(x, x_len)
        for block in self.blocks:
            x, x_len = block(x, x_len)
        x, x_len = self.conv_out(x, x_len)
        return x, x_len
    

    

    
