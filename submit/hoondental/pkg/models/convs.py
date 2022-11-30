import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from .config import Config, ConfigList, ConfigDict, configurable


@configurable()
class MaskedConv1d(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=3, stride=1, dilation=1, activation=None, dropout_rate=0.0, 
                 padding="same", groups=1, bias=True):
        min_len = 1 + (kernel_size - 1) * dilation
        self.pad_left = 0
        self.pad_right = 0
        if padding == "left":
            self.pad_left = (kernel_size - 1) * dilation
        elif padding == "right":
            self.pad_right = (kernel_size - 1) * dilation
        elif padding == "same":
            _pad = (kernel_size - 1) * dilation
            self.pad_left = _pad // 2
            self.pad_right = _pad - self.pad_left
        elif padding == "valid":
            pass
        else:
            raise ValueError("[MaskedConv1d]: padding shoule be 'valid', 'left', 'right' or 'same'. But received ", padding)
            
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, x, x_len=None):
        x = F.pad(x, pad=(self.pad_left, self.pad_right))
        if self.activation is not None:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.conv(x)
        if torch.is_anomaly_enabled():
            if x.isnan().any():
                print(type(self), 'nan in x:', x)
        if x_len is not None:
            x_len = torch.floor((x_len.float() + self.pad_left + self.pad_right - self.dilation * (self.kernel_size - 1) - 1 ) / \
                                   float(self.stride) + 1).to(torch.int32)
        return x, x_len



@configurable()    
class HighwayConv1d(nn.Module):
    def __init__(self, in_channels=128, kernel_size=3, stride=1, dilation=1, dropout_rate=0.0, padding='same', 
                 groups=1, bias=True, normalization=None):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.normalization = normalization  
        
        if padding != 'same' and padding != 'left' and padding != 'right':
            raise ValueError("[HighwayConv1d]: padding should be 'same' or 'left' or 'right'")
        min_len = 1 + (kernel_size - 1) * dilation
        self.pad_left = 0
        self.pad_right = 0
        if padding == "left":
            self.pad_left = (kernel_size - 1) * dilation
        elif padding == "right":
            self.pad_right = (kernel_size - 1) * dilation
        elif padding == "same":
            _pad = (kernel_size - 1) * dilation
            self.pad_left = _pad // 2
            self.pad_right = _pad - self.pad_left
        elif padding == "valid":
            pass

        super().__init__()
        if normalization is None or '':
            self.norm = None
        elif normalization == 'batch':
            self.norm = nn.BatchNorm1d(in_channels)
        elif normalization == 'syncbatch':
            self.norm = nn.SyncBatchNorm(in_channels)
        elif normalization == 'layer':
            self.norm = nn.LayerNorm([in_channels])
            
        self.conv = MaskedConv1d(in_channels, 2 * in_channels, kernel_size, stride=stride, dilation=dilation, activation=None, 
                                 dropout_rate=dropout_rate, padding=padding, groups=groups, bias=bias)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, x_len=None):
        if self.normalization == 'batch':
            x = self.norm(x)
        elif self.normalization == 'layer':
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        L, x_len = self.conv(x, x_len)      
        H1, H2 = torch.chunk(L, 2, 1)  # chunk at the feature dim
        H1 = self.sigmoid(H1)
        if torch.is_anomaly_enabled():
            if x.isnan().any():
                print(type(self), 'nan in x:', x)
            if L.isnan().any():
                print(type(self), 'nan in L:', L)
        return H1 * H2 + (1.0 - H1) * x, x_len

    
    
    
@configurable()
class ConvBlock(nn.Module):
    def __init__(self, in_channels=128, kernel_size=3, num_layers=3, dilation_base=1, dilation_power=1, dropout_rate=0.0, 
                 activation=nn.ReLU(), padding='same', groups=1, bias=True):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dilation_power = dilation_power
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        super().__init__()
        layers = []
        for i in range(num_layers):
            _dilation = dilation_base ** math.floor(i * dilation_power)
            _activation = activation if i != num_layers - 1 else None
            layers.append(MaskedConv1d(in_channels, in_channels, kernel_size, stride=1, dilation=_dilation, dropout_rate=dropout_rate, 
                                        activation=_activation, padding=padding, groups=groups, bias=bias))
        self.layers = nn.ModuleList(layers)
        

    def forward(self, x, x_len=None):
        for layer in self.layers:
            x, x_len = layer(x, x_len)
        return x, x_len
    
    
    
@configurable()
class HighwayBlock(nn.Module):
    def __init__(self, in_channels=128, kernel_size=3, num_layers=3, dilation_base=1, dilation_power=1, dropout_rate=0.0, 
                 padding='same', groups=1, bias=True, normalization=None):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation_base = dilation_base
        self.dilation_power = dilation_power
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.normalization = normalization
        
        super().__init__()
        layers = []
        for i in range(num_layers):
            _dilation = dilation_base ** math.floor(i * dilation_power)
            layers.append(HighwayConv1d(in_channels, kernel_size, stride=1, dilation=_dilation, dropout_rate=dropout_rate, 
                                        padding=padding, groups=groups, bias=bias, normalization=normalization))
        self.layers = nn.ModuleList(layers)
        

    def forward(self, x, x_len=None):
        for layer in self.layers:
            x, x_len = layer(x, x_len)
        return x, x_len

    





                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       
