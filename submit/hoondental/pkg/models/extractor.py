import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import Config, ConfigList, ConfigDict, configurable

from .convs import MaskedConv1d, HighwayConv1d, ConvBlock, HighwayBlock
from .encoders_conv import ConvEncoder, HighwayEncoder
from .embedding import Embed, Regressor


        
@configurable()
class AverageExtractor(nn.Module):
    def __init__(self, in_dim=256, out_dim=1024, hidden_dims=[1024]):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        _layers = []
        _in_dim = in_dim
        _out_dim = in_dim
        for _out_dim in hidden_dims:
            _layers.append(MaskedConv1d(_in_dim, _out_dim, 1, activation=nn.ReLU()))
            _in_dim = _out_dim
        self.layers = nn.ModuleList(_layers)
        self.conv = MaskedConv1d(_out_dim, self.out_dim, 1)
        
    def forward(self, x, x_len=None):
        for layer in self.layers:
            x, x_len = layer(x, x_len)
        x, x_len = self.conv(x)
        bsize, fsize, tsize = x.shape
        if x_len is None:
            x_len = torch.full([bsize], tsize, dtype=torch.int32, device=x.device)
        tgrid  = torch.linspace(0, tsize - 1, tsize, dtype=torch.int32, device=x.device).unsqueeze(dim=0)
        mask0 = (tgrid < x_len.unsqueeze(-1)).to(torch.float32).unsqueeze(dim=1)
        x = (x * mask0).sum(dim=-1) / (x_len.to(torch.float32).unsqueeze(-1) + 1e-10)
        return x
    
    
@configurable()
class RNNExtractor(nn.Module):
    def __init__(self, in_dim=256, out_dim=1024, num_layers=1, rnn='gru', bidirectional=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.rnn = rnn
        self.bidirectional = bidirectional
        self.dsize = dsize = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim = int(out_dim / num_layers / dsize)
        assert hidden_dim * num_layers * dsize == out_dim
        if rnn.lower() == 'gru':
            self.layers = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn.lower() == 'lstm':
            self.layers = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)    
        else:
            raise Exception('Unsupported rnn type. ', rnn)
        
    def forward(self, x, x_len=None):
        bsize, fsize, tsize = x.shape
        device = x.device
        x = x.transpose(2, 1)
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        if x_len is None:
            x_len = torch.full([bsize], tsize, dtype=torch.int32, device=x.device)
        if self.rnn.lower() == 'gru':
            h0 = torch.zeros(self.dsize * self.num_layers, bsize, self.hidden_dim, device=device)
            x, hn = self.layers(x, h0)
        elif self.rnn.lower() == 'lstm':
            h0 = torch.zeros(self.dsize * self.num_layers, bsize, self.hidden_dim, device=device)
            c0 = torch.zeros(self.dsize * self.num_layers, bsize, self.hidden_dim, device=device)
            x, (hn, cn) = self.layers(x, (h0, c0))
        hn = hn.transpose(1, 0).reshape(bsize, self.out_dim)
        return hn
        
    