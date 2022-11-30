import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from .config import Config, ConfigList, ConfigDict, configurable

from .convs import MaskedConv1d, HighwayConv1d, ConvBlock, HighwayBlock
from .encoders_conv import ConvEncoder, HighwayEncoder
from .embedding import Embed, Regressor
from .extractor import AverageExtractor, RNNExtractor



@configurable()
class QuestionClassifier(nn.Module):
    def __init__(self, text_embed=Embed(num_symbols=512), 
                 encoders=[HighwayEncoder()], 
                 extractor=AverageExtractor(), 
                 regressor=Regressor(num_symbols=44)):
        super().__init__()
        self.text_embed = text_embed
        self.encoders = encoders
        self.extractor = extractor
        self.regressor = regressor
        
    def forward(self, x, x_len=None):
        bsize, tsize = x.shape
        if x_len is None:
            x_len = torch.full([bsize], tsize, dtype=torch.int32, device=x.device)
        x, x_len = self.text_embed(x, x_len)
        for encoder in self.encoders:
            x, x_len = encoder(x, x_len)
        x = self.extractor(x, x_len)
        x = self.regressor(x)
        return x
    

