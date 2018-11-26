#!/usr/bin/python
# coding:utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device())).transpose(1, 2)


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                               (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal

class PosEncoder2(nn.Module):
    def __init__(self, d_model, dropout, max_len=400):
        super(PosEncoder2, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000)/d_model)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(dim=0).transpose(2,1)    # [1, max_len, d_model]
        # print(pe.shape)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x : [batch, d_model, sequence_len]
        """
        # print(x.shape)
        x = x + Variable(self.pe[:, :, :x.size(2)], requires_grad=False)    # [1, sequence_len, d_model]
        return self.dropout(x)