#!/usr/bin/python
# coding:utf-8


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.embedding import Embedding
from modules.encoder_blocks import EncoderBlock
from modules.intraction import CQAttention
from modules.layers import Initialized_Conv1d
from modules.pointer import Pointer

class QANet(nn.Module):
    def __init__(self, word_mat, char_mat,
                 c_max_len, q_max_len, d_model, train_cemb=False, pad=0,
                 dropout=0.1, num_head=1):  # !!! notice: set it to be a config parameter later.
        super().__init__()
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_mat)
        self.word_emb = nn.Embedding.from_pretrained(word_mat)
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)
        self.num_head = num_head
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = CQAttention(d_model=d_model)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList(
            [EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1)
             for _ in range(7)])
        self.out = Pointer(d_model)
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.ones_like(Cwid) *
                 self.PAD != Cwid).float()
        maskQ = (torch.ones_like(Qwid) *
                 self.PAD != Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)
