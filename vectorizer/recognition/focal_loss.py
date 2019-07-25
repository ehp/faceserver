# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7, is_cuda=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()
        if is_cuda:
            self.ce = self.ce.cuda()

    def forward(self, inp, target):
        logp = self.ce(inp, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
