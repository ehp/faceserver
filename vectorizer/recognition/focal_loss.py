# -*- coding: utf-8 -*-
"""
   Copyright 2019 Petr Masopust, Aprar s.r.o.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Adopted code from https://github.com/ronghuaiyang/arcface-pytorch

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
