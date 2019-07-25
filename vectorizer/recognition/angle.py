# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(AngleLinear, self).__init__()
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        x = F.normalize(input)
        W = F.normalize(self.W)
        return F.linear(x, W)


class AdaCos(nn.Module):
    def __init__(self, num_classes, m=0.50, is_cuda=True):
        super(AdaCos, self).__init__()
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.base_s = self.s
        self.m = m
        self.criterion = nn.CrossEntropyLoss()
        if is_cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, input, label):
# changed to fixed adacos
#        theta = torch.acos(torch.clamp(input, -1.0 + 1e-7, 1.0 - 1e-7))
#        one_hot = torch.zeros_like(input)
#        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#        with torch.no_grad():
#            B_avg = torch.where(one_hot < 1, torch.exp(self.s * input), torch.zeros_like(input))
#            B_avg = torch.sum(B_avg) / input.size(0)
#            theta_med = torch.median(theta)
#            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
#            # TODO why converge to infinity ?
#            self.s = torch.clamp(self.s, self.base_s / 2, self.base_s * 2)
#            print(self.s)
        output = self.s * input

        return self.criterion(output, label)


class ArcFace(nn.Module):
    def __init__(self, s=30.0, m=0.50, is_cuda=True):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.criterion = nn.CrossEntropyLoss()
        if is_cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, input, label):
        theta = torch.acos(torch.clamp(input, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = input * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return self.criterion(output, label)


class SphereFace(nn.Module):
    def __init__(self, s=30.0, m=1.35, is_cuda=True):
        super(SphereFace, self).__init__()
        self.s = s
        self.m = m
        self.criterion = nn.CrossEntropyLoss()
        if is_cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, input, label):
        theta = torch.acos(torch.clamp(input, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = input * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return self.criterion(output, label)


class CosFace(nn.Module):
    def __init__(self, s=30.0, m=0.35, is_cuda=True):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.criterion = nn.CrossEntropyLoss()
        if is_cuda:
            self.criterion = self.criterion.cuda()

    def forward(self, input, label):
        target_logits = input - self.m
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = input * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return self.criterion(output, label)
