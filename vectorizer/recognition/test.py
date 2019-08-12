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

   Created on 18-5-30 下午4:55

   @author: ronghuaiyang
"""
import os
import argparse

from torch.utils.data import TensorDataset, DataLoader

from recognition.nets import get_net_by_depth
import torch
import numpy as np
from torch.nn import DataParallel
from PIL import Image
from torchvision import transforms as T

imagesize = 224
batch_size = 20


class Dataset(torch.utils.data.Dataset):
    def __init__(self, identity_list, root_path):
        self.identity_list = identity_list
        self.root_path = root_path

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.transforms = T.Compose([
            T.Resize(imagesize),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        a, b, label = self.identity_list[index]
        a_data = self.load_image(a)
        b_data = self.load_image(b)
        return a_data, b_data, label

    def load_image(self, p):
        img_path = os.path.join(self.root_path, p)
        data = Image.open(img_path)
        if data is None:
            return None
        data = data.convert(mode="RGB")
        data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.identity_list)


def get_pair_list(pair_list):
    print('Loading pair list')
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    return [line.split() for line in pairs]


def load_img_data(identity_list, root_path):
    dataset = Dataset(identity_list, root_path)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        # pin_memory=True,
                        num_workers=0)
    return loader


def lfw_test2(model, identity_list, img_data, is_cuda=True):
    print('Converting to features')
    sims = []
    labels = []
    max_size = len(img_data) * batch_size
    for i, sample in enumerate(img_data):
        if i % 10 == 0:
            print('%d of %d' % (i * batch_size, max_size))
        a_data, b_data, label = sample
        if is_cuda:
            a_data = a_data.cuda()
            b_data = b_data.cuda()

        a_output = model(a_data).detach().cpu().numpy()
        b_output = model(b_data).detach().cpu().numpy()

        for idx in range(batch_size):
            sim = cosin_metric(a_output[idx], b_output[idx])
            sims.append(sim)
            labels.append(np.bool(label[idx] == '1'))

    acc, th = cal_accuracy(sims, labels)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_acc, best_th


def main(args=None):
    parser = argparse.ArgumentParser(description='Testing script for face identification.')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152 or 20 for sphere', type=int,
                        default=50)
    parser.add_argument('--parallel', help='Run training with DataParallel', dest='parallel',
                        default=False, action='store_true')
    parser.add_argument('--model', help='Path to model')
    parser.add_argument('--batch_size', help='Batch size (default 50)', type=int, default=50)
    parser.add_argument('--lfw_root', help='Path to LFW dataset')
    parser.add_argument('--lfw_pair_list', help='Path to LFW pair list file')

    parser = parser.parse_args(args)

    is_cuda = torch.cuda.is_available()
    print('CUDA available: {}'.format(is_cuda))

    model = get_net_by_depth(parser.depth)

    if parser.parallel:
        model = DataParallel(model)

    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(parser.model))
    if is_cuda:
        model.cuda()

    identity_list = get_pair_list(parser.lfw_pair_list)
    img_data = load_img_data(identity_list, parser.lfw_root)

    model.eval()
    lfw_test2(model, identity_list, img_data, is_cuda=is_cuda)


if __name__ == '__main__':
    main()
