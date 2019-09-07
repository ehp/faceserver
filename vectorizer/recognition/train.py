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
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms as T

from recognition.angle import AngleLinear, CosFace, SphereFace, ArcFace, AdaCos
from recognition.focal_loss import FocalLoss
from recognition.nets import get_net_by_name
from recognition.test import lfw_test2, get_pair_list, load_img_data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list_file, imagesize):
        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.labels = list(set([img.split()[1] for img in imgs]))
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        self.transforms = T.Compose([
            T.RandomResizedCrop(imagesize),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        data = data.convert(mode="RGB")
        data = self.transforms(data)
        cls = self.label_to_class(splits[1])
        return data.float(), cls

    def __len__(self):
        return len(self.imgs)

    def label_to_class(self, label):
        for idx, v in enumerate(self.labels):
            if v == label:
                return idx
        raise Exception("Unknown label %s" % label)

    def num_labels(self):
        return len(self.labels)


def main(args=None):
    parser = argparse.ArgumentParser(description='Training script for face identification.')

    parser.add_argument('--print_freq', help='Print every N batch (default 100)', type=int, default=100)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--net', help='Net name, must be one of resnet18, resnet34, resnet50, resnet101, resnet152, resnext50, resnext101 or spherenet',
                        default='resnet50')
    parser.add_argument('--lr_step', help='Learning rate step (default 10)', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate (default 0.1)', type=float, default=0.1)
    parser.add_argument('--weight_decay', help='Weight decay (default 0.0005)', type=float, default=0.0005)
    parser.add_argument('--easy_margin', help='Use easy margin (default false)', dest='easy_margin', default=False,
                        action='store_true')
    parser.add_argument('--parallel', help='Run training with DataParallel', dest='parallel',
                        default=False, action='store_true')
    parser.add_argument('--loss',
                        help='One of focal_loss. cross_entropy, arcface, cosface, sphereface, adacos (default cross_entropy)',
                        type=str, default='cross_entropy')
    parser.add_argument('--optimizer', help='One of sgd, adam (default sgd)',
                        type=str, default='sgd')
    parser.add_argument('--batch_size', help='Batch size (default 16)', type=int, default=16)
    parser.add_argument('--casia_list', help='Path to CASIA dataset file list (training)')
    parser.add_argument('--casia_root', help='Path to CASIA images (training)')
    parser.add_argument('--lfw_root', help='Path to LFW dataset (testing)')
    parser.add_argument('--lfw_pair_list', help='Path to LFW pair list file (testing)')
    parser.add_argument('--model_name', help='Name of the model to save')

    parser = parser.parse_args(args)

    is_cuda = torch.cuda.is_available()
    print('CUDA available: {}'.format(is_cuda))

    imagesize = 224
    model = get_net_by_name(parser.net)

    # TODO split training dataset to train/validation and stop using test dataset for acc
    train_dataset = Dataset(parser.casia_root, parser.casia_list, imagesize)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=parser.batch_size,
                                              shuffle=True,
                                              # pin_memory=True,
                                              num_workers=0)
    num_classes = train_dataset.num_labels()

    if parser.loss == 'focal_loss':
        metric_fc = nn.Linear(512, num_classes)
        criterion = FocalLoss(gamma=2, is_cuda=is_cuda)
    elif parser.loss == 'cross_entropy':
        metric_fc = nn.Linear(512, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        if is_cuda:
            criterion = criterion.cuda()
    elif parser.loss == 'cosface':
        metric_fc = AngleLinear(512, num_classes)
        criterion = CosFace(is_cuda=is_cuda)
    elif parser.loss == 'arcface':
        metric_fc = AngleLinear(512, num_classes)
        criterion = ArcFace(is_cuda=is_cuda)
    elif parser.loss == 'sphereface':
        metric_fc = AngleLinear(512, num_classes)
        criterion = SphereFace(is_cuda=is_cuda)
    elif parser.loss == 'adacos':
        metric_fc = AngleLinear(512, num_classes)
        criterion = AdaCos(num_classes, is_cuda=is_cuda)
    else:
        raise ValueError('Unknown loss %s' % parser.loss)

    if parser.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=parser.lr, weight_decay=parser.weight_decay)
    elif parser.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=parser.lr, weight_decay=parser.weight_decay)
    else:
        raise ValueError('Unknown optimizer %s' % parser.optimizer)

    scheduler = StepLR(optimizer, step_size=parser.lr_step, gamma=0.1)

    if parser.parallel:
        model = nn.DataParallel(model)
        metric_fc = nn.DataParallel(metric_fc)

    if is_cuda:
        model.cuda()
        metric_fc.cuda()

    print(model)
    print(metric_fc)

    identity_list = get_pair_list(parser.lfw_pair_list)
    img_data = load_img_data(identity_list, parser.lfw_root)

    print('{} train iters per epoch:'.format(len(trainloader)))

    start = time.time()
    last_acc = 0.0
    for i in range(parser.epochs):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            if is_cuda:
                data_input = data_input.cuda()
                label = label.cuda().long()
            feature = model(data_input)
            output = metric_fc(feature)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % parser.print_freq == 0:
                speed = parser.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {}'.format(time_str, i, ii, speed, loss.item()))

                start = time.time()

        scheduler.step()
        model.eval()
        acc = lfw_test2(model, identity_list, img_data, is_cuda=is_cuda)
        print('Accuracy: %f' % acc)
        if last_acc < acc:
            # TODO remove makedir
            os.makedirs('./ckpt', exist_ok=True)
            torch.save(model.state_dict(), './ckpt/' + parser.model_name + '_{}.pt'.format(i))
            torch.save(metric_fc.state_dict(), './ckpt/' + parser.model_name + '_metric_{}.pt'.format(i))


if __name__ == '__main__':
    main()
