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

   Adopted code from https://github.com/rainofmine/Face_Attention_Network
"""

import argparse
import collections
import os

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.utils.model_zoo as model_zoo

from identification.model_level_attention import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils.data import DataLoader
from identification.csv_eval import evaluate
from identification.dataloader import WIDERDataset, AspectRatioBasedSampler, collater, Resizer, Augmenter, Normalizer, \
    CSVDataset

is_cuda = torch.cuda.is_available()
print('CUDA available: {}'.format(is_cuda))

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

ckpt = False


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--wider_train', help='Path to file containing WIDER training annotations (see readme)')
    parser.add_argument('--wider_val',
                        help='Path to file containing WIDER validation annotations (optional, see readme)')
    parser.add_argument('--wider_train_prefix', help='Prefix path to WIDER train images')
    parser.add_argument('--wider_val_prefix', help='Prefix path to WIDER validation images')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size (default 2)', type=int, default=2)

    parser.add_argument('--model_name', help='Name of the model to save')
    parser.add_argument('--parallel', help='Run training with DataParallel', dest='parallel',
                        default=False, action='store_true')
    parser.add_argument('--pretrained', help='Pretrained model name in weight directory')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.wider_train is None:
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Resizer(), Augmenter(), Normalizer()]))
    else:
        dataset_train = WIDERDataset(train_file=parser.wider_train, img_prefix=parser.wider_train_prefix,
                                     transform=transforms.Compose([Resizer(), Augmenter(), Normalizer()]))

    if parser.wider_val is None:
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            print('Loading CSV validation dataset')
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Resizer(), Normalizer()]))
    else:
        print('Loading WIDER validation dataset')
        dataset_val = WIDERDataset(train_file=parser.wider_val, img_prefix=parser.wider_val_prefix,
                                   transform=transforms.Compose([Resizer(), Normalizer()]))

    print('Loading training dataset')
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    if parser.parallel:
        dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)
    else:
        dataloader_train = DataLoader(dataset_train, collate_fn=collater, batch_sampler=sampler)

    # Create the model_pose_level_attention
    if parser.depth == 18:
        retinanet = resnet18(num_classes=dataset_train.num_classes(), is_cuda=is_cuda)
    elif parser.depth == 34:
        retinanet = resnet34(num_classes=dataset_train.num_classes(), is_cuda=is_cuda)
    elif parser.depth == 50:
        retinanet = resnet50(num_classes=dataset_train.num_classes(), is_cuda=is_cuda)
    elif parser.depth == 101:
        retinanet = resnet101(num_classes=dataset_train.num_classes(), is_cuda=is_cuda)
    elif parser.depth == 152:
        retinanet = resnet152(num_classes=dataset_train.num_classes(), is_cuda=is_cuda)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if ckpt:
        retinanet = torch.load('')
        print('Loading checkpoint')
    else:
        print('Loading pretrained model')
        retinanet_dict = retinanet.state_dict()
        if parser.pretrained is None:
            pretrained_dict = model_zoo.load_url(model_urls['resnet' + str(parser.depth)])
        else:
            pretrained_dict = torch.load(parser.pretrained)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in retinanet_dict}
        retinanet_dict.update(pretrained_dict)
        retinanet.load_state_dict(retinanet_dict)
        print('load pretrained backbone')

    print(retinanet)
    if parser.parallel:
        retinanet = torch.nn.DataParallel(retinanet, device_ids=[0])
    if is_cuda:
        retinanet.cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    # optimizer = optim.SGD(retinanet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    if parser.parallel:
        retinanet.module.freeze_bn()
    else:
        retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    iters = 0
    for epoch_num in range(0, parser.epochs):

        retinanet.train()
        if parser.parallel:
            retinanet.module.freeze_bn()
        else:
            retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):

            iters += 1

            optimizer.zero_grad()

            img_data = data['img'].float()
            annot_data = data['annot']
            if is_cuda:
                img_data = img_data.cuda()
                annot_data = annot_data.cuda()

            print("GPU memory allocated: %d max memory allocated: %d memory cached: %d max memory cached: %d" % (
            torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.max_memory_allocated() / 1024 ** 2,
            torch.cuda.memory_cached() / 1024 ** 2, torch.cuda.max_memory_cached() / 1024 ** 2))
            classification_loss, regression_loss, mask_loss = retinanet([img_data, annot_data])

            del img_data
            del annot_data

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            mask_loss = mask_loss.mean()

            loss = classification_loss + regression_loss + mask_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss.item()))

            epoch_loss.append(float(loss.item()))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | '
                'mask_loss {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), float(mask_loss),
                    np.mean(loss_hist)))

            del classification_loss
            del regression_loss
            del loss

        if parser.wider_val is not None:
            print('Evaluating dataset')
            evaluate(dataset_val, retinanet, is_cuda=is_cuda)

        scheduler.step(np.mean(epoch_loss))

        # TODO remove makedir
        os.makedirs('./ckpt', exist_ok=True)
        if parser.parallel:
            torch.save(retinanet.module, './ckpt/' + parser.model_name + '_{}.pt'.format(epoch_num))
        else:
            torch.save(retinanet, './ckpt/' + parser.model_name + '_{}.pt'.format(epoch_num))


if __name__ == '__main__':
    main()
