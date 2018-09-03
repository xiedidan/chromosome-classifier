# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import pickle
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from datasets.simple import *
from resnet import *

# config
num_classes = 2
pretrained = False # keep this to be false, we'll load weights manually
size = 448
original_size = 1024

start_epoch = 0
batch_size = 1
num_workers = 4
best_loss = float('inf')

# arg
parser = argparse.ArgumentParser(description='PyTorch ResNet Classifier Training')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--end_epoch', default=200, type=int, help='epcoh to stop training')
parser.add_argument('--transfer', action='store_true', help='use pretrained feature layers for transfer learning')
parser.add_argument('--lock_feature', action='store_true', help='lock featrue layers')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/chromosome/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
flags = parser.parse_args()

device = torch.device(flags.device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# data
trainTransform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomRotation(45, resample=Image.BILINEAR),
    transforms.CenterCrop(size=original_size),
    transforms.RandomResizedCrop(original_size, scale=(0.8, 1.0), ratio=(1., 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

valTransform = transforms.Compose([
    transforms.Grayscale(),
    transforms.CenterCrop(size=1024),
    transforms.Resize(size),
    transforms.ToTensor()
])

trainSet = SimpleDataset(
    root=flags.root,
    phase='train',
    transform=trainTransform
)

trainLoader = DataLoader(
    dataset=trainSet,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

valSet = SimpleDataset(
    root=flags.root,
    phase='val',
    transform=valTransform
)

valLoader = DataLoader(
    dataset=valSet,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# model
model = resnet101(
    pretrained,
    num_classes=num_classes
)

# transfer learning
if (flags.transfer):
    checkpoint = torch.load(flags.checkpoint)
    updating_parameters = model.transfer(checkpoint, flags.lock_feature)
elif (flags.resume):
    checkpoint = torch.load(flags.checkpoint)

    updating_parameters = model.transfer(checkpoint['net'], flags.lock_feature)
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
    updating_parameters = model.parameters()

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    updating_parameters,
    lr=flags.lr,
    momentum=0.9,
    weight_decay=5e-4
)

# pipeline
def train(epoch):
    print('Training Epoch: {}'.format(epoch))

    model.train()
    train_loss = 0

    for batch_index, (samples, gts) in enumerate(trainLoader):
        samples = samples.to(device)
        samples.contiguous()

        gts = gts.to(device)
        gts.contiguous()

        optimizer.zero_grad()

        output = model(samples)
        loss = criterion(output, gts)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('Epoch: {}, batch: {}, Sample loss: {:.5f}, batch avg loss: {:.5f}'.format(
            epoch,
            batch_index,
            loss.item(),
            train_loss / (batch_index + 1)
        ))

def val(epoch):
    print('Val')

    with torch.no_grad():
        model.eval()
        val_loss = 0

        for batch_index, (samples, gts) in enumerate(valLoader):
            samples = samples.to(device)
            samples.contiguous()

            gts = gts.to(device)
            gts.contiguous()

            output = model(samples)
            loss = criterion(output, gts)
            val_loss += loss.item()

        # save checkpoint
        global best_loss
        val_loss /= len(valLoader)

        if val_loss < best_loss:
            print('Saving checkpoint, best loss: {}'.format(val_loss))

            state = {
                'net': model.state_dict(),
                'loss': val_loss,
                'epoch': epoch,
            }
            
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            torch.save(state, './checkpoint/epoch_{:0>4}_loss_{:.6f}.pth'.format(
                epoch,
                val_loss
            ))

            best_loss = val_loss

# main loop
if __name__ == '__main__':
    for epoch in range(start_epoch, flags.end_epoch):
        train(epoch)
        val(epoch)
