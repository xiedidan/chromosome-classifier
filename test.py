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
from transforms import *
from plot import *

# config
num_classes = 2
size = 448
original_size = 1024

pretrained = False # keep this to be false, we'll load weights manually
num_workers = 4

# arg
parser = argparse.ArgumentParser(description='PyTorch ResNet Classifier Testing')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='/media/voyager/ssd-ext4/chromosome/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--plot', action='store_true', help='plot result')
flags = parser.parse_args()

device = torch.device(flags.device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# data
testTransform = transforms.Compose([
    transforms.Grayscale(),
    AutoLevel(0.7, 0.0001),
    transforms.CenterCrop(size=original_size),
    transforms.Resize(size),
    transforms.ToTensor()
])

testSet = SimpleDataset(
    root=flags.root,
    phase='test',
    transform=testTransform
)

testLoader = DataLoader(
    dataset=testSet,
    batch_size=flags.batch_size,
    shuffle=False,
    num_workers=num_workers
)

# model
model = resnet101(
    pretrained,
    num_classes=num_classes
)

# load parameter
checkpoint = torch.load(flags.checkpoint)
model.load_state_dict(checkpoint['net'])

model.to(device)

criterion = nn.CrossEntropyLoss()

# pipeline
def test():
    print('Test')

    with torch.no_grad():
        model.eval()
        test_loss = 0

        for batch_index, (paths, samples, gts) in enumerate(testLoader):
            samples = samples.to(device)
            samples.contiguous()

            gts = gts.to(device)
            gts.contiguous()

            if torch.cuda.device_count() > 1:
                output = nn.parallel.data_parallel(model, samples)
            else:
                output = model(samples)

            scores, results = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)

            # collect loss
            loss = criterion(output, gts)
            test_loss += loss.item()

            # plot
            if flags.plot:
                gts_array = gts.cpu().numpy()
                scores_array = scores.cpu().numpy()
                results_array = results.cpu().numpy()

                labels = ['gt: {}, pred: {}, s:{:.2f}'.format(gt, results_array[idx], scores_array[idx]) for idx, gt in enumerate(gts_array)]
                plot_classification(samples.cpu(), labels, 2)
        
        test_loss /= len(testLoader)
        print('Avg Loss: {}'.format(test_loss))

# main loop
if __name__ == '__main__':
    test()
