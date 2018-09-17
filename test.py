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

from datasets.simple import *
from resnet import *
from utils.plot import *
from utils.export import *

# config
num_classes = 2
pretrained = False # keep this to be false, we'll load weights manually
size = 448
original_size = 1024

num_workers = 4

# arg
parser = argparse.ArgumentParser(description='PyTorch VGG Classifier Testing')
parser.add_argument('--batch_size', default=6, type=int, help='batch size')
parser.add_argument('--plot', action='store_true', help='plot result')
parser.add_argument('--save_file', default='./classification.csv', type=str, help='Filename to save results')
parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint file path')
parser.add_argument('--root', default='./rsna-pneumonia-detection-challenge/', help='dataset root path')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
flags = parser.parse_args()

device = torch.device(flags.device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# data
testTransform = transforms.Compose([
    transforms.Resize((size, size)),
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

checkpoint = torch.load(flags.checkpoint)
model.load_state_dict(checkpoint['net'])

model.to(device)

# pipeline
def test():
    with torch.no_grad():
        model.eval()

        for batch_index, samples in enumerate(testLoader):
            image_paths, images, gts = samples

            images = images.to(device)

            output = model(images)
            output = F.softmax(output, dim=len(output.size())-1)
            output = torch.argmax(
                output,
                dim=len(output.size())-1,
                keepdim=False
            )

            # for idx, image_path in enumerate(image_paths):
                # plot_leakage(image_path, output[idx])

            print('\nbatch: {}\noutput: {}\ngt: {}'.format(batch_index, output, gts))

# main
if __name__ == '__main__':
    test()
