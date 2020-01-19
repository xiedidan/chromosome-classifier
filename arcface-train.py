import sys
import os
import pickle
import argparse
import itertools
from datetime import datetime
import gc
import csv

from apex import amp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from tensorboardX import SummaryWriter
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn.utils.fixes import logsumexp

from datasets.utils import BalancedBatchSampler
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from losses import OnlineTripletLoss
from metrics import AverageNonzeroTripletsMetric
from resnet import *
from transforms import *
from plot import *
from autoencoder import *
from arcface import *

# consts
class_mapping = {
    'chromosome': 0,
    'cell': 1,
    'impurity': 2
}

n_classes=len(class_mapping.keys())

# args
parser = argparse.ArgumentParser(description='ArcFace Training')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--img_size', default=256, type=int, help='image size')
parser.add_argument('--device', default='cuda:0', help='device (cuda / cpu)')
parser.add_argument('--data_root', default='/media/voyager/ssd-ext4/chromosome/', help='dataset root path')
parser.add_argument('--img_path', default='neg-chunk', help='image subpath')
parser.add_argument('--anno_paths', nargs='+', default=['neg-chunk.csv'], help='annotation filenames')
parser.add_argument('--lr', default=1e-7, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
parser.add_argument('--metric', default='arc_margin', help='training metric - arc_margin / add_margin / sphere')
parser.add_argument('--easy_margin', action='store_true', default=False, help='easy margin switch for arc_margin metric')
parser.add_argument('--scheduler_step', default=800, type=int, help='epoch to reduce learning rate')
parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='step scheduler gamma')
parser.add_argument('--checkpoint', help='checkpoint file path')
parser.add_argument('--round_id', default=1, type=int, help='round id')
parser.add_argument('--train_id', default=1, type=int, help='train id')
parser.add_argument('--epoches', default=1000, type=int, help='epoch to train')
parser.add_argument('--amp_opt', default='O0', help='AMP training optimization')
flags = parser.parse_args()

# config
batch_size = flags.batch_size # actual batch size = 128 // 3 * 3
img_size = flags.img_size

device = torch.device(flags.device)

data_root = flags.data_root # '/mnt/nvme/data/chromosome'
img_path = flags.img_path # 'neg-chunk'
anno_paths = flags.anno_paths

learning_rate = flags.lr
weight_decay = flags.weight_decay
metric = flags.metric
easy_margin = flags.easy_margin

scheduler_step = flags.scheduler_step
scheduler_gamma = flags.scheduler_gamma

checkpoint = flags.checkpoint
round_id = flags.round_id
train_id = flags.train_id
epoches = flags.epoches
amp_opt = flags.amp_opt

print(flags)

metric_fc = nn.Linear(512, n_classes)
if metric == 'arc_margin':
    metric_fc = ArcMarginProduct(512, n_classes, s=30, m=0.5, easy_margin=easy_margin)
elif metric == 'add_margin':
    metric_fc = AddMarginProduct(512, n_classes, s=30, m=0.35)
elif metric == 'sphere':
    metric_fc = SphereProduct(512, n_classes, m=4)

# simple dataset

class ChunkDataset(Dataset):
    def __init__(
        self,
        data_root,
        img_path,
        anno_paths,
        class_mapping,
        transform=None
    ):
        self.data_root = data_root
        self.img_path = img_path
        self.anno_paths = anno_paths
        self.class_mapping = class_mapping
        self.transform = transform
        
        self.anno_df = []
        
        for anno_path in self.anno_paths:
            anno_df = pd.read_csv(os.path.join(self.data_root, anno_path))
            self.anno_df.append(anno_df)
        
        self.anno_df = pd.concat(self.anno_df, axis=0)
        print(len(self.anno_df))
        
        self.anno_df = self.anno_df[self.anno_df['class']!='mixture']
        print(len(self.anno_df))
        
        self.labels = list(self.anno_df['class'])
        self.labels = torch.tensor([self.class_mapping[class_name] for class_name in self.labels])
        
        print(self.anno_df.head())
        
        self.total_len = len(self.anno_df)
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index):
        row = self.anno_df.iloc[index]
        img_file = os.path.join(self.data_root, self.img_path, row['filename'])
        
        img = Image.open(img_file)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.labels[index].item()

# data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    PadOrCrop(img_size),
    transforms.RandomAffine(30, translate=(0.2, 0.2), resample=PIL.Image.BILINEAR, fillcolor=255),
    transforms.ToTensor(),
    ChannelExpand()
])

train_dataset = ChunkDataset(
    data_root,
    img_path,
    anno_paths,
    class_mapping,
    transform=transform
)

train_sampler = BalancedBatchSampler(
    train_dataset.labels,
    n_classes=n_classes,
    n_samples=batch_size//n_classes
)

online_train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=8,
    pin_memory=True
)

# create a embedding resnet

class EmbeddingNet(nn.Module):
    def __init__(self, resnet, metric_fc, criterion):
        super(EmbeddingNet, self).__init__()
        self.resnet = resnet
        self.metric_fc = metric_fc
        self.criterion = criterion

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

resnet = models.resnet34(pretrained=True)
# TODO : focal loss
criterion = torch.nn.CrossEntropyLoss()
model = EmbeddingNet(resnet, metric_fc, criterion)

model = model.to(device)

# trainer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, scheduler_step, gamma=scheduler_gamma, last_epoch=-1)

if checkpoint:
    model.load_state_dict(torch.load(checkpoint))

model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt)

# train

iter_count = len(online_train_loader)
writer = SummaryWriter(comment='_{}-{}'.format(round_id, train_id))

model.train()

for epoch in range(epoches):
    print('epoch: {}/{}'.format(epoch+1, epoches))
    
    with tqdm(total=iter_count) as pbar:
        for iter_no, (data, target) in enumerate(online_train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            
            data = tuple(d.to(device) for d in data)
            if target is not None:
                target = target.to(device)
                
            optimizer.zero_grad()
            
            outputs = model(*data)
            
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
                
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            
            loss_outputs = model.criterion(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # loss.backward()
            optimizer.step()

            writer.add_scalar(
                'train/loss',
                loss.item(),
                epoch*iter_count+iter_no
            )
            
            pbar.update(1)
        
        scheduler.step()
        
        if not os.path.exists('./models'):
            os.mkdir('./models')
            
        torch.save(model.state_dict(), './models/EmbeddingNet-{}-{}.pth'.format(round_id, train_id))
