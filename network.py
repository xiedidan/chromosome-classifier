import torch
import torch.nn as nn

from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from losses import OnlineTripletLoss

class EmbeddingNet(nn.Module):
    def __init__(self, resnet, margin):
        super(EmbeddingNet, self).__init__()
        self.resnet = resnet
        self.margin = margin
        self.criterion = OnlineTripletLoss(self.margin, RandomNegativeTripletSelector(self.margin))

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
