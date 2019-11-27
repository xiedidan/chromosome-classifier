import torch
import torchvision
from torch import nn

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 32*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 8*4, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(8, 1*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        )

        # this is a regression problem
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        #x = self.encoder.layer3(x)
        #x = self.encoder.layer4(x)

        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x_1 = self.encoder.layer1(x)
        x_2 = self.encoder.layer2(x_1)

        return x_1, x_2

    def _calc_size(self, dummy_size):
        x = torch.ones(dummy_size)
        print(x.shape)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        print(x.shape)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        #x = self.encoder.layer3(x)
        #x = self.encoder.layer4(x)
        print(x.shape)

        x = self.decoder(x)
        print(x.shape)

if __name__ == '__main__':
    ae = autoencoder()
    ae._calc_size((4, 3, 256, 256))
