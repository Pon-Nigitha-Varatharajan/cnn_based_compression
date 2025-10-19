import torch
import torch.nn as nn

class ComCNN(nn.Module):
    """Compression CNN: 3-layer block as in paper"""
    def __init__(self, in_channels=3):
        super(ComCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x