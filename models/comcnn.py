import torch
import torch.nn as nn

class ComCNN(nn.Module):
    """Compression CNN: 3-layer block as in paper"""
    def __init__(self, in_channels=3):
        super(ComCNN, self).__init__()
        # Layer 1: 64 filters of 3x3, ReLU
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Layer 2: 64 filters of 3x3, stride=2, ReLU (downscaling)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Layer 3: 3 filters of 3x3 (back to original channels)
        self.layer3 = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # This downscales by 2: 40x40 -> 20x20
        x = self.layer3(x)
        return x