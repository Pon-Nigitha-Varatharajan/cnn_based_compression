import torch
import torch.nn as nn

class RecCNN(nn.Module):
    """Reconstruction CNN: residual network ~20 layers"""
    def __init__(self, in_channels=3, num_layers=20):
        super(RecCNN, self).__init__()
        layers = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        # add (num_layers - 2) conv blocks with BN
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ]
        layers.append(nn.Conv2d(64, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)