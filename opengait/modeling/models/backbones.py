"""Frame feature encoders for hierarchical and GaitPart evidence extraction."""

import torch
from torch import nn


class FocalConv(nn.Module):
    def __init__(self, cin, cout, parts):
        super().__init__()
        self.parts = parts
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)

    def forward(self, x):
        return torch.cat([self.conv(part) for part in x.chunk(self.parts, dim=2)], dim=2)


class HierarchicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.global_path = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.local_path = FocalConv(64, 128, 8)
        self.refine = nn.Sequential(
            nn.LeakyReLU(0.1), nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.middle(self.low(x))
        return self.refine(self.global_path(x) + self.local_path(x))


class GaitPartEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            FocalConv(32, 64, 4),
            nn.LeakyReLU(0.01),
            FocalConv(64, 64, 4),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            FocalConv(64, 128, 8),
            nn.LeakyReLU(0.01),
            FocalConv(128, 128, 8),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        return self.layers(x)
