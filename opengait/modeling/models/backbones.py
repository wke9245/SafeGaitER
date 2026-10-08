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


class MultiScaleReliability(nn.Module):
    """Activation-guided feature refinement at multiple horizontal granularities."""

    def __init__(self, levels=(1, 2, 4)):
        super().__init__()
        self.levels = tuple(levels)
        self.scale = nn.Parameter(torch.ones(len(self.levels)))
        self.bias = nn.Parameter(torch.zeros(len(self.levels)))

    def forward(self, features):
        refined_levels = []
        for i, num_parts in enumerate(self.levels):
            gated_parts = []
            for part in features.chunk(num_parts, dim=2):
                energy = part.abs().mean(dim=(1, 2, 3), keepdim=True)
                reliability = torch.sigmoid(self.scale[i] * energy + self.bias[i])
                gated_parts.append(part * reliability)
            refined_levels.append(torch.cat(gated_parts, dim=2))
        return torch.stack(refined_levels, dim=0).mean(dim=0)


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

        self.sgra = MultiScaleReliability(levels=(1, 2, 4))

    def forward(self, x):
        x = self.middle(self.low(x))
        global_features = self.global_path(x)
        local_features = self.local_path(x)
        features = self.refine(global_features + local_features)
        return self.sgra(features)


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
