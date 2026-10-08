import torch
import torch.nn.functional as F
from torch import nn
from .common import mlp


class InversionAttack(nn.Module):
    def __init__(self, raw=False):
        super().__init__()
        self.explanation_encoder = mlp([540, 512, 256] if raw else [19, 64, 256])
        self.project = nn.Linear(512, 4096)
        blocks = []
        for cin, cout in zip((512, 256, 128, 64), (256, 128, 64, 32)):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.BatchNorm3d(cout),
                nn.ReLU(),
            ]
        self.decoder = nn.Sequential(*blocks)
        self.output = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 1, 1))

    def forward(self, embedding, explanation, logits=False):
        encoded = self.explanation_encoder(explanation)
        h = self.project(torch.cat((embedding, encoded), 1)).reshape(-1, 512, 2, 2, 2)
        decoded = self.decoder(h)
        decoded = F.interpolate(decoded, size=(30, 64, 44), mode="nearest")
        value = self.output(decoded)[:, 0]
        return value if logits else value.sigmoid()
