import torch
from torch import nn


class SoftRouting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 128))
                for width in (10, 19, 12)
            ]
        )
        self.settings = config["release"]

    def weights(self, risk):
        low = 1 - torch.sigmoid(self.settings["kappa"] * (risk - self.settings["tau1"]))
        high = torch.sigmoid(self.settings["kappa"] * (risk - self.settings["tau2"]))
        return torch.stack((low, 1 - low - high, high), 1)

    def forward(self, branches, risk):
        vectors = torch.stack(
            [encoder(branch) for encoder, branch in zip(self.encoders, branches)], 1
        )
        return (vectors * self.weights(risk)[..., None]).sum(1)
