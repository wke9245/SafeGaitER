import torch
import torch.nn.functional as F
from torch import nn


def local_descriptor(q, omega, alpha, contribution):
    normalized = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    expanded_alpha = alpha[:, :, None].expand_as(omega)
    return torch.cat(
        (normalized, omega[..., None], expanded_alpha[..., None], contribution[..., None]), dim=-1
    )


def stability_target(e1, e2):
    return ((1 + F.cosine_similarity(e1, e2, dim=-1)) / 2).detach()


class StabilityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(132, 704),
            nn.ReLU(),
            nn.Linear(704, 352),
            nn.ReLU(),
            nn.Linear(352, 1),
            nn.Sigmoid(),
        )

    def forward(self, descriptor, confidence):
        p = confidence[:, None, None, None].expand(*descriptor.shape[:-1], 1)
        return self.network(torch.cat((descriptor, p), dim=-1)).squeeze(-1)
