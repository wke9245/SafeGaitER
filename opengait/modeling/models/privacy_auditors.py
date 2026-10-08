import torch
import torch.nn.functional as F
from torch import nn


class PrivacyAuditors(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.membership = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.identity = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.linkage = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def loss(self, target, auxiliary, labels, lambda_y=0.5, lambda_l=0.5):
        joined = torch.cat((target, auxiliary))
        membership_labels = torch.cat(
            (
                torch.ones(len(target), device=target.device),
                torch.zeros(len(auxiliary), device=target.device),
            )
        )
        mem = F.binary_cross_entropy_with_logits(
            self.membership(joined).squeeze(-1), membership_labels
        )
        identity = F.cross_entropy(self.identity(target), labels)
        n = len(target)
        left, right = torch.triu_indices(n, n, offset=1, device=target.device)
        if len(left):
            pairs = torch.cat(
                ((target[left] - target[right]).abs(), target[left] * target[right]), 1
            )
            link_labels = (labels[left] == labels[right]).float()
            link = F.binary_cross_entropy_with_logits(self.linkage(pairs).squeeze(-1), link_labels)
        else:
            link = target.sum() * 0
        return mem + lambda_y * identity + lambda_l * link
