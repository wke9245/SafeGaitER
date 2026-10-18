"""Level-specific encoders of recipient-visible release packet fields."""

import numpy as np
import torch
from torch import nn

from attacks.descriptors import LEVELS, onehot
from opengait.modeling.models.diagnostic_mapper import CATEGORIES
from opengait.modeling.models.release_policy import validate_packet


def mlp(widths):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def field_vector(packet):
    validate_packet(packet)
    level = packet["level"]
    indicator = onehot(LEVELS.index(level), 3)
    if level == "low":
        values = [*packet["spatial_scores"], *packet["temporal_scores"], *indicator]
    else:
        diagnostic = onehot(CATEGORIES.index(packet["diagnostic"]), 9)
        if level == "medium":
            values = [
                *onehot(packet["dominant_region"], 3),
                *onehot(packet["dominant_interval"], 4),
                *indicator,
                *diagnostic,
            ]
        else:
            values = [*indicator, *diagnostic]
    return np.asarray(values, dtype=np.float32)


class NativePacketEncoder(nn.Module):
    """Encode each packet before mean pooling over repeated releases."""

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                mlp([10, 64, 64]),
                mlp([19, 64, 64]),
                mlp([12, 32, 64]),
            ]
        )

    def forward(self, fields, levels):
        if fields.ndim != 3 or fields.shape[-1] != 19 or levels.shape != fields.shape[:1]:
            raise ValueError(
                "Native inputs require [sequences, releases, 19] fields and level labels."
            )
        if not torch.isfinite(fields).all() or ((levels < 0) | (levels > 2)).any():
            raise ValueError("Invalid native-field values or release levels.")
        output = fields.new_zeros(fields.shape[0], fields.shape[1], 64)
        for level, (width, encoder) in enumerate(zip((10, 19, 12), self.encoders)):
            mask = levels == level
            if mask.any():
                output[mask] = encoder(fields[mask, :, :width])
        return output.mean(dim=1)


class MembershipAttack(nn.Module):
    def __init__(self, family, fusion=False):
        super().__init__()
        if family not in ("score", "descriptor", "native"):
            raise ValueError("Unknown membership attack family.")
        self.family, self.fusion = family, fusion
        if family == "native":
            self.encoder = NativePacketEncoder()
            self.head = mlp([65 if fusion else 64, 32, 1])
        else:
            dimension = 1 if family == "score" else 36 + int(fusion)
            self.head = mlp([dimension, 128, 64, 32, 1])

    def forward(self, batch):
        if self.family == "score":
            values = batch["scores"]
        else:
            values = (
                self.encoder(batch["fields"], batch["levels"])
                if self.family == "native"
                else batch["descriptor"]
            )
            if self.fusion:
                values = torch.cat((values, batch["scores"]), dim=1)
        return self.head(values).squeeze(-1)


class NativeLinkageAttack(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NativePacketEncoder()
        self.head = mlp([128, 64, 1])

    def forward(self, first, second):
        left = self.encoder(first["fields"], first["levels"])
        right = self.encoder(second["fields"], second["levels"])
        return self.head(torch.cat(((left - right).abs(), left * right), dim=1)).squeeze(-1)
