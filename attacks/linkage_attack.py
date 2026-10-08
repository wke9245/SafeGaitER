from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .common import mlp
from .descriptors import statistical_descriptor
from .native_field_attack import NativeEncoder
from opengait.evaluation.metrics import binary_metrics
from opengait.utils import seed_everything


def make_pairs(records, indices, per_class, seed=2025):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i in indices:
        groups[records[i].subject].append(i)
    positive_ids = [s for s, values in groups.items() if len(values) >= 2]
    subjects = sorted(groups)
    if not positive_ids or len(subjects) < 2:
        raise ValueError(
            "Linkage evaluation requires at least two identities and two sequences for a positive pair."
        )
    pairs = []
    for _ in range(per_class):
        subject = rng.choice(positive_ids)
        a, b = rng.choice(groups[subject], 2, replace=False)
        pairs.append((int(a), int(b), 1))
        s1, s2 = rng.choice(subjects, 2, replace=False)
        pairs.append((int(rng.choice(groups[s1])), int(rng.choice(groups[s2])), 0))
    rng.shuffle(pairs)
    return pairs


class LinkageAttack(nn.Module):
    def __init__(self, native=False):
        super().__init__()
        self.native = native
        self.encoder = NativeEncoder() if native else mlp([36, 128, 64])
        self.head = mlp([128, 64, 1])

    def encode(self, releases):
        if isinstance(releases[0], list):
            q = len(releases[0])
            if self.native:
                encoded = self.encoder([p for group in releases for p in group])
                return encoded.reshape(len(releases), q, 64).mean(1)
            device = next(self.parameters()).device
            values = torch.tensor(
                np.stack(
                    [
                        np.stack([statistical_descriptor(p) for p in group]).mean(0)
                        for group in releases
                    ]
                ),
                device=device,
            )
            return self.encoder(values)
        if self.native:
            return self.encoder(releases)
        device = next(self.parameters()).device
        values = torch.tensor(
            np.stack([statistical_descriptor(p) for p in releases]), device=device
        )
        return self.encoder(values)

    def forward(self, first, second):
        z1, z2 = self.encode(first), self.encode(second)
        return self.head(torch.cat(((z1 - z2).abs(), z1 * z2), 1)).squeeze(-1)


def fit_linkage(releases, fit_pairs, test_pairs, settings, device="cpu", seed=2025, native=False):
    seed_everything(seed)
    model = LinkageAttack(native).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    batch_size = settings.get("linkage_batch_size", 512)
    for epoch in range(settings["epochs"]):
        model.train()
        for indices in torch.randperm(len(fit_pairs)).split(batch_size):
            batch = [fit_pairs[i] for i in indices.tolist()]
            first, second = [releases[a] for a, b, y in batch], [releases[b] for a, b, y in batch]
            y = torch.tensor([y for a, b, y in batch], device=device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(model(first, second), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for start in range(0, len(test_pairs), batch_size):
            batch = test_pairs[start : start + batch_size]
            scores.extend(
                model([releases[a] for a, b, y in batch], [releases[b] for a, b, y in batch])
                .cpu()
                .tolist()
            )
            labels.extend(y for a, b, y in batch)
    return binary_metrics(labels, scores)
