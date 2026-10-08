import torch
import torch.nn.functional as F
from torch import nn
from .common import mlp
from .descriptors import field_vector, raw_vector, LEVELS


class NativeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.level_encoders = nn.ModuleList(
            [mlp([10, 64, 64]), mlp([19, 64, 64]), mlp([12, 32, 64])]
        )
        self.coarse_encoder = mlp([19, 64, 64])
        self.raw_projections = nn.ModuleList(
            [nn.Linear(240, 128), nn.Linear(30, 32), nn.Linear(240, 128), nn.Linear(30, 32)]
        )
        self.raw_trunk = mlp([320, 256, 128, 64])

    def forward(self, packets):
        device = next(self.parameters()).device
        # Group-by-level computation keeps only the released fields in each branch.
        output = torch.zeros(len(packets), 64, device=device)
        for level, encoder in zip(LEVELS, self.level_encoders):
            ids = [i for i, p in enumerate(packets) if p["level"] == level]
            if ids:
                vectors = torch.tensor(
                    [field_vector(p).tolist() for p in (packets[i] for i in ids)], device=device
                )
                output[ids] = encoder(vectors)
        coarse_ids = [i for i, p in enumerate(packets) if p["level"] == "coarse"]
        if coarse_ids:
            values = torch.tensor(
                [field_vector(packets[i]).tolist() for i in coarse_ids], device=device
            )
            output[coarse_ids] = self.coarse_encoder(values)
        ids = [i for i, p in enumerate(packets) if p["level"] == "raw"]
        if ids:
            vectors = torch.tensor([raw_vector(packets[i]).tolist() for i in ids], device=device)
            branches = vectors.split([240, 30, 240, 30], 1)
            output[ids] = self.raw_trunk(
                torch.cat(
                    [projection(x) for projection, x in zip(self.raw_projections, branches)], 1
                )
            )
        return output


class NativeMembership(nn.Module):
    def __init__(self, fusion=False):
        super().__init__()
        self.encoder = NativeEncoder()
        self.fusion = fusion
        self.head = mlp([65 if fusion else 64, 32, 1])

    def forward(self, releases, scores=None):
        if releases and isinstance(releases[0], list):
            q = len(releases[0])
            encoded = (
                self.encoder([p for group in releases for p in group])
                .reshape(len(releases), q, 64)
                .mean(1)
            )
        else:
            encoded = self.encoder(releases)
        if self.fusion:
            encoded = torch.cat((encoded, scores[:, None]), 1)
        return self.head(encoded).squeeze(-1)


def fit_native_membership(releases, scores, split, settings, device="cpu", seed=2025, fusion=False):
    from opengait.utils import seed_everything
    from opengait.evaluation.metrics import binary_metrics

    seed_everything(seed)
    model = NativeMembership(fusion).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    fit_indices, fit_labels = zip(*split["fit"])
    for epoch in range(settings["epochs"]):
        model.train()
        for ids in torch.randperm(len(fit_indices)).split(settings["batch_size"]):
            actual = [fit_indices[i] for i in ids.tolist()]
            y = torch.tensor(
                [fit_labels[i] for i in ids.tolist()], dtype=torch.float32, device=device
            )
            s = torch.tensor([scores[i] for i in actual], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(model([releases[i] for i in actual], s), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    test_indices, test_labels = zip(*split["test"])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_indices), settings["batch_size"]):
            actual = test_indices[start : start + settings["batch_size"]]
            s = torch.tensor([scores[i] for i in actual], dtype=torch.float32, device=device)
            predictions.extend(model([releases[i] for i in actual], s).cpu().tolist())
    return binary_metrics(test_labels, predictions)
