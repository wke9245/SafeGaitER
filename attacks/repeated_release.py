"""Independent membership and native-field linkage attacks at fixed query budgets."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

from attacks.descriptors import LEVELS, statistical_descriptor
from attacks.native_fields import MembershipAttack, NativeLinkageAttack, field_vector
from opengait.evaluation.observations import iter_observations
from opengait.utils import seed_everything

QUERY_BUDGETS = (1, 5, 10, 20, 50, 100, 200)
ATTACK_SETTINGS = {
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "membership_batch_size": 256,
    "linkage_batch_size": 512,
}
CHANNELS = {
    "score_only": ("score", False),
    "descriptor_packet": ("descriptor", False),
    "descriptor_fusion": ("descriptor", True),
    "native_packet": ("native", False),
    "native_fusion": ("native", True),
}


def binary_metrics(labels, logits):
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    if (
        labels.ndim != 1
        or logits.shape != labels.shape
        or set(labels.tolist()) != {0, 1}
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Attack metrics require finite scores and both binary classes.")
    fpr, tpr, _ = roc_curve(labels, logits, drop_intermediate=False)
    return {
        "auc": float(roc_auc_score(labels, logits)),
        "tpr_at_1pct_fpr_pct": 100.0 * float(np.interp(0.01, fpr, tpr)),
    }


class ReleaseFeatures:
    """CPU feature storage; minibatches are transferred to the selected CUDA device."""

    def __init__(self, records, budgets):
        self.keys = [r.key for r in records]
        self.index = {key: i for i, key in enumerate(self.keys)}
        self.budgets = tuple(sorted(budgets))
        count, maximum = len(records), max(self.budgets)
        self.fields = np.zeros((count, maximum, 19), dtype=np.float32)
        self.levels = np.empty(count, dtype=np.int64)
        self.scores = np.empty((count, 1), dtype=np.float32)
        self.descriptors = {q: np.empty((count, 36), dtype=np.float32) for q in self.budgets}

    @classmethod
    def read(cls, path, records, metadata, budgets):
        if (
            not budgets
            or len(set(budgets)) != len(budgets)
            or any(q not in QUERY_BUDGETS for q in budgets)
        ):
            raise ValueError("Select distinct query budgets from 1, 5, 10, 20, 50, 100, 200.")
        if metadata.get("release_count", 1) < max(budgets):
            raise ValueError("The inference artifact contains too few independent releases.")
        features = cls(records, budgets)
        maximum = max(budgets)
        for entry in iter_observations(path, records, metadata):
            index = features.index[entry["key"]]
            packets = entry.get("release_packets", [entry["safe_packet"]])[:maximum]
            descriptor_sum = np.zeros(36, dtype=np.float64)
            for query, packet in enumerate(packets, 1):
                native = field_vector(packet)
                features.fields[index, query - 1, : len(native)] = native
                descriptor_sum += statistical_descriptor(packet)
                if query in features.descriptors:
                    features.descriptors[query][index] = descriptor_sum / query
            features.scores[index, 0] = entry["score"]
            features.levels[index] = LEVELS.index(entry["safe_packet"]["level"])
        return features

    def batch(self, indices, budget, device, family):
        indices = np.asarray(indices, dtype=np.int64)
        result = {"scores": torch.from_numpy(self.scores[indices]).to(device)}
        if family == "native":
            result.update(
                fields=torch.from_numpy(self.fields[indices, :budget]).to(device),
                levels=torch.from_numpy(self.levels[indices]).to(device),
            )
        elif family == "descriptor":
            result["descriptor"] = torch.from_numpy(self.descriptors[budget][indices]).to(device)
        return result


def fit_binary(model, fit_labels, test_labels, fit_batch, test_batch, batch_size, device, seed):
    """Fit a fresh attacker for the fixed final-epoch schedule."""
    if set(np.asarray(fit_labels).tolist()) != {0, 1}:
        raise ValueError("Attacker fitting requires both membership or pair classes.")
    seed_everything(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ATTACK_SETTINGS["lr"],
        weight_decay=ATTACK_SETTINGS["weight_decay"],
    )
    labels = torch.as_tensor(fit_labels, dtype=torch.float32, device=device)
    for _ in range(ATTACK_SETTINGS["epochs"]):
        model.train()
        for indices in torch.randperm(len(labels)).split(batch_size):
            logits = fit_batch(model, indices.numpy())
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels[indices.to(device)]
            )
            if not torch.isfinite(loss):
                raise ValueError("Attacker optimization produced a nonfinite loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval().requires_grad_(False)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(test_labels)))
            predictions.append(test_batch(model, indices).detach().cpu().numpy())
    logits = np.concatenate(predictions)
    return model, binary_metrics(test_labels, logits), logits


def fit_membership(features, partitions, budget, family, fusion, device, seed):
    indices, labels = {}, {}
    for phase in ("fit", "evaluate"):
        indices[phase] = np.asarray([features.index[r.key] for r, _ in partitions[phase]])
        labels[phase] = np.asarray([label for _, label in partitions[phase]])

    def batch(phase, model, selected):
        return model(features.batch(indices[phase][selected], budget, device, family))

    seed_everything(seed)
    return fit_binary(
        MembershipAttack(family, fusion),
        labels["fit"],
        labels["evaluate"],
        lambda model, selected: batch("fit", model, selected),
        lambda model, selected: batch("evaluate", model, selected),
        ATTACK_SETTINGS["membership_batch_size"],
        device,
        seed,
    )


def fit_linkage(features, partitions, budget, device, seed):
    indices, labels = {}, {}
    for phase in ("fit", "evaluate"):
        indices[phase] = np.asarray(
            [
                (features.index[first], features.index[second])
                for first, second, _ in partitions[phase]
            ]
        )
        labels[phase] = np.asarray([label for _, _, label in partitions[phase]])

    def batch(phase, model, selected):
        pairs = indices[phase][selected]
        return model(
            features.batch(pairs[:, 0], budget, device, "native"),
            features.batch(pairs[:, 1], budget, device, "native"),
        )

    seed_everything(seed)
    return fit_binary(
        NativeLinkageAttack(),
        labels["fit"],
        labels["evaluate"],
        lambda model, selected: batch("fit", model, selected),
        lambda model, selected: batch("evaluate", model, selected),
        ATTACK_SETTINGS["linkage_batch_size"],
        device,
        seed,
    )


def select_best(channels):
    """Select by per-seed AUC and retain the same attacker's interpolated TPR."""
    if set(channels) != set(CHANNELS):
        raise ValueError("Best-of selection requires all five independently fitted channels.")
    name = max(CHANNELS, key=lambda candidate: channels[candidate]["auc"])
    return name, dict(channels[name])
