"""Independent post-hoc membership inference from frozen CASIA-B observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from attacks.descriptors import statistical_descriptor
from opengait.data.protocols import identity_disjoint_stress, matched_membership, read_manifest
from opengait.evaluation.observations import read_observations
from opengait.utils import get_device, seed_everything, write_json

CHANNELS = {
    "score_only": ("score",),
    "raw_only": ("raw",),
    "score_plus_raw": ("raw", "score"),
    "safe_only": ("safe",),
    "score_plus_safe": ("safe", "score"),
}


def validate_scores(y_true, values):
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(values, dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or not np.isfinite(scores).all()
        or set(labels.tolist()) != {0, 1}
    ):
        raise ValueError("Attack metrics require finite scores and both binary membership classes.")
    return labels, scores


def auc(y_true, values):
    """ROC AUC with midranks for tied scores."""
    labels, scores = validate_scores(y_true, values)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def tpr_at_fpr(y_true, values, max_fpr=0.01):
    """Highest empirical true-positive rate attainable without exceeding the FPR bound."""
    labels, scores = validate_scores(y_true, values)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    order = np.argsort(-scores, kind="mergesort")
    true_positive = false_positive = 0
    best = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        group = labels[order[start:stop]]
        true_positive += int(group.sum())
        false_positive += int(len(group) - group.sum())
        if false_positive / negatives <= max_fpr + 1e-12:
            best = max(best, true_positive / positives)
        start = stop
    return float(best)


def fit_attack(x_train, y_train, x_eval, seed=2025, device="cuda"):
    device = get_device(device)
    seed_everything(seed)
    if (
        x_train.ndim != 2
        or x_eval.ndim != 2
        or x_train.shape[1] != x_eval.shape[1]
        or y_train.shape != (len(x_train),)
        or not np.isfinite(x_train).all()
        or not np.isfinite(x_eval).all()
        or set(y_train.tolist()) != {0, 1}
    ):
        raise ValueError("Invalid post-hoc attacker inputs.")
    net = nn.Sequential(
        nn.Linear(x_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    inputs = torch.from_numpy(x_train.astype(np.float32)).to(device)
    labels = torch.from_numpy(y_train.astype(np.float32)).to(device)
    for _ in range(50):
        net.train()
        for indices in torch.randperm(len(inputs), device=device).split(256):
            prediction = net(inputs[indices]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(prediction, labels[indices])
            if not torch.isfinite(loss):
                raise ValueError("Post-hoc attacker loss became nonfinite.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    net.eval().requires_grad_(False)
    with torch.no_grad():
        prediction = net(torch.from_numpy(x_eval.astype(np.float32)).to(device)).squeeze(1)
    return prediction.cpu().numpy(), net


def run_attack(
    records,
    observations,
    seed=2025,
    device="cuda",
    protocol="identity_matched",
    checkpoint_dir=None,
    artifacts=None,
):
    device = get_device(device)
    if protocol == "identity_matched":
        split = matched_membership(records, seed)
    elif protocol == "identity_disjoint_stress":
        split = identity_disjoint_stress(records, seed)
    else:
        raise ValueError("Unknown membership protocol.")
    if {r.key for r, _ in split["fit"]} & {r.key for r, _ in split["evaluate"]}:
        raise ValueError("Attacker fitting and evaluation sequences overlap.")
    selected_keys = {r.key for rows in split.values() for r, _ in rows}
    features = {}
    for key in sorted(selected_keys):
        entry = observations[key]
        features[key] = {
            "score": np.asarray([entry["score"]], dtype=np.float32),
            "raw": statistical_descriptor(entry["raw_packet"]),
            "safe": statistical_descriptor(entry["safe_packet"]),
        }
    output = {}
    for name, fields in CHANNELS.items():
        arrays = {}
        for phase in ("fit", "evaluate"):
            items = split[phase]
            arrays[phase] = (
                np.stack(
                    [np.concatenate([features[r.key][field] for field in fields]) for r, _ in items]
                ),
                np.asarray([label for _, label in items], dtype=np.int64),
            )
        fitted_x, fitted_y = arrays["fit"]
        evaluated_x, evaluated_y = arrays["evaluate"]
        estimates, attacker = fit_attack(fitted_x, fitted_y, evaluated_x, seed, device)
        output[name] = {
            "auc": auc(evaluated_y, estimates),
            "tpr_at_1pct_fpr_pct": 100.0 * tpr_at_fpr(evaluated_y, estimates),
        }
        if checkpoint_dir is not None:
            destination = Path(checkpoint_dir) / f"{protocol}_seed{seed}_{name}.pt"
            if destination.exists():
                raise ValueError(
                    "Attacker checkpoint already exists; choose a new output directory."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": attacker.state_dict(),
                    "input_dimension": fitted_x.shape[1],
                    "hidden_dimensions": [128, 64, 32],
                    "channels": list(fields),
                    "seed": seed,
                    "completed_epochs": 50,
                    "protocol": protocol,
                    "metrics": output[name],
                    "artifacts": artifacts,
                },
                destination,
            )
        print(f"Fitted {protocol} seed={seed} {name}", flush=True)
    for prefix, variants in (
        ("raw_interface", ("score_only", "raw_only", "score_plus_raw")),
        ("safe_interface", ("score_only", "safe_only", "score_plus_safe")),
    ):
        output[prefix + "_best_of"] = {
            metric: max(output[variant][metric] for variant in variants)
            for metric in ("auc", "tpr_at_1pct_fpr_pct")
        }
    return {
        "seed": seed,
        "protocol": protocol,
        "artifacts": artifacts,
        "partition": {
            phase: {
                "instances_per_class": {
                    str(label): sum(y == label for _, y in rows) for label in (0, 1)
                },
                "unique_sequences": len({r.key for r, _ in rows}),
            }
            for phase, rows in split.items()
        },
        "attacks": output,
    }


def main():
    parser = argparse.ArgumentParser(description="CASIA-B independent membership inference")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), default=2025)
    parser.add_argument(
        "--protocol",
        choices=("identity_matched", "identity_disjoint_stress"),
        default="identity_matched",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/casia_b/membership.json")
    parser.add_argument("--checkpoint-dir", help="Destination for fitted attacker weights")
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    observations, metadata = read_observations(args.observations, records, seed=args.seed)
    checkpoint_dir = args.checkpoint_dir or Path(args.output).parent / "attack_checkpoints"
    report = run_attack(
        records, observations, args.seed, args.device, args.protocol, checkpoint_dir, metadata
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
