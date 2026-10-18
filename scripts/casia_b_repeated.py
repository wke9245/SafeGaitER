"""CASIA-B query-budget membership and native-field linkage evaluation."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from attacks.repeated_release import (
    ATTACK_SETTINGS,
    CHANNELS,
    QUERY_BUDGETS,
    ReleaseFeatures,
    fit_linkage,
    fit_membership,
    select_best,
)
from opengait.data.protocols import linkage_pairs, matched_membership, read_manifest
from opengait.evaluation.observations import read_metadata
from opengait.utils import get_device, write_json


def save_attacker(path, model, seed, budget, channel, metrics, artifacts):
    torch.save(
        {
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "seed": seed,
            "query_budget": budget,
            "channel": channel,
            "training": ATTACK_SETTINGS,
            "metrics": metrics,
            "artifacts": artifacts,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Repeated-release CASIA-B privacy evaluation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), default=2025)
    parser.add_argument(
        "--budgets", nargs="+", type=int, choices=QUERY_BUDGETS, default=list(QUERY_BUDGETS)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/casia_b/repeated.json")
    parser.add_argument(
        "--checkpoint-dir", help="Directory for fitted attackers and pair manifests"
    )
    args = parser.parse_args()
    destination = Path(args.output)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else destination.parent / "repeated_attackers"
    )
    if destination.exists() or checkpoint_dir.exists():
        raise ValueError("Choose new output paths for this repeated-release run.")
    device = get_device(args.device)
    records = read_manifest(args.manifest)
    metadata = read_metadata(args.observations, records, args.seed)
    features = ReleaseFeatures.read(args.observations, records, metadata, args.budgets)
    membership = matched_membership(records, args.seed)
    linkage = linkage_pairs(records, args.seed)
    checkpoint_dir.mkdir(parents=True)
    membership_hashes = {}
    for phase, rows in membership.items():
        path = checkpoint_dir / f"membership_{phase}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "sequence_key", "subject", "membership"])
            writer.writerows(
                (index, record.key, record.subject, label)
                for index, (record, label) in enumerate(rows)
            )
        membership_hashes[phase] = hashlib.sha256(path.read_bytes()).hexdigest()
    pair_hashes = {}
    for phase, pairs in linkage.items():
        path = checkpoint_dir / f"linkage_{phase}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "first_key", "second_key", "same_identity"])
            writer.writerows((index, *pair) for index, pair in enumerate(pairs))
        pair_hashes[phase] = hashlib.sha256(path.read_bytes()).hexdigest()
    results, selection = {}, {}
    for budget in sorted(args.budgets):
        channels = {}
        for name, (family, fusion) in CHANNELS.items():
            print(f"Fitting Q={budget} membership channel {name}", flush=True)
            model, metrics, logits = fit_membership(
                features,
                membership,
                budget,
                family,
                fusion,
                device,
                args.seed,
            )
            channels[name] = metrics
            save_attacker(
                checkpoint_dir / f"q{budget}_{name}.pt",
                model,
                args.seed,
                budget,
                name,
                metrics,
                metadata,
            )
            np.save(checkpoint_dir / f"q{budget}_{name}_test_logits.npy", logits)
            del model
        selected, best = select_best(channels)
        print(f"Fitting Q={budget} native-field linkage", flush=True)
        model, link_metrics, logits = fit_linkage(features, linkage, budget, device, args.seed)
        save_attacker(
            checkpoint_dir / f"q{budget}_native_linkage.pt",
            model,
            args.seed,
            budget,
            "native_linkage",
            link_metrics,
            metadata,
        )
        np.save(checkpoint_dir / f"q{budget}_native_linkage_test_logits.npy", logits)
        del model
        results[str(budget)] = {
            "membership": channels,
            "best_of": best,
            "native_linkage": link_metrics,
        }
        selection[str(budget)] = selected
    artifacts = dict(
        metadata,
        membership_partition_sha256=membership_hashes,
        linkage_pair_sha256=pair_hashes,
        auc_selected_channel=selection,
    )
    report = {
        "seed": args.seed,
        "protocol": "identity_matched_repeated_release",
        "artifacts": artifacts,
        "attack_settings": ATTACK_SETTINGS,
        "partition": {
            "membership_fit_instances_per_class": 2156,
            "membership_test_instances_per_class": 1100,
            "linkage_fit_pairs_per_class": 50000,
            "linkage_test_pairs_per_class": 20000,
        },
        "budgets": results,
    }
    write_json(destination, report)
    print(f"Saved repeated-release privacy results to {destination}")


if __name__ == "__main__":
    main()
