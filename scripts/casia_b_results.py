"""CASIA-B subject-level recognition and explanation release statistics."""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from opengait.data.protocols import grouped_roles, read_manifest
from opengait.evaluation.observations import read_observations
from opengait.evaluation.retrieval import subject_match
from opengait.utils import get_device, write_json


def recognition(records, outputs, device="cuda"):
    device = get_device(device)
    groups = grouped_roles(records)
    gallery, probes = groups["gallery"], groups["probe"]
    if len(gallery) != 2200 or len(probes) != 3300:
        raise ValueError("CASIA-B recognition requires 2,200 gallery and 3,300 probe sequences.")
    subjects = [record.subject for record in gallery]
    if any(subjects.count(subject) != 44 for subject in set(subjects)):
        raise ValueError("Every held-out identity must have 44 NM gallery sequences.")
    embeddings = torch.tensor(
        [outputs[r.key]["embedding"] for r in gallery], dtype=torch.float32, device=device
    )
    successes = defaultdict(list)
    for start in range(0, len(probes), 512):
        block = probes[start : start + 512]
        queries = torch.tensor(
            [outputs[r.key]["embedding"] for r in block], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            predicted = subject_match(queries, embeddings, subjects)["predicted_subjects"]
        for record, subject in zip(block, predicted):
            successes[record.condition[:2].upper()].append(float(record.subject == subject))
    result = {
        label + "_rank1_pct": 100.0 * float(np.mean(successes[label]))
        for label in ("NM", "BG", "CL")
    }
    result["average_rank1_pct"] = float(
        np.mean([result[label + "_rank1_pct"] for label in ("NM", "BG", "CL")])
    )
    result["probe_sequences"] = len(probes)
    result["gallery_sequences_per_subject"] = 44
    return result


def disclosure(records, outputs):
    roles = grouped_roles(records)
    heldout = roles["gallery"] + roles["probe"]
    if len(heldout) != 5500:
        raise ValueError("CASIA-B release reporting requires 5,500 held-out sequences.")
    frequency = Counter(outputs[r.key]["safe_packet"]["level"] for r in heldout)
    return {
        "total": len(heldout),
        "counts": {level: frequency[level] for level in ("low", "medium", "high")},
        "fractions": {
            level: frequency[level] / len(heldout) for level in ("low", "medium", "high")
        },
        "region_field_coverage": (frequency["low"] + frequency["medium"]) / len(heldout),
        "temporal_field_coverage": (frequency["low"] + frequency["medium"]) / len(heldout),
        "diagnostic_field_coverage": (frequency["medium"] + frequency["high"]) / len(heldout),
        "mean_risk": float(np.mean([outputs[r.key]["risk"] for r in heldout])),
    }


def main():
    parser = argparse.ArgumentParser(description="CASIA-B recognition and release statistics")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/casia_b/principal_results.json")
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    outputs, metadata = read_observations(args.observations, records)
    result = {
        "seed": metadata["seed"],
        "artifacts": metadata,
        "recognition": recognition(records, outputs, args.device),
        "release": disclosure(records, outputs),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
