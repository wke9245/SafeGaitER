"""Frozen-model CASIA-B gallery retrieval and risk-conditioned explanation release."""

import argparse
import json
from pathlib import Path

import torch

from opengait.checkpoint import load_frozen_model, sha256_file
from opengait.data.collate_fn import sample_frames
from opengait.data.dataset import read_sequence
from opengait.data.protocols import GALLERY_CONDITIONS, grouped_roles, read_manifest
from opengait.evaluation.observations import manifest_digest, metadata_path
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.utils import get_device, seed_everything, write_json


def load_batch(record, config, device):
    sequence = read_sequence(record.path, resolution=tuple(config["resolution"]))
    frames = sample_frames(sequence, count=config["frames"], random=False)
    return frames.unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="CASIA-B frozen-model inference")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--releases",
        type=int,
        choices=range(1, 201),
        default=1,
        help="Independent stochastic releases per fixed source sequence",
    )
    parser.add_argument("--output", default="outputs/casia_b/observations.jsonl")
    args = parser.parse_args()
    device = get_device(args.device)
    model, config, training = load_frozen_model(args.checkpoint, device, seed=args.seed)
    seed_everything(args.seed)
    protector = PrivateExplanation(config).to(device).eval()
    release = ReleasePolicy(config)
    records = read_manifest(args.manifest)
    for record in records:
        if not Path(record.path).exists():
            raise FileNotFoundError(f"Missing silhouette sequence: {record.path}")
    roles = grouped_roles(records)
    gallery_pools = {}
    for group, entries in (
        (
            "train",
            [r for r in records if int(r.subject) <= 74 and r.condition in GALLERY_CONDITIONS],
        ),
        ("heldout", roles["gallery"]),
    ):
        vectors = []
        for record in entries:
            with torch.no_grad():
                embedding = model(load_batch(record, config, device), contribution=False)[
                    "embedding"
                ]
            vectors.append(embedding[0].detach())
        gallery_pools[group] = {
            "records": entries,
            "embeddings": torch.stack(vectors),
            "subjects": [r.subject for r in entries],
            "keys": [r.key for r in entries],
        }

    def extract(record):
        group = "train" if int(record.subject) <= 74 else "heldout"
        gallery = gallery_pools[group]
        out = model(
            load_batch(record, config, device),
            gallery=gallery["embeddings"],
            gallery_subjects=gallery["subjects"],
            query_keys=[record.key],
            gallery_keys=gallery["keys"],
        )
        with torch.no_grad():
            state = protector(out)
        return out, state, gallery

    # Condition labels select the reference pool; the diagnostic mapper reads protected values.
    gallery_rho = {}
    for pool in gallery_pools.values():
        for record in pool["records"]:
            _, state, _ = extract(record)
            gallery_rho[record.key] = state["rho"][0].detach()

    def reference_for(record, predicted, pool):
        values = [
            gallery_rho[other.key]
            for other in pool["records"]
            if other.subject == predicted and other.view == record.view and other.key != record.key
        ]
        if not values:
            raise ValueError("Same-view normal-walking diagnostic reference is missing.")
        return torch.stack(values).mean(0).unsqueeze(0)

    destination = Path(args.output)
    if destination.exists() or metadata_path(destination).exists():
        raise ValueError("Choose a new inference output path for this run.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination) + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(records, 1):
                out, state, pool = extract(record)
                reference = reference_for(record, out["predicted_subjects"][0], pool)
                packet = release.packets(state, out["risk"], reference)[0]
                result = {
                    "key": record.key,
                    "subject": record.subject,
                    "condition": record.condition,
                    "view": record.view,
                    "score": float(out["confidence"][0].detach()),
                    "risk": float(out["risk"][0].detach()),
                    "predicted_subject": out["predicted_subjects"][0],
                    "embedding": out["embedding"][0].detach().cpu().tolist(),
                    "raw_packet": {
                        "level": "raw",
                        "omega": out["omega"][0].detach().cpu().tolist(),
                        "alpha": out["alpha"][0].detach().cpu().tolist(),
                        "contribution": out["contribution"][0].detach().cpu().tolist(),
                        "keyframes": out["keyframes"][0].detach().cpu().tolist(),
                    },
                    "safe_packet": packet,
                }
                if args.releases > 1:
                    packets = [packet]
                    with torch.no_grad():
                        for _ in range(1, args.releases):
                            repeated_state = protector(out)
                            packets.append(
                                release.packets(repeated_state, out["risk"], reference)[0]
                            )
                    result["release_packets"] = packets
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                if index % 500 == 0:
                    print(f"Processed {index}/{len(records)} CASIA-B sequences", flush=True)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_json(
        metadata_path(destination),
        {
            "schema_version": 1,
            "dataset": "CASIA-B",
            "seed": args.seed,
            "training": training,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "manifest_key_sha256": manifest_digest(records),
            "observations_sha256": sha256_file(destination.resolve()),
            "sequences": len(records),
            "release_count": args.releases,
            "release_settings": config["release"],
            "device": str(device),
            "torch_version": str(torch.__version__),
        },
    )
    print(f"Saved CASIA-B observations to {destination}")


if __name__ == "__main__":
    main()
