"""Export a compatible trained model checkpoint with source provenance."""

import argparse
from pathlib import Path

import torch

from opengait.checkpoint import frozen_model, sha256_file
from opengait.utils import get_device, load_checkpoint, resolve, write_json


def main():
    parser = argparse.ArgumentParser(description="Export a trained CASIA-B inference checkpoint")
    parser.add_argument("--source", required=True, help="Existing trained model checkpoint")
    parser.add_argument("--output", required=True, help="Destination .pt file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trusted-training-source",
        action="store_true",
        help="Load an author-controlled training archive with serialized RNG state",
    )
    args = parser.parse_args()
    source = resolve(args.source).resolve()
    destination = resolve(args.output).resolve()
    if source == destination or destination.exists():
        raise ValueError("Checkpoint destination must be a new file.")
    if destination.suffix != ".pt":
        raise ValueError("Checkpoint destination must use the .pt extension.")
    device = get_device(args.device)
    saved = load_checkpoint(source, device, weights_only=not args.trusted_training_source)
    model, config, training = frozen_model(saved, device=device)
    provenance = {"source_sha256": sha256_file(source)}
    checkpoint = {
        "format_version": 1,
        "model": {name: value.detach() for name, value in model.state_dict().items()},
        "config": config,
        "training": training,
        "provenance": provenance,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.partial")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_json(
        Path(str(destination) + ".json"),
        {
            "format_version": 1,
            "checkpoint_sha256": sha256_file(destination),
            "dataset": config["dataset"],
            "model": config["model"],
            "training": training,
            "provenance": provenance,
        },
    )
    print(f"Exported trained checkpoint to {destination}")


if __name__ == "__main__":
    main()
