"""Single-sequence SafeGait-ER explanation pipeline."""

import argparse
import json
import sys

from opengait.data.collate_fn import sample_frames
from opengait.data.dataset import read_sequence
from opengait.modeling.models import SafeGaitER
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.utils import (
    get_device,
    load_checkpoint,
    load_config,
    seed_everything,
    write_json,
)


def release_sequence(input_path, config, saved=None, device="cpu"):
    """Compute evidence, protection, and the permitted release fields."""
    seed_everything(config["seed"])
    execution_device = get_device(device)
    sequence = read_sequence(input_path, resolution=tuple(config["resolution"]))
    sequence = sample_frames(sequence, count=config["frames"], random=False)
    sequence = sequence.unsqueeze(0).to(execution_device)

    model = SafeGaitER(config).to(execution_device).eval()
    if saved is not None:
        if "model" not in saved:
            raise ValueError("The checkpoint must contain a model state dictionary.")
        try:
            model.load_state_dict(saved["model"], strict=True)
        except RuntimeError as error:
            raise ValueError(
                "The checkpoint does not match the configured model architecture."
            ) from error

    evidence = model(sequence)
    protected = PrivateExplanation(config).to(execution_device)(evidence)
    reference = config.get("diagnostic_reference", [0.70, 0.70, 0.70])
    packet = ReleasePolicy(config).packets(protected, evidence["risk"], reference)[0]
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafeGait-ER method pipeline")
    parser.add_argument(
        "--input", required=True, help="Silhouette array, pickle, or image directory"
    )
    parser.add_argument("--config", help="YAML configuration; otherwise use checkpoint settings")
    parser.add_argument("--checkpoint", help="Compatible model checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", help="Optional JSON file containing the release packet")
    args = parser.parse_args(argv)

    saved = load_checkpoint(args.checkpoint, "cpu") if args.checkpoint else None
    if args.config:
        config = load_config(args.config)
    elif saved is not None:
        if "config" not in saved:
            raise ValueError("The checkpoint must contain its model configuration.")
        config = saved["config"]
    else:
        config = load_config("configs/safegait_er/casia_b.yaml")

    if saved is None:
        print("No checkpoint supplied: model parameters are untrained.", file=sys.stderr)
    packet = release_sequence(args.input, config, saved=saved, device=args.device)
    if args.output:
        write_json(args.output, packet)
    print(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False))
    return packet


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
