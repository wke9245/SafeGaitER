"""Single-sequence SafeGait-ER explanation release."""

import argparse
import json
import sys

import torch

from opengait.checkpoint import frozen_model
from opengait.data.collate_fn import sample_frames
from opengait.data.dataset import read_sequence
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.utils import get_device, load_checkpoint, load_config, seed_everything, write_json


def release_sequence(input_path, saved, device="cuda", config=None):
    model, settings, training = frozen_model(saved, device=device, config=config)
    seed_everything(training["seed"])
    execution_device = get_device(device)
    sequence = read_sequence(input_path, resolution=tuple(settings["resolution"]))
    sequence = sample_frames(sequence, count=settings["frames"], random=False)
    sequence = sequence.unsqueeze(0).to(execution_device)
    evidence = model(sequence)
    with torch.no_grad():
        protected = PrivateExplanation(settings).to(execution_device)(evidence)
    reference = settings.get("diagnostic_reference", [0.70, 0.70, 0.70])
    return ReleasePolicy(settings).packets(protected, evidence["risk"], reference)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafeGait-ER explanation release")
    parser.add_argument("--input", required=True, help="Silhouette file or image directory")
    parser.add_argument("--checkpoint", required=True, help="Compatible trained checkpoint")
    parser.add_argument("--config", help="Optional matching YAML configuration")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1")
    parser.add_argument("--output", help="Optional release packet JSON file")
    args = parser.parse_args(argv)
    device = get_device(args.device)
    saved = load_checkpoint(args.checkpoint, device)
    config = load_config(args.config) if args.config else None
    packet = release_sequence(args.input, saved, device=device, config=config)
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
