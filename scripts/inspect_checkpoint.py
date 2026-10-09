"""Inspect saved model and optimizer states on a CUDA device."""

import argparse
import copy
import json
import sys
from collections.abc import Mapping

import torch

from opengait.checkpoint import frozen_model, sha256_file, training_metadata, validate_architecture
from opengait.data.collate_fn import sample_frames
from opengait.data.dataset import read_sequence
from opengait.utils import get_device, resolve, write_json


def finite_state(state, name):
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Missing tensor state: {name}")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid tensor: {name}/{key}")


def inspect_optimizer(state, parameters, total_steps):
    if not isinstance(state, Mapping) or not {"state", "param_groups"} <= state.keys():
        raise ValueError("Invalid optimizer state dictionary.")
    identifiers = [identifier for group in state["param_groups"] for identifier in group["params"]]
    if len(identifiers) != len(parameters) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Optimizer parameter identifiers do not match the network.")
    lookup = dict(zip(identifiers, parameters))
    counters = []
    for identifier, values in state["state"].items():
        if identifier not in lookup:
            raise ValueError("Optimizer state references an unknown parameter.")
        step = values.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not torch.isfinite(step).all()
        ):
            raise ValueError("Invalid optimizer update counter.")
        count = float(step)
        if count < 0 or count != int(count) or (total_steps is not None and count > total_steps):
            raise ValueError("Optimizer counter disagrees with the recorded training steps.")
        for field in ("exp_avg", "exp_avg_sq"):
            value = values.get(field)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != lookup[identifier].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid optimizer moment: {field}")
        counters.append(count)
    return {
        "parameter_states": len(counters),
        "minimum_step": min(counters) if counters else None,
        "maximum_step": max(counters) if counters else None,
        "shape_and_finiteness": "passed",
    }


def inspect(saved, device, input_path=None):
    if not isinstance(saved, Mapping) or not {"model", "config", "training"} <= saved.keys():
        raise ValueError("Checkpoint requires model, config, and training entries.")
    config = copy.deepcopy(saved["config"])
    recorded_training = training_metadata(saved)
    training = saved["training"]
    if not isinstance(training, Mapping):
        raise ValueError("Invalid training metadata.")
    epochs = recorded_training["completed_epochs"]
    steps = training.get("optimization_steps", saved.get("iteration"))
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("Invalid optimization step count.")
    if "iteration" in saved and saved["iteration"] != steps:
        raise ValueError("Iteration and optimization step counts disagree.")
    if "epoch" in saved and saved["epoch"] != epochs:
        raise ValueError("Epoch and completed epoch counts disagree.")
    if training.get("seed") != config.get("seed"):
        raise ValueError("The training seed must match the saved configuration.")
    config["device"] = str(device)
    finite_state(saved["model"], "model")
    model_tensors = {name: value.to(device) for name, value in saved["model"].items()}
    try:
        validate_architecture(config)
    except ValueError:
        architecture_binding = "requires the originating network definition and method settings"
    else:
        model, _, _ = frozen_model(saved, device=str(device))
        architecture_binding = "strict model loading passed"
    report = {
        "training": dict(training),
        "recorded_architecture": config["model"],
        "architecture_binding": architecture_binding,
        "tensor_loading": "passed",
        "model_tensor_count": len(saved["model"]),
        "model_parameter_values": sum(value.numel() for value in saved["model"].values()),
        "tensor_device": str(next(iter(model_tensors.values())).device),
        "model_finiteness": "passed",
    }
    if "optimizer" in saved:
        report["optimizer"] = inspect_optimizer(
            saved["optimizer"], list(model_tensors.values()), steps
        )
    auxiliary = saved.get("auxiliary_models")
    if auxiliary is not None:
        if not isinstance(auxiliary, Mapping) or set(auxiliary) != {"routing", "auditors"}:
            raise ValueError("Expected routing and auditor auxiliary model states.")
        auxiliary_parameters = []
        for name in ("routing", "auditors"):
            finite_state(auxiliary[name], name)
            auxiliary_parameters.extend(value.to(device) for value in auxiliary[name].values())
            report[name] = {"tensor_loading": "passed", "finite_tensors": "passed"}
        if "auxiliary_optimizer" in saved:
            report["auxiliary_optimizer"] = inspect_optimizer(
                saved["auxiliary_optimizer"],
                auxiliary_parameters,
                steps,
            )
    if input_path:
        if architecture_binding != "strict model loading passed":
            raise ValueError(
                "Sequence inference requires the originating network definition and method settings."
            )
        sequence = read_sequence(resolve(input_path), resolution=tuple(config["resolution"]))
        inputs = sample_frames(sequence, config["frames"], random=False).unsqueeze(0).to(device)
        out = model(inputs)
        expected = {
            "embedding": [1, 256],
            "logits": [1, 74],
            "contribution": [1, 30, 8],
            "stability": [1, 30, 8],
            "local_risk": [1, 30, 8],
            "risk": [1],
        }
        for name, shape in expected.items():
            if list(out[name].shape) != shape or not torch.isfinite(out[name]).all():
                raise ValueError(f"Invalid forward output: {name}")
        report["forward_pass"] = {
            "input_shape": list(inputs.shape),
            "output_shapes": {name: list(out[name].shape) for name in expected},
            "finite_outputs": "passed",
            "recognition_accuracy_evaluated": False,
        }
    else:
        report["forward_pass"] = "not requested; provide --input to inspect sequence outputs"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a SafeGait-ER checkpoint on a CUDA GPU")
    parser.add_argument(
        "--checkpoint", default="checkpoints/safegait_er_casia_b_seed2025.pt"
    )
    parser.add_argument("--device", default="cuda", help="CUDA device such as cuda or cuda:0")
    parser.add_argument("--input", help="Optional silhouette sequence for forward inspection")
    parser.add_argument("--output", default="outputs/checkpoint_inspection.json")
    args = parser.parse_args(argv)
    device = get_device(args.device)
    path = resolve(args.checkpoint)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    report = inspect(saved, device, args.input)
    report.update(
        checkpoint=path.name,
        checkpoint_bytes=path.stat().st_size,
        checkpoint_sha256=sha256_file(path),
        runtime={
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "device": str(device),
        },
    )
    write_json(resolve(args.output), report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
