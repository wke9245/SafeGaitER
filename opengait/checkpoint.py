"""Architecture validation and frozen-model loading for trained checkpoints."""

import copy
import hashlib
from collections.abc import Mapping

import torch

from opengait.modeling.models import SafeGaitER
from opengait.utils import get_device, load_checkpoint, resolve


def sha256_file(path):
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_metadata(saved):
    if not isinstance(saved, Mapping) or not {"model", "config"} <= saved.keys():
        raise ValueError("Checkpoint requires model weights and their configuration.")
    config = saved["config"]
    if not isinstance(config, Mapping) or config.get("dataset") != "CASIA-B":
        raise ValueError("Checkpoint must contain a CASIA-B configuration.")
    training = saved.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("Invalid checkpoint training metadata.")
    epochs = training.get("completed_epochs", saved.get("epoch"))
    seed = training.get("seed", config.get("seed"))
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Checkpoint must record a positive number of completed training epochs.")
    if type(seed) is not int or seed != config.get("seed"):
        raise ValueError("Checkpoint training seed must match its configuration.")
    return {"seed": seed, "completed_epochs": epochs}


def validate_architecture(config):
    model = config.get("model", {})
    if (
        config.get("frames") != 30
        or config.get("resolution") != [64, 44]
        or model.get("channels") != 128
        or model.get("strips") != 8
        or model.get("embedding_dim") != 256
        or model.get("num_classes") != 74
        or model.get("backbone") not in ("hierarchical", "gaitpart")
        or model.get("spatial_groups") != [[0, 1, 2], [3, 4], [5, 6, 7]]
        or model.get("temporal_boundaries") != [0, 8, 15, 23, 30]
    ):
        raise ValueError("Checkpoint configuration does not match the CASIA-B model dimensions.")


def frozen_model(saved, device="cuda", seed=None, config=None):
    training = training_metadata(saved)
    execution_device = get_device(device)
    if seed is not None and seed != training["seed"]:
        raise ValueError("Inference seed must equal the recorded model training seed.")
    recorded = copy.deepcopy(saved["config"])
    validate_architecture(recorded)
    if config is not None and config != recorded:
        raise ValueError("Explicit configuration must equal the checkpoint configuration.")
    recorded["device"] = str(execution_device)
    state = saved["model"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint model state must be a nonempty tensor dictionary.")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid model parameter: {name}")
    model = SafeGaitER(recorded).to(execution_device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint weights do not match the model architecture.") from error
    model.eval().requires_grad_(False)
    return model, recorded, training


def load_frozen_model(path, device="cuda", seed=None):
    return frozen_model(load_checkpoint(path, device), device=device, seed=seed)
