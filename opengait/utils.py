import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = cfg.pop("extends", None)
    return merge(load_config(path.parent / base), cfg) if base else cfg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name="cuda"):
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Choose a CUDA device such as cuda or cuda:0.") from error
    if device.type != "cuda":
        raise ValueError("SafeGait-ER requires an NVIDIA CUDA GPU.")
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable. Install the GPU dependencies and a compatible NVIDIA driver."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("The requested CUDA device is not visible.")
    return device


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_checkpoint(path, device="cuda", weights_only=True):
    return torch.load(resolve(path), map_location=get_device(device), weights_only=weights_only)
