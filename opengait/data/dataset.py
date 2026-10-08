"""Silhouette sequence reading and spatial normalization."""

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def normalize_frames(array, resolution=(64, 44)):
    a = np.asarray(array)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 4 and a.shape[-1] in (1, 3, 4):
        a = a.mean(-1)
    if a.ndim != 3 or len(a) == 0:
        raise ValueError("A silhouette sequence must be a nonempty [frames, height, width] array.")
    if tuple(a.shape[1:]) == tuple(resolution):
        return torch.from_numpy((a > (0.5 if a.max() <= 1 else 127)).astype(np.float32))
    output = []
    height, width = resolution
    for frame in a:
        binary = frame > (0.5 if frame.max() <= 1 else 127)
        rows = np.flatnonzero(binary.any(1))
        if not len(rows):
            output.append(np.zeros((height, width), np.float32))
            continue
        crop = binary[rows[0] : rows[-1] + 1]
        target_width = max(1, round(crop.shape[1] * height / crop.shape[0]))
        resized = (
            np.asarray(
                Image.fromarray(crop.astype(np.uint8) * 255).resize(
                    (target_width, height), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        mass = resized.sum(0)
        center = int(np.searchsorted(np.cumsum(mass), mass.sum() / 2))
        left = center - width // 2
        out = np.zeros((height, width), np.float32)
        lo, hi = max(0, left), min(target_width, left + width)
        if hi > lo:
            out[:, lo - left : hi - left] = resized[:, lo:hi]
        output.append(out)
    return torch.from_numpy(np.stack(output))


def read_sequence(path, resolution=(64, 44)):
    p = Path(path)
    if p.is_dir():
        files = sorted(
            q for q in p.iterdir() if q.suffix.lower() in (".png", ".bmp", ".jpg", ".jpeg")
        )
        array = np.stack([np.asarray(Image.open(q).convert("L")) for q in files])
    elif p.suffix.lower() == ".pkl":
        with p.open("rb") as f:
            array = pickle.load(f)
    elif p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as data:
            if "silhouettes" in data:
                array = data["silhouettes"]
            elif len(data.files) == 1:
                array = data[data.files[0]]
            else:
                raise ValueError("NumPy archives with multiple arrays require a silhouettes key.")
    else:
        array = np.load(p, allow_pickle=False)
    return normalize_frames(array, resolution)
