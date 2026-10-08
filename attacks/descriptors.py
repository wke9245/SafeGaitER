"""36-dimensional attack features built from observable explanation fields only."""

import numpy as np

from opengait.modeling.models.diagnostic_mapper import CATEGORIES
from opengait.modeling.models.release_policy import validate_packet

LEVELS = ("low", "medium", "high")


def onehot(index, width):
    output = np.zeros(width, dtype=np.float32)
    if isinstance(index, bool) or int(index) != index or not 0 <= int(index) < width:
        raise ValueError("Invalid categorical index.")
    output[int(index)] = 1.0
    return output


def entropy(values):
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    total = values.sum()
    if total == 0:
        return 0.0
    normalized = values / total
    nonzero = normalized[normalized > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def raw_summary(raw, groups=((0, 1, 2), (3, 4), (5, 6, 7)), bounds=(0, 8, 15, 23, 30)):
    omega = np.asarray(raw["omega"], dtype=np.float32)
    alpha = np.asarray(raw["alpha"], dtype=np.float32)
    if (
        omega.shape != (30, 8)
        or alpha.shape != (30,)
        or not np.isfinite(omega).all()
        or not np.isfinite(alpha).all()
        or (omega < 0).any()
        or (omega > 1).any()
        or (alpha < 0).any()
        or (alpha > 1).any()
    ):
        raise ValueError("Raw attribution dimensions must be (30,8) and (30,).")
    spatial = np.asarray([omega[:, group].mean() for group in groups], dtype=np.float32)
    temporal = np.asarray(
        [alpha[left:right].mean() for left, right in zip(bounds[:-1], bounds[1:])],
        dtype=np.float32,
    )
    temporal = temporal / temporal.sum() if temporal.sum() > 0 else np.full(4, 0.25)
    return spatial, temporal


def statistical_descriptor(packet):
    """Return 36 fields; absent release fields remain zero and have zero visibility flags."""
    values = np.zeros(36, dtype=np.float32)
    level = packet["level"]
    if level == "raw":
        spatial, temporal = raw_summary(packet)
        values[3:6], values[6:10] = spatial, temporal
        values[10:13] = onehot(int(np.argmax(spatial)), 3)
        values[13:17] = onehot(int(np.argmax(temporal)), 4)
        values[26:30] = 1.0
    else:
        validate_packet(packet)
        values[:3] = onehot(LEVELS.index(level), 3)
        values[31] = float(level == "high")
        if level == "low":
            spatial = np.asarray(packet["spatial_scores"], dtype=np.float32)
            temporal = np.asarray(packet["temporal_scores"], dtype=np.float32)
            values[3:6], values[6:10] = spatial, temporal
            values[26:28] = 1.0
        if level == "medium":
            values[10:13] = onehot(packet["dominant_region"], 3)
            values[13:17] = onehot(packet["dominant_interval"], 4)
            values[28:30] = 1.0
        if level in ("medium", "high"):
            values[17:26] = onehot(CATEGORIES.index(packet["diagnostic"]), 9)
            values[30] = 1.0
    if level in ("raw", "low"):
        values[32:] = [
            entropy(spatial),
            float(np.max(spatial)),
            entropy(temporal),
            float(np.max(temporal)),
        ]
    return values
