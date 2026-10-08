import numpy as np
from opengait.modeling.models.diagnostic_mapper import CATEGORIES
from opengait.modeling.models.release_policy import validate_packet

LEVELS = ("low", "medium", "high")


def onehot(index, width):
    result = np.zeros(width, dtype=np.float32)
    result[index] = 1
    return result


def entropy(x):
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    p = np.asarray(x) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def raw_summary(packet, groups=((0, 1, 2), (3, 4), (5, 6, 7)), bounds=(0, 8, 15, 23, 30)):
    omega, alpha = np.asarray(packet["omega"]), np.asarray(packet["alpha"])
    s = np.array([omega[:, g].mean() for g in groups], dtype=np.float32)
    p = np.array([alpha[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])], dtype=np.float32)
    p = p / p.sum() if p.sum() > 0 else np.full(4, 0.25, np.float32)
    return s, p


def statistical_descriptor(packet):
    z = np.zeros(36, dtype=np.float32)
    level = packet["level"]
    if level not in ("raw", "coarse"):
        validate_packet(packet)
        z[:3] = onehot(LEVELS.index(level), 3)
        z[31] = float(level == "high")
    if level in ("low", "raw", "coarse"):
        if level == "raw":
            s, p = raw_summary(packet)
        else:
            s, p = np.asarray(packet["spatial_scores"]), np.asarray(packet["temporal_scores"])
        z[3:6], z[6:10] = s, p
        z[26:28] = 1
        z[32:] = [entropy(s), float(np.max(s)), entropy(p), float(np.max(p))]
    if level in ("medium", "raw", "coarse"):
        if level in ("raw", "coarse"):
            s, p = (
                raw_summary(packet)
                if level == "raw"
                else (np.asarray(packet["spatial_scores"]), np.asarray(packet["temporal_scores"]))
            )
            region, interval = int(np.argmax(s)), int(np.argmax(p))
        else:
            region, interval = packet["dominant_region"], packet["dominant_interval"]
        z[10:13], z[13:17] = onehot(region, 3), onehot(interval, 4)
        z[28:30] = 1
    if level in ("medium", "high"):
        z[17:26] = onehot(CATEGORIES.index(packet["diagnostic"]), 9)
        z[30] = 1
    return z


def field_vector(packet):
    if packet["level"] == "coarse":
        return np.r_[
            packet["spatial_scores"],
            packet["temporal_scores"],
            np.asarray(packet["coarse_contribution"]).flatten(),
        ].astype(np.float32)
    validate_packet(packet)
    level = packet["level"]
    qr = onehot(LEVELS.index(level), 3)
    if level == "low":
        return np.r_[packet["spatial_scores"], packet["temporal_scores"], qr].astype(np.float32)
    d = onehot(CATEGORIES.index(packet["diagnostic"]), 9)
    if level == "medium":
        return np.r_[
            onehot(packet["dominant_region"], 3), onehot(packet["dominant_interval"], 4), qr, d
        ].astype(np.float32)
    return np.r_[qr, d].astype(np.float32)


def raw_vector(packet):
    return np.concatenate(
        [
            np.asarray(packet[k], dtype=np.float32).reshape(-1)
            for k in ("omega", "alpha", "contribution", "keyframes")
        ]
    )


def inversion_vector(packet):
    if packet["level"] == "raw":
        return raw_vector(packet)
    if packet["level"] == "coarse":
        return field_vector(packet)
    qr = onehot(LEVELS.index(packet["level"]), 3)
    payload = np.zeros(16, dtype=np.float32)
    if packet["level"] == "low":
        payload[:7] = np.r_[packet["spatial_scores"], packet["temporal_scores"]]
    else:
        payload[7:] = onehot(CATEGORIES.index(packet["diagnostic"]), 9)
        if packet["level"] == "medium":
            payload[:7] = np.r_[
                onehot(packet["dominant_region"], 3), onehot(packet["dominant_interval"], 4)
            ]
    return np.r_[qr, payload]
