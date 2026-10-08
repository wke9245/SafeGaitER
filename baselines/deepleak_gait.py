"""Independent gait adaptation of DeepLeak protection operations."""

import torch


def raw_packet(out, i):
    return {
        "level": "raw",
        "omega": out["omega"][i].detach().cpu().tolist(),
        "alpha": out["alpha"][i].detach().cpu().tolist(),
        "contribution": out["contribution"][i].detach().cpu().tolist(),
        "keyframes": out["keyframes"][i].detach().cpu().tolist(),
    }


def protect_raw(packet, strategy="deepleak", sigma=0.1, clip=0.7, mask_ratio=0.2, deepleak=None):
    if strategy not in {"raw", "clipping", "noise", "masking", "coarsening", "generic", "deepleak"}:
        raise ValueError("Unknown explanation protection strategy: " + str(strategy))
    settings = {
        "noise_level": 0.01,
        "clamp_range": [0.0, 1.0],
        "precision": 4,
        "mask_threshold": 0.001,
    }
    settings.update(deepleak or {})
    if strategy == "coarsening":
        omega = torch.tensor(packet["omega"])
        alpha = torch.tensor(packet["alpha"])
        contribution = torch.tensor(packet["contribution"])
        groups, bounds = ((0, 1, 2), (3, 4), (5, 6, 7)), (0, 8, 15, 23, 30)
        s = torch.stack([omega[:, g].mean() for g in groups])
        p = torch.stack([alpha[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
        p = p / p.sum().clamp_min(1e-8)
        c = torch.stack(
            [
                torch.stack([contribution[a:b][:, g].mean() for g in groups])
                for a, b in zip(bounds[:-1], bounds[1:])
            ]
        )
        return {
            "level": "coarse",
            "spatial_scores": s.tolist(),
            "temporal_scores": p.tolist(),
            "coarse_contribution": c.tolist(),
        }
    result = {"level": "raw", "keyframes": list(packet["keyframes"])}
    for field in ("omega", "alpha", "contribution"):
        value = torch.tensor(packet[field], dtype=torch.float32)
        if strategy == "deepleak":
            value = value + settings["noise_level"] * torch.randn_like(value)
            value = value.clamp(*settings["clamp_range"])
            scale = 10 ** settings["precision"]
            value = torch.round(value * scale) / scale
            value = value.masked_fill(value.abs() < settings["mask_threshold"], 0)
            result[field] = value.tolist()
            continue
        if strategy in ("clipping", "generic"):
            value = value.clamp(max=clip)
        if strategy in ("noise", "generic"):
            value = (value + sigma * torch.randn_like(value)).clamp(0, 1)
        if strategy in ("masking", "generic"):
            value = value * (torch.rand_like(value) >= mask_ratio)
        result[field] = value.tolist()
    return result
