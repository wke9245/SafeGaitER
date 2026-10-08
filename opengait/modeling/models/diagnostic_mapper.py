import torch

CATEGORIES = (
    "normal_walking",
    "carried_object",
    "clothing_variation",
    "local_occlusion",
    "lower_body_occlusion",
    "consecutive_frame_occlusion",
    "silhouette_jitter",
    "frame_dropping",
    "combined_disturbance",
)


def diagnostic_categories(state, reference=None):
    # No subject ID, condition label, or disturbance ground truth enters this mapper.
    rho = state["rho"].detach()
    p = state["p"].detach()
    interval = state["interval_reliability"].detach()
    psi = state["psi"].detach()
    if reference is None:
        reference = torch.full_like(rho, 0.70)
    else:
        reference = torch.as_tensor(reference, dtype=rho.dtype, device=rho.device).expand_as(rho)
    delta = reference - rho
    other_mean = (rho.sum(1, keepdim=True) - rho) / 2
    carried = ((rho - other_mean >= 0.20) & (delta >= 0.15)).any(1)
    clothing = (((delta >= 0.10) & (delta <= 0.25)).sum(1) >= 2) & (
        rho.max(1).values - rho.min(1).values < 0.20
    )
    lower = (rho[:, 2] < 0.30) & (rho[:, :2] >= 0.50).all(1)
    local = ((rho < 0.30).sum(1) == 1) & ((rho >= 0.50).sum(1) == 2) & ~lower
    other_interval = (interval.sum(1, keepdim=True) - interval) / 3
    consecutive = (other_interval - interval >= 0.25).any(1)
    # Dominant degradation includes an isolated relative reliability decrease.
    dominant = (delta.max(1).values - delta.min(1).values >= 0.20) | local | lower | carried
    jitter = ~dominant & (psi < 0.60)
    adjacent = torch.full_like(p, -1)
    adjacent[:, :-1] = p[:, 1:] - p[:, :-1]
    backward = torch.full_like(p, -1)
    backward[:, 1:] = p[:, :-1] - p[:, 1:]
    dropping = ((p < 0.10) & (torch.maximum(adjacent, backward) >= 0.15)).any(1)
    flags = torch.stack((carried, clothing, local, lower, consecutive, jitter, dropping), 1)
    result = torch.zeros(rho.shape[0], dtype=torch.long, device=rho.device)
    for column in range(7):
        result = torch.where(flags[:, column], column + 1, result)
    return torch.where(flags.sum(1) >= 2, 8, result)
