import numpy as np
import torch
import torch.nn.functional as F


def binary_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Attack labels and scores must be one-dimensional arrays of equal length.")
    if not np.isfinite(scores).all() or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Attack scores must be finite and labels must be binary.")
    labels = labels.astype(int)
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    if not positive or not negative:
        raise ValueError("Attack evaluation requires both membership classes.")
    order = np.argsort(-scores, kind="stable")
    ordered_labels, ordered_scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(scores) - 1]
    tpr = np.r_[0.0, np.cumsum(ordered_labels)[ends] / positive]
    fpr = np.r_[0.0, np.cumsum(1 - ordered_labels)[ends] / negative]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(integrate(tpr, fpr))
    low_fpr_tpr = float(tpr[fpr <= 0.01].max())
    fnr = 1 - tpr
    cross = np.flatnonzero(fpr >= fnr)
    j = int(cross[0]) if len(cross) else len(fpr) - 1
    if j == 0:
        eer = float((fpr[j] + fnr[j]) / 2)
    else:
        d0, d1 = fpr[j - 1] - fnr[j - 1], fpr[j] - fnr[j]
        weight = -d0 / (d1 - d0) if d1 != d0 else 0
        eer = float(fpr[j - 1] + weight * (fpr[j] - fpr[j - 1]))
    return {"auc": auc, "tpr_at_1pct_fpr": low_fpr_tpr, "eer": eer, "count": len(labels)}


def reconstruction_metrics(predicted, target):
    # Local-window SSIM on all frames, rather than SSIM on selected display frames.
    x, y = predicted.flatten(0, 1)[:, None], target.flatten(0, 1)[:, None]
    mean_x, mean_y = F.avg_pool2d(x, 11, 1), F.avg_pool2d(y, 11, 1)
    var_x = F.avg_pool2d(x * x, 11, 1) - mean_x.square()
    var_y = F.avg_pool2d(y * y, 11, 1) - mean_y.square()
    cov = F.avg_pool2d(x * y, 11, 1) - mean_x * mean_y
    ssim = (
        (2 * mean_x * mean_y + 0.01**2)
        * (2 * cov + 0.03**2)
        / ((mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    ).mean()
    a, b = predicted >= 0.5, target >= 0.5
    intersection = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union)).mean()
    return {"ssim": float(ssim), "iou": float(iou)}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "runs": len(values),
    }
