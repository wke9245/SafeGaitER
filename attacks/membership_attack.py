import numpy as np
from .common import mlp, fit_numeric, predict_numeric
from .descriptors import statistical_descriptor
from opengait.evaluation.metrics import binary_metrics
from opengait.utils import seed_everything


def run_membership(packets, scores, split, settings, device="cpu", seed=2025):
    descriptors = np.stack([statistical_descriptor(p) for p in packets])
    result = {}
    for channel in ("score_only", "packet_only", "score_plus_packet"):
        if channel == "score_only":
            x = np.asarray(scores, np.float32)[:, None]
        elif channel == "packet_only":
            x = descriptors
        else:
            x = np.c_[descriptors, np.asarray(scores, np.float32)]
        fit, yfit = zip(*split["fit"])
        test, ytest = zip(*split["test"])
        seed_everything(seed)
        model = mlp([x.shape[1], 128, 64, 32, 1])
        fit_numeric(model, x[list(fit)], yfit, settings, device, seed)
        logits = predict_numeric(model, x[list(test)], device).squeeze(1).numpy()
        result[channel] = binary_metrics(ytest, logits)
    result["best_of"] = {
        metric: max(result[c][metric] for c in ("score_only", "packet_only", "score_plus_packet"))
        for metric in ("auc", "tpr_at_1pct_fpr")
    }
    return result
