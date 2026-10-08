from collections import defaultdict
import numpy as np
from .common import mlp, fit_numeric, predict_numeric
from .descriptors import statistical_descriptor
from opengait.utils import seed_everything


def run_identity_probe(records, packets, settings, device="cpu", seed=2025, dataset="CASIA-B"):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        if (dataset == "CASIA-B" and 75 <= int(record.subject) <= 124) or (
            dataset == "OU-MVLP" and record.index >= 5153
        ):
            groups[record.subject].append(i)
    rng = np.random.default_rng(seed)
    fit, test, yfit, ytest = [], [], [], []
    for label, subject in enumerate(sorted(groups)):
        indices = rng.permutation(groups[subject])
        cut = len(indices) // 2
        if not cut or cut == len(indices):
            raise ValueError("Identity probing requires at least two sequences per identity.")
        fit.extend(indices[:cut])
        test.extend(indices[cut:])
        yfit.extend([label] * cut)
        ytest.extend([label] * (len(indices) - cut))
    x = np.stack([statistical_descriptor(p) for p in packets])
    seed_everything(seed)
    model = mlp([36, 128, 64, 32, len(groups)])
    fit_numeric(model, x[fit], yfit, settings, device, seed, multiclass=True)
    predicted = predict_numeric(model, x[test], device).argmax(1).numpy()
    return {
        "accuracy": float((predicted == np.asarray(ytest)).mean()),
        "classes": len(groups),
        "fit_count": len(fit),
        "test_count": len(test),
    }
