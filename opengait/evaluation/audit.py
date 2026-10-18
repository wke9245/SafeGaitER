"""Independent recognition-error triage with audit-fit-only preprocessing."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from attacks.descriptors import LEVELS, onehot, statistical_descriptor

INPUT_DIMENSIONS = {
    "score_only": 1,
    "release_level_only": 3,
    "score_plus_level": 4,
    "score_plus_raw": 37,
    "score_plus_safe": 37,
}


def audit_inputs(entries):
    score = np.asarray([[entry["score"]] for entry in entries], dtype=np.float64)
    levels = np.stack([onehot(LEVELS.index(entry["safe_packet"]["level"]), 3) for entry in entries])
    raw = np.stack([statistical_descriptor(entry["raw_packet"]) for entry in entries])
    safe = np.stack([statistical_descriptor(entry["safe_packet"]) for entry in entries])
    return {
        "score_only": score,
        "release_level_only": levels,
        "score_plus_level": np.concatenate((score, levels), axis=1),
        "score_plus_raw": np.concatenate((raw, score), axis=1),
        "score_plus_safe": np.concatenate((safe, score), axis=1),
    }


def audit_metrics(labels, scores, seed):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Audit metrics require finite one-dimensional prediction scores.")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Audit metrics require correct and incorrect recognition decisions.")
    budget = int(np.ceil(0.1 * len(labels)))
    priority = np.random.default_rng(seed).permutation(len(labels))
    selected = np.lexsort((priority, -scores))[:budget]
    hits, errors = int(labels[selected].sum()), int(labels.sum())
    return {
        "ap": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "recall_at_10pct_pct": 100.0 * hits / errors,
        "review_budget": budget,
        "review_hits": hits,
    }


def fit_auditors(fit_entries, test_entries, seed):
    x_fit, x_test = audit_inputs(fit_entries), audit_inputs(test_entries)
    y_fit = np.asarray([int(e["predicted_subject"] != e["subject"]) for e in fit_entries])
    y_test = np.asarray([int(e["predicted_subject"] != e["subject"]) for e in test_entries])
    if set(y_fit.tolist()) != {0, 1} or set(y_test.tolist()) != {0, 1}:
        raise ValueError("Each audit partition requires both correct and incorrect decisions.")
    models, predictions, results = {}, {}, {}
    for name, width in INPUT_DIMENSIONS.items():
        if x_fit[name].shape != (1650, width) or x_test[name].shape != (1650, width):
            raise ValueError("Audit inputs must match the 1,650-probe partition dimensions.")
        if not np.isfinite(x_fit[name]).all() or not np.isfinite(x_test[name]).all():
            raise ValueError("Audit input features must be finite.")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                tol=1e-4,
                random_state=seed,
            ),
        )
        model.fit(x_fit[name], y_fit)
        scores = model.predict_proba(x_test[name])[:, 1]
        models[name], predictions[name] = model, scores
        results[name] = audit_metrics(y_test, scores, seed)
    baseline = results["score_only"]["ap"]
    for values in results.values():
        values["paired_delta_ap_vs_score"] = values["ap"] - baseline
    levels = [entry["safe_packet"]["level"] for entry in test_entries]
    within_level = {}
    for level in LEVELS:
        mask = np.asarray([value == level for value in levels])
        row = {"probes": int(mask.sum()), "errors": int(y_test[mask].sum())}
        for name in ("score_only", "score_plus_safe"):
            row[name + "_ap"] = (
                float(average_precision_score(y_test[mask], predictions[name][mask]))
                if y_test[mask].sum()
                else None
            )
        within_level[level] = row
    report = {
        "counts": {
            "fit_probes": len(y_fit),
            "test_probes": len(y_test),
            "fit_errors": int(y_fit.sum()),
            "test_errors": int(y_test.sum()),
            "test_error_prevalence": float(y_test.mean()),
        },
        "audit": results,
        "safe_minus_score_plus_level": {
            "paired_delta_ap": results["score_plus_safe"]["ap"] - results["score_plus_level"]["ap"],
            "paired_delta_recall_pct": results["score_plus_safe"]["recall_at_10pct_pct"]
            - results["score_plus_level"]["recall_at_10pct_pct"],
        },
        "within_release_level": within_level,
    }
    return report, models, predictions
