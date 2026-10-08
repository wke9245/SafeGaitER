import numpy as np
from .descriptors import statistical_descriptor
from .common import mlp, fit_numeric, predict_numeric
from .native_field_attack import fit_native_membership
from opengait.evaluation.metrics import binary_metrics
from opengait.utils import seed_everything


def run_repeated_membership(releases, scores, split, settings, device="cpu", seed=2025):
    result = {}
    for budget in settings["repeated_budgets"]:
        groups = [group[:budget] for group in releases]
        if any(len(group) != budget for group in groups):
            raise ValueError("Insufficient repeated releases.")
        x = np.stack(
            [np.stack([statistical_descriptor(p) for p in group]).mean(0) for group in groups]
        )
        fit, yfit = zip(*split["fit"])
        test, ytest = zip(*split["test"])
        channels = {}
        for fusion in (False, True):
            values = np.c_[x, scores] if fusion else x
            seed_everything(seed + budget)
            model = mlp([values.shape[1], 128, 64, 32, 1])
            fit_numeric(model, values[list(fit)], yfit, settings, device, seed + budget)
            logits = predict_numeric(model, values[list(test)], device).squeeze(1).numpy()
            channels["descriptor_fusion" if fusion else "descriptor_packet"] = binary_metrics(
                ytest, logits
            )
            channels["native_fusion" if fusion else "native_packet"] = fit_native_membership(
                groups, scores, split, settings, device, seed + budget, fusion
            )
        seed_everything(seed + budget)
        score_values = np.asarray(scores, dtype=np.float32)[:, None]
        score_model = mlp([1, 128, 64, 32, 1])
        fit_numeric(score_model, score_values[list(fit)], yfit, settings, device, seed + budget)
        score_logits = (
            predict_numeric(score_model, score_values[list(test)], device).squeeze(1).numpy()
        )
        channels["score_only"] = binary_metrics(ytest, score_logits)
        channels["best_of"] = {
            metric: max(
                channels[name][metric]
                for name in (
                    "descriptor_fusion",
                    "descriptor_packet",
                    "native_fusion",
                    "native_packet",
                    "score_only",
                )
            )
            for metric in ("auc", "tpr_at_1pct_fpr")
        }
        result[str(budget)] = channels
    return result
