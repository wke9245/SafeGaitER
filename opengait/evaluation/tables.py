"""Manuscript table exports from full-precision, independently computed seed reports."""

import csv
from pathlib import Path

import numpy as np


def format_stat(values, decimals, signed=False):
    if any(value is None for value in values):
        return ""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Table statistics require three finite seed-level values.")
    mean, std = float(values.mean()), float(values.std(ddof=1))
    mean_text = format(mean, f"+.{decimals}f" if signed else f".{decimals}f")
    return mean_text + " +/- " + format(std, f".{decimals}f")


def write_tables(reports, output):
    if [row.get("seed") for row in reports] != [2024, 2025, 2026]:
        raise ValueError("Table reports must be ordered by seeds 2024, 2025, and 2026.")
    protocols = {row.get("protocol") for row in reports}
    if len(protocols) != 1:
        raise ValueError("Table exports require a single evaluation protocol.")
    protocol = next(iter(protocols))
    if protocol not in {"recognition_error_triage", "identity_matched_repeated_release"}:
        raise ValueError("Table export supports independent triage and repeated-release reports.")
    destination = Path(output)
    names = (
        ["table11.csv", "tableD11.csv"]
        if protocol == "recognition_error_triage"
        else ["table12.csv"]
    )
    if any((destination / name).exists() for name in names):
        raise ValueError("Choose a new table directory or remove existing table files explicitly.")
    destination.mkdir(parents=True, exist_ok=True)
    if protocol == "recognition_error_triage":
        with (destination / "table11.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["audit_input", "AP", "AUROC", "Recall_at_10pct_pct", "paired_delta_AP_vs_score"]
            )
            for name in reports[0]["audit"]:
                rows = [report["audit"][name] for report in reports]
                writer.writerow(
                    [
                        name,
                        format_stat([row["ap"] for row in rows], 3),
                        format_stat([row["auroc"] for row in rows], 3),
                        format_stat([row["recall_at_10pct_pct"] for row in rows], 1),
                        format_stat(
                            [row["paired_delta_ap_vs_score"] for row in rows], 3, signed=True
                        ),
                    ]
                )
        with (destination / "tableD11.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "release_level",
                    "test_probes_seed2024_2025_2026",
                    "errors_seed2024_2025_2026",
                    "score_only_AP",
                    "score_plus_safe_AP",
                ]
            )
            for level in ("low", "medium", "high"):
                rows = [report["within_release_level"][level] for report in reports]
                writer.writerow(
                    [
                        level,
                        "/".join(str(row["probes"]) for row in rows),
                        "/".join(str(row["errors"]) for row in rows),
                        format_stat([row["score_only_ap"] for row in rows], 3),
                        format_stat([row["score_plus_safe_ap"] for row in rows], 3),
                    ]
                )
    else:
        budgets = set(reports[0]["budgets"])
        if any(set(report["budgets"]) != budgets for report in reports):
            raise ValueError(
                "Repeated-release tables require identical query budgets across seeds."
            )
        with (destination / "table12.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["query_budget", "best_of_MIA_AUC", "TPR_at_1pct_FPR_pct", "native_linkage_AUC"]
            )
            for budget in sorted(budgets, key=int):
                rows = [report["budgets"][budget] for report in reports]
                writer.writerow(
                    [
                        budget,
                        format_stat([row["best_of"]["auc"] for row in rows], 2),
                        format_stat([row["best_of"]["tpr_at_1pct_fpr_pct"] for row in rows], 1),
                        format_stat([row["native_linkage"]["auc"] for row in rows], 2),
                    ]
                )
