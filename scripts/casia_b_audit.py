"""CASIA-B independent recognition-error triage and fitted auditor export."""

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from opengait.data.protocols import audit_partitions, read_manifest
from opengait.evaluation.audit import INPUT_DIMENSIONS, fit_auditors
from opengait.evaluation.observations import read_observations
from opengait.utils import write_json


def main():
    parser = argparse.ArgumentParser(description="Independent CASIA-B recognition-error audit")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), default=2025)
    parser.add_argument("--output", default="outputs/casia_b/triage.json")
    parser.add_argument(
        "--model-dir", help="Directory for fitted scalers, coefficients, and predictions"
    )
    args = parser.parse_args()
    destination = Path(args.output)
    model_dir = Path(args.model_dir) if args.model_dir else destination.parent / "audit_models"
    if destination.exists() or model_dir.exists():
        raise ValueError("Choose new paths for the audit report and model directory.")
    records = read_manifest(args.manifest)
    observations, metadata = read_observations(args.observations, records, args.seed)
    partitions = audit_partitions(records)
    fit_entries = [observations[r.key] for r in partitions["fit"]]
    test_entries = [observations[r.key] for r in partitions["evaluate"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        report, models, predictions = fit_auditors(fit_entries, test_entries, args.seed)
    model_dir.mkdir(parents=True)
    for name, model in models.items():
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        np.savez(
            model_dir / f"{name}.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            coefficients=classifier.coef_,
            intercept=classifier.intercept_,
            classes=classifier.classes_,
        )
    with (model_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence_key",
                "subject",
                "predicted_subject",
                "recognition_error",
                "release_level",
                *INPUT_DIMENSIONS,
            ]
        )
        for index, entry in enumerate(test_entries):
            writer.writerow(
                [
                    entry["key"],
                    entry["subject"],
                    entry["predicted_subject"],
                    int(entry["subject"] != entry["predicted_subject"]),
                    entry["safe_packet"]["level"],
                    *(float(predictions[name][index]) for name in INPUT_DIMENSIONS),
                ]
            )
    settings = {
        "classifier": "L2 logistic regression",
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 5000,
        "tol": 1e-4,
        "preprocessing": "StandardScaler fitted on audit-fit probes only",
        "input_dimensions": INPUT_DIMENSIONS,
        "seed": args.seed,
    }
    write_json(model_dir / "settings.json", settings)
    report.update(seed=args.seed, protocol="recognition_error_triage", artifacts=metadata)
    write_json(destination, report)
    print(f"Saved recognition-error triage results to {destination}")


if __name__ == "__main__":
    main()
