"""Aggregate three seed-specific CASIA-B recognition or membership reports."""

import argparse
import json
from pathlib import Path

import numpy as np

from opengait.utils import write_json


def aggregate(readings):
    first = readings[0]
    if isinstance(first, dict):
        if any(not isinstance(row, dict) or set(row) != set(first) for row in readings):
            raise ValueError("Seed reports must have identical metric schemas.")
        return {key: aggregate([row[key] for row in readings]) for key in sorted(first)}
    if all(type(value) in (float, int) for value in readings):
        vector = np.asarray(readings, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("Seed reports contain a nonfinite metric.")
        return {"mean": float(vector.mean()), "sample_std": float(vector.std(ddof=1))}
    if any(value != first for value in readings):
        raise ValueError("Seed reports contain incompatible protocol metadata.")
    return first


def main():
    parser = argparse.ArgumentParser(description="Three-seed CASIA-B statistics")
    parser.add_argument("--inputs", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    seeds = [report.get("seed") for report in reports]
    if set(seeds) != {2024, 2025, 2026}:
        raise ValueError("Provide one result each for seeds 2024, 2025 and 2026.")
    ordered = sorted(reports, key=lambda row: row["seed"])
    if any(report.get("artifacts", {}).get("seed") != report["seed"] for report in ordered):
        raise ValueError("Seed report metadata does not match its inference artifacts.")
    if len({row["artifacts"]["manifest_key_sha256"] for row in ordered}) != 1:
        raise ValueError("All seed reports must use the same manifest sequences.")
    metrics = [{k: v for k, v in row.items() if k not in ("seed", "artifacts")} for row in ordered]
    result = {
        "seeds": [row["seed"] for row in ordered],
        "runs": 3,
        "metrics": aggregate(metrics),
        "artifacts": [row["artifacts"] for row in ordered],
    }
    write_json(args.output, result)
    print(f"Saved three-seed summary to {args.output}")


if __name__ == "__main__":
    main()
