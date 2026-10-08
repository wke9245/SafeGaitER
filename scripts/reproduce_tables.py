import csv
import json
from collections import defaultdict
from pathlib import Path
from opengait.evaluation.metrics import mean_std
from opengait.utils import write_json


def flatten(obj, prefix=""):
    for key, value in obj.items():
        name = prefix + "." + key if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, value


def aggregate(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    grouped, per_run = defaultdict(dict), []
    for filename in (
        "recognition.json",
        "privacy_matched.json",
        "privacy_stress.json",
        "inversion_raw.json",
        "inversion_safe.json",
        "audit.json",
        "profile.json",
    ):
        for path in sorted(root.rglob(filename)):
            obj = json.loads(path.read_text(encoding="utf-8"))
            relative = path.parent.relative_to(root)
            # Seed directories of the same experiment are combined; other runs remain separate.
            parts = [p for p in relative.parts if not p.startswith("seed_")]
            method = "/".join(parts) or "current"
            seed = obj.get("seed")
            config_file = path.parent / "config.json"
            if seed is None and config_file.is_file():
                seed = json.loads(config_file.read_text(encoding="utf-8")).get("seed")
            for metric, value in flatten(obj):
                key = (method, filename.removesuffix(".json"), metric)
                grouped[key][str(seed) if seed is not None else str(path)] = value
                per_run.append(
                    {
                        "method": method,
                        "source": filename,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                    }
                )
    rows = []
    for (method, source, metric), runs in sorted(grouped.items()):
        rows.append(
            dict(method=method, source=source, metric=metric, **mean_std(list(runs.values())))
        )
    with (destination / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "source", "metric", "mean", "std", "runs"])
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "per_run.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "source", "seed", "metric", "value"])
        writer.writeheader()
        writer.writerows(per_run)
    write_json(destination / "summary.json", rows)
    print("Tables exported: " + str(destination), flush=True)
    return rows


if __name__ == "__main__":
    from run import main

    main(["tables"] + __import__("sys").argv[1:])
