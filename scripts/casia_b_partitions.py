"""Export the paper-defined CASIA-B sequence and attack partitions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from opengait.evaluation.observations import manifest_digest
from opengait.data.protocols import (
    audit_partitions,
    grouped_roles,
    identity_disjoint_stress,
    matched_membership,
    linkage_pairs,
    read_manifest,
)


def write_partition(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "sequence_key", "subject", "membership"])
        for index, (record, label) in enumerate(rows):
            writer.writerow([index, record.key, record.subject, label])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="outputs/casia_b/partitions")
    parser.add_argument("--include-linkage", action="store_true", help="Export linkage pair CSVs")
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=(2024, 2025, 2026), default=[2024, 2025, 2026]
    )
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    roles = grouped_roles(records)
    counts = {key: len(value) for key, value in roles.items()}
    summary = {
        "dataset_roles": counts,
        "manifest_key_sha256": manifest_digest(records),
        "partitions": {},
    }
    if counts != {"train": 4884, "reserved_nonmember": 3256, "gallery": 2200, "probe": 3300}:
        raise ValueError(f"Unexpected CASIA-B recognition partition counts: {counts}")
    (out / "dataset_roles.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Partition seeds must be distinct.")
    for name, entries in roles.items():
        write_partition(
            out / f"recognizer_{name}.csv", [(r, int(name == "train")) for r in entries]
        )
    for phase, entries in audit_partitions(records).items():
        with (out / f"audit_{phase}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sequence_key", "subject", "condition", "view"])
            writer.writerows((r.key, r.subject, r.condition, r.view) for r in entries)
    for seed in args.seeds:
        if args.include_linkage:
            for phase, pairs in linkage_pairs(records, seed).items():
                filename = out / f"linkage_seed{seed}_{phase}.csv"
                with filename.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sample_index", "first_key", "second_key", "same_identity"])
                    writer.writerows((i, *pair) for i, pair in enumerate(pairs))
                summary["partitions"][filename.name] = {
                    "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                    "pairs": len(pairs),
                }
        for protocol, split in (
            ("identity_matched", matched_membership(records, seed)),
            ("identity_disjoint_stress", identity_disjoint_stress(records, seed)),
        ):
            for phase, rows in split.items():
                filename = out / f"{protocol}_seed{seed}_{phase}.csv"
                write_partition(filename, rows)
                summary["partitions"][filename.name] = {
                    "sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                    "instances": len(rows),
                    "unique_sequence_keys": len({r.key for r, _ in rows}),
                }
                counts_here = {label: sum(x[1] == label for x in rows) for label in (0, 1)}
                print(f"{protocol} seed={seed} {phase}: {counts_here}")
    (out / "partition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CASIA-B partition files saved to", out)


if __name__ == "__main__":
    main()
