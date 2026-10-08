"""Index licensed, preprocessed CASIA-B sequence folders into a portable manifest."""

import argparse
import csv
import os
from pathlib import Path

from opengait.data.protocols import ALL_CONDITIONS, VIEWS, read_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default="outputs/casia_b/manifest.csv")
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    destination = Path(args.manifest).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for identity in range(1, 125):
        subject = f"{identity:03d}"
        for condition in ALL_CONDITIONS:
            for view in VIEWS:
                folder = root / subject / condition / view
                if not folder.is_dir():
                    raise FileNotFoundError(f"Missing CASIA-B sequence folder: {folder}")
                arrays = sorted(
                    p for p in folder.iterdir() if p.suffix.lower() in (".npy", ".npz", ".pkl")
                )
                images = sorted(
                    p
                    for p in folder.iterdir()
                    if p.suffix.lower() in (".png", ".bmp", ".jpg", ".jpeg")
                )
                if len(arrays) == 1 and not images:
                    source = arrays[0]
                elif len(arrays) == 0 and images:
                    source = folder
                else:
                    raise ValueError(f"Expected one sequence array OR image frames at {folder}")
                try:
                    sequence_path = os.path.relpath(source, destination.parent)
                except ValueError:
                    sequence_path = str(source)
                records.append((subject, condition, view, sequence_path))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject", "condition", "view", "path"))
        writer.writerows(records)
    read_manifest(destination, complete=True)
    print(f"Indexed {len(records)} CASIA-B sequences at {destination}")


if __name__ == "__main__":
    main()
