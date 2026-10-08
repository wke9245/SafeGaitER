"""Exact CASIA-B sequence roles and privacy-attack partitions.

The manifest has the columns: subject, condition, view, path.
One record represents one complete silhouette sequence, not one frame.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIEWS = tuple(f"{v:03d}" for v in range(0, 181, 18))
TRAIN_CONDITIONS = ("nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01")
RESERVED_CONDITIONS = ("nm-05", "nm-06", "bg-02", "cl-02")
GALLERY_CONDITIONS = TRAIN_CONDITIONS[:4]
PROBE_CONDITIONS = ("nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02")
ALL_CONDITIONS = tuple(sorted(set(TRAIN_CONDITIONS + RESERVED_CONDITIONS)))


@dataclass(frozen=True)
class SequenceRecord:
    subject: str
    condition: str
    view: str
    path: str

    @property
    def key(self):
        return f"{self.subject}/{self.condition}/{self.view}"


def read_manifest(filename, complete=True):
    manifest = Path(filename).resolve()
    rows = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"subject", "condition", "view", "path"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires subject,condition,view,path columns.")
        for raw in reader:
            subject = f"{int(raw['subject']):03d}"
            view = f"{int(raw['view']):03d}"
            condition = raw["condition"].strip().lower().replace("_", "-")
            if not raw["path"].strip():
                raise ValueError("Manifest sequence path cannot be empty.")
            data_path = Path(raw["path"])
            if not data_path.is_absolute():
                data_path = (manifest.parent / data_path).resolve()
            if not (1 <= int(subject) <= 124 and view in VIEWS and condition in ALL_CONDITIONS):
                raise ValueError(f"Unexpected CASIA-B sequence: {subject}/{condition}/{view}")
            rows.append(SequenceRecord(subject, condition, view, str(data_path)))
    rows.sort(key=lambda record: record.key)
    keys = [r.key for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicated CASIA-B sequence identity/condition/view key.")
    if complete:
        if len(rows) != 124 * 11 * 10:
            raise ValueError(f"Expected 13640 CASIA-B sequences, found {len(rows)}.")
        expected = {
            f"{person:03d}/{cond}/{view}"
            for person in range(1, 125)
            for cond in ALL_CONDITIONS
            for view in VIEWS
        }
        if set(keys) != expected:
            raise ValueError("CASIA-B manifest is missing or mislabeling one or more sequences.")
    return rows


def role(record):
    subject = int(record.subject)
    if subject <= 74:
        return "train" if record.condition in TRAIN_CONDITIONS else "reserved_nonmember"
    return "gallery" if record.condition in GALLERY_CONDITIONS else "probe"


def grouped_roles(records):
    groups = {key: [] for key in ("train", "reserved_nonmember", "gallery", "probe")}
    for record in records:
        groups[role(record)].append(record)
    return groups


def matched_membership(records, seed=2025):
    """2,156 fitting + 1,100 evaluation instances per membership class."""
    lookup = {r.key: r for r in records}
    rng = np.random.default_rng(seed)
    result = {"fit": [], "evaluate": []}
    for subject_number in range(1, 75):
        subject = f"{subject_number:03d}"
        phase = "fit" if subject_number <= 49 else "evaluate"
        for view in VIEWS:
            nm_choices = [
                lookup[f"{subject}/{condition}/{view}"] for condition in GALLERY_CONDITIONS
            ]
            selected = rng.choice(len(nm_choices), size=2, replace=False)
            member = [nm_choices[int(i)] for i in selected]
            member += [lookup[f"{subject}/bg-01/{view}"], lookup[f"{subject}/cl-01/{view}"]]
            nonmember = [
                lookup[f"{subject}/{condition}/{view}"] for condition in RESERVED_CONDITIONS
            ]
            result[phase].extend((r, 1) for r in member)
            result[phase].extend((r, 0) for r in nonmember)
    expected = {"fit": 2156, "evaluate": 1100}
    for phase in result:
        labels = [label for _, label in result[phase]]
        if labels.count(0) != expected[phase] or labels.count(1) != expected[phase]:
            raise ValueError(f"Incorrect identity-matched counts for {phase}.")
    return result


def identity_disjoint_stress(records, seed=2025):
    """2,750 instances per class per phase; evaluation members use replacement."""
    rng = np.random.default_rng(seed)
    result = {}
    for phase, member_range, nonmember_range in (
        ("fit", range(1, 50), range(75, 100)),
        ("evaluate", range(50, 75), range(100, 125)),
    ):
        members = [
            r for r in records if int(r.subject) in member_range and r.condition in TRAIN_CONDITIONS
        ]
        nonmembers = [r for r in records if int(r.subject) in nonmember_range]
        if len(nonmembers) != 2750:
            raise ValueError("The stress nonmember pool requires 2,750 unique sequences.")
        if len(members) != (3234 if phase == "fit" else 1650):
            raise ValueError("Incorrect stress member source pool size.")
        selected_member = rng.choice(len(members), 2750, replace=(phase == "evaluate"))
        selected_nonmember = rng.choice(len(nonmembers), 2750, replace=False)
        result[phase] = [(members[int(i)], 1) for i in selected_member]
        result[phase] += [(nonmembers[int(i)], 0) for i in selected_nonmember]
    return result
