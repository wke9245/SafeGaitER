"""Validate dataset roles before paper-protocol optimization."""

from collections import Counter

from .protocols import MEMBER_CONDITIONS


def validate_training_pool(records, dataset, classes):
    actual_classes = len({record.subject for record in records})
    expected_classes = 74 if dataset == "CASIA-B" else 5153
    if actual_classes != expected_classes or classes != expected_classes:
        raise ValueError(
            f"{dataset} training requires {expected_classes} identities; "
            f"found {actual_classes} and configured {classes}."
        )
    if dataset == "CASIA-B":
        expected = {
            (f"{identity:03d}", condition, f"{view:03d}")
            for identity in range(1, 75)
            for condition in MEMBER_CONDITIONS
            for view in range(0, 181, 18)
        }
        actual = {(record.subject, record.condition, record.view) for record in records}
        if actual != expected or len(records) != len(expected):
            raise ValueError(
                "CASIA-B training requires the complete 4,884-sequence member pool "
                "(NM 01-04, BG 01 and CL 01 across 11 views)."
            )


def validate_auxiliary_pool(records):
    if len(records) != 4884:
        raise ValueError(
            f"The configured auxiliary pool requires 4,884 sequences; found {len(records)}."
        )
    if len({record.key for record in records}) != len(records):
        raise ValueError("The auxiliary pool contains duplicate sequence keys.")


def validate_unique_records(records):
    keys = Counter(record.key for record in records)
    if any(count > 1 for count in keys.values()):
        raise ValueError("The dataset contains duplicate sequence keys.")
