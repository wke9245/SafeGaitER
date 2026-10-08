from collections import defaultdict
import numpy as np

MEMBER_CONDITIONS = {"nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"}
NONMEMBER_CONDITIONS = {"nm-05", "nm-06", "bg-02", "cl-02"}
GALLERY_CONDITIONS = {"nm-01", "nm-02", "nm-03", "nm-04"}


def role(record, dataset="CASIA-B"):
    if dataset == "OU-MVLP":
        return (
            "train" if record.index < 5153 else ("gallery" if record.condition == "01" else "probe")
        )
    if 1 <= int(record.subject) <= 74:
        return "train" if record.condition in MEMBER_CONDITIONS else "nonmember"
    return "gallery" if record.condition in GALLERY_CONDITIONS else "probe"


def auxiliary_pool(records, target_dataset):
    if target_dataset == "CASIA-B":
        return [
            r
            for r in records
            if r.index <= 406
            and r.condition in {"00", "01"}
            and int(r.view) in {0, 15, 30, 45, 60, 90}
        ]
    return [r for r in records if 1 <= int(r.subject) <= 74 and r.condition in MEMBER_CONDITIONS]


def matched_membership_indices(records, seed=2025):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, r in enumerate(records):
        if 1 <= int(r.subject) <= 74:
            groups[r.subject].append(i)
    result = {"fit": [], "test": []}
    for subject, indices in sorted(groups.items()):
        nm = [i for i in indices if records[i].condition in GALLERY_CONDITIONS]
        non_nm = [i for i in indices if records[i].condition in {"nm-05", "nm-06"}]
        chosen = []
        for view in sorted({records[i].view for i in nm}):
            candidates = [i for i in nm if records[i].view == view]
            count = sum(records[i].view == view for i in non_nm)
            if count > len(candidates):
                raise ValueError("Member and nonmember NM view counts cannot be matched.")
            chosen.extend(rng.choice(candidates, count, replace=False).tolist())
        chosen += [i for i in indices if records[i].condition in {"bg-01", "cl-01"}]
        negatives = [i for i in indices if records[i].condition in NONMEMBER_CONDITIONS]
        if len(chosen) != len(negatives):
            raise ValueError("Identity-matched membership classes have unequal counts.")
        phase = "fit" if int(subject) <= 49 else "test"
        result[phase].extend((i, 1) for i in chosen)
        result[phase].extend((i, 0) for i in negatives)
    return result


def stress_membership_indices(records, dataset="CASIA-B", seed=2025, count=None):
    rng = np.random.default_rng(seed)
    result = {}
    for phase in ("fit", "test"):
        if dataset == "CASIA-B":
            low, high = (1, 49) if phase == "fit" else (50, 74)
            nlow, nhigh = (75, 99) if phase == "fit" else (100, 124)
            members = [
                i
                for i, r in enumerate(records)
                if low <= int(r.subject) <= high and r.condition in MEMBER_CONDITIONS
            ]
            nonmembers = [i for i, r in enumerate(records) if nlow <= int(r.subject) <= nhigh]
            number, replace = count or 2750, phase == "test"
        else:
            low, high = (0, 2575) if phase == "fit" else (2576, 5152)
            nlow, nhigh = (5153, 7729) if phase == "fit" else (7730, 10306)
            members = [i for i, r in enumerate(records) if low <= r.index <= high]
            nonmembers = [i for i, r in enumerate(records) if nlow <= r.index <= nhigh]
            number, replace = count or 10000, False
        if not members or not nonmembers:
            raise ValueError(
                "The identity-disjoint protocol requires fitting and evaluation identities."
            )
        if (not replace and len(members) < number) or len(nonmembers) < number:
            raise ValueError("Insufficient sequences for the identity-disjoint protocol.")
        result[phase] = [(int(i), 1) for i in rng.choice(members, number, replace=replace)]
        result[phase] += [(int(i), 0) for i in rng.choice(nonmembers, number, replace=False)]
    return result


def linkage_identity_pools(records, dataset="CASIA-B"):
    if dataset == "CASIA-B":
        fit = [i for i, r in enumerate(records) if 75 <= int(r.subject) <= 99]
        test = [i for i, r in enumerate(records) if 100 <= int(r.subject) <= 124]
    else:
        fit = [i for i, r in enumerate(records) if 5153 <= r.index <= 7729]
        test = [i for i, r in enumerate(records) if 7730 <= r.index <= 10306]
    return fit, test
