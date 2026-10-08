from opengait.data.dataset import SequenceRecord
from opengait.data.protocols import (
    matched_membership_indices,
    stress_membership_indices,
    linkage_identity_pools,
)
from opengait.utils import get_device, resolve, load_checkpoint, write_json
from attacks.common import check_identity_disjoint
from attacks.membership_attack import run_membership
from attacks.native_field_attack import fit_native_membership
from attacks.linkage_attack import make_pairs, fit_linkage
from attacks.identity_probe import run_identity_probe
from attacks.repeated_release import run_repeated_membership


def evaluate_privacy(config, archive=None, inputs=None, protocol="matched", repeated=False):
    if archive is None:
        archive = load_checkpoint(inputs or resolve(config["output"]) / "attack_inputs.pt")
    if archive["config"]["dataset"] != config["dataset"]:
        raise ValueError("The attack configuration and target model use different datasets.")
    device = get_device(config["device"])
    records = [
        SequenceRecord(**{k: row[k] for k in ("path", "subject", "condition", "view", "index")})
        for row in archive["records"]
    ]
    if protocol == "matched":
        if config["dataset"] != "CASIA-B":
            raise ValueError("Use the stress membership protocol for OU-MVLP.")
        split = matched_membership_indices(records, config["seed"])
    else:
        split = stress_membership_indices(records, config["dataset"], config["seed"])
    if not split["fit"] or not split["test"]:
        raise ValueError("Membership evaluation requires fitting and evaluation samples.")
    check_identity_disjoint(records, [i for i, y in split["fit"]], [i for i, y in split["test"]])
    settings = config["attacks"]
    first = [group[0] for group in archive["releases"]]
    raw = archive["raw_packets"]
    scores = archive["scores"]
    report = {
        "protocol": protocol,
        "seed": config["seed"],
        "fitting_instances": len(split["fit"]),
        "testing_instances": len(split["test"]),
        "unique_testing_members": len({i for i, y in split["test"] if y == 1}),
    }
    report["descriptor_safe"] = run_membership(
        first, scores, split, settings, device, config["seed"]
    )
    report["descriptor_raw"] = run_membership(raw, scores, split, settings, device, config["seed"])
    native = {}
    for interface, packets in (("safe", first), ("raw", raw)):
        for fusion in (False, True):
            native[interface + ("_fusion" if fusion else "_packet")] = fit_native_membership(
                packets, scores, split, settings, device, config["seed"], fusion
            )
    report["native"] = native
    report["best_of_safe"] = {
        metric: max(
            report["descriptor_safe"]["best_of"][metric],
            native["safe_packet"][metric],
            native["safe_fusion"][metric],
        )
        for metric in ("auc", "tpr_at_1pct_fpr")
    }
    report["best_of_raw"] = {
        metric: max(
            report["descriptor_raw"]["best_of"][metric],
            native["raw_packet"][metric],
            native["raw_fusion"][metric],
        )
        for metric in ("auc", "tpr_at_1pct_fpr")
    }
    fit_ids, test_ids = linkage_identity_pools(records, config["dataset"])
    check_identity_disjoint(records, fit_ids, test_ids)
    fit_pairs = make_pairs(
        records, fit_ids, settings["linkage_fit_pairs_per_class"], config["seed"]
    )
    test_pairs = make_pairs(
        records, test_ids, settings["linkage_test_pairs_per_class"], config["seed"] + 1
    )
    report["linkage"] = {}
    for interface, packets in (("safe", first), ("raw", raw)):
        for native_flag in (False, True):
            name = interface + ("_native" if native_flag else "_descriptor")
            report["linkage"][name] = fit_linkage(
                packets, fit_pairs, test_pairs, settings, device, config["seed"], native_flag
            )
    report["identity_probe"] = {
        interface: run_identity_probe(
            records, packets, settings, device, config["seed"], config["dataset"]
        )
        for interface, packets in (("safe", first), ("raw", raw))
    }
    if repeated:
        report["repeated_membership"] = run_repeated_membership(
            archive["releases"], scores, split, settings, device, config["seed"]
        )
        report["repeated_linkage"] = {}
        for budget in settings["repeated_budgets"]:
            groups = [group[:budget] for group in archive["releases"]]
            if any(len(g) != budget for g in groups):
                raise ValueError("Insufficient releases. Increase recognition --queries.")
            report["repeated_linkage"][str(budget)] = {
                family: fit_linkage(
                    groups,
                    fit_pairs,
                    test_pairs,
                    settings,
                    device,
                    config["seed"] + budget,
                    native_flag,
                )
                for family, native_flag in (("descriptor", False), ("native", True))
            }
    output = resolve(config["output"])
    write_json(output / ("privacy_" + protocol + ".json"), report)
    print(
        "Privacy best-of AUC: %.4f; results: %s" % (report["best_of_safe"]["auc"], output),
        flush=True,
    )
    return report


if __name__ == "__main__":
    from run import main

    main(["privacy"] + __import__("sys").argv[1:])
