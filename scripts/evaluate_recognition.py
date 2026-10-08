import time
import torch
from torch.utils.data import DataLoader
from opengait.data.dataset import GaitDataset, index_dataset
from opengait.data.protocols import role
from opengait.modeling.models import SafeGaitER
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.evaluation.retrieval import recognition_report
from opengait.utils import resolve, get_device, load_checkpoint, seed_everything, write_json
from baselines.deepleak_gait import raw_packet, protect_raw


def build_galleries(model, records, config, device):
    pools = {}
    dataset = config["dataset"]
    for partition in ("train", "test"):
        if dataset == "CASIA-B":
            selected = [
                r
                for r in records
                if r.condition in {"nm-01", "nm-02", "nm-03", "nm-04"}
                and (
                    (int(r.subject) <= 74)
                    if partition == "train"
                    else (75 <= int(r.subject) <= 124)
                )
            ]
        else:
            selected = [
                r
                for r in records
                if r.condition == "01"
                and ((r.index < 5153) if partition == "train" else (r.index >= 5153))
            ]
        if not selected:
            raise ValueError("Missing " + partition + " gallery")
        features = []
        for batch in DataLoader(
            GaitDataset(selected),
            batch_size=config["evaluation"]["batch_size"],
            num_workers=config["workers"],
        ):
            with torch.no_grad():
                out = model(batch["x"].to(device), contribution=False)
            features.append(out["embedding"].detach())
        pools[partition] = {
            "embeddings": torch.cat(features),
            "subjects": [r.subject for r in selected],
            "keys": [r.key for r in selected],
            "records": selected,
        }
    return pools


def evaluate(config, checkpoint, records=None, queries=1, strategy=None):
    if queries < 1:
        raise ValueError("The number of releases must be positive.")
    device = get_device(config["device"])
    saved = load_checkpoint(checkpoint, device)
    if saved["config"]["dataset"] != config["dataset"]:
        raise ValueError("The checkpoint and evaluation configuration use different datasets.")
    strategy = strategy or saved["config"]["training"]["strategy"]
    config["training"]["strategy"] = strategy
    # Architecture comes from the checkpoint; release settings may be overridden.
    config["model"] = saved["config"]["model"]
    config["risk"] = saved["config"]["risk"]
    seed_everything(config["seed"])
    model = SafeGaitER(config).to(device).eval()
    model.load_state_dict(saved["model"])
    transform, policy = PrivateExplanation(config).to(device), ReleasePolicy(config)
    if records is None:
        records = index_dataset(
            resolve(config["data_root"]), config["dataset"], resolve(config["ou_split"])
        )
    galleries = build_galleries(model, records, config, device)
    # References are protected NM states of the predicted identity/view, never test labels.
    reference_by_key = {}
    for pool in galleries.values():
        selected = pool["records"]
        loader = DataLoader(
            GaitDataset(selected),
            batch_size=config["evaluation"]["batch_size"],
            num_workers=config["workers"],
        )
        for batch in loader:
            local = [selected[i] for i in batch["record_index"].tolist()]
            out = model(
                batch["x"].to(device),
                pool["embeddings"],
                pool["subjects"],
                [r.key for r in local],
                pool["keys"],
            )
            state = transform(out)
            for record, rho in zip(local, state["rho"].detach()):
                reference_by_key[record.key] = rho
    packets, raw, scores, embeddings, predictions = [], [], [], [], []
    references, risk_values = [], []
    profile_total, profile_count = 0.0, 0
    loader = DataLoader(
        GaitDataset(records),
        batch_size=config["evaluation"]["batch_size"],
        num_workers=config["workers"],
    )
    for batch in loader:
        local = [records[i] for i in batch["record_index"].tolist()]
        # Dataset order is identity sorted; a boundary batch may span two identity partitions.
        for partition in ("train", "test"):
            chosen = [
                i
                for i, r in enumerate(local)
                if ("train" if role(r, config["dataset"]) in ("train", "nonmember") else "test")
                == partition
            ]
            if not chosen:
                continue
            current = [local[i] for i in chosen]
            x = batch["x"][chosen].to(device)
            pool = galleries[partition]
            if device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            out = model(
                x, pool["embeddings"], pool["subjects"], [r.key for r in current], pool["keys"]
            )
            ref = []
            for r, subject in zip(current, out["predicted_subjects"]):
                nm = [
                    g
                    for g in pool["records"]
                    if g.subject == subject and g.view == r.view and g.key != r.key
                ]
                values = [reference_by_key[g.key] for g in nm]
                ref.append(
                    torch.stack(values).mean(0)
                    if values
                    else torch.tensor(config.get("diagnostic_reference", [0.70] * 3), device=device)
                )
            ref = torch.stack(ref)
            state = transform(out)
            first_packets = policy.packets(state, out["risk"], ref)
            if device.type == "cuda":
                torch.cuda.synchronize()
            profile_total += time.perf_counter() - begin
            profile_count += len(current)
            release_groups = [[p] for p in first_packets]
            for q in range(1, queries):
                next_packets = policy.packets(transform(out), out["risk"], ref)
                for group, packet in zip(release_groups, next_packets):
                    group.append(packet)
            for i, record in enumerate(current):
                rawp = raw_packet(out, i)
                if strategy and strategy not in ("safe", "no_auditor"):
                    release_groups[i] = [
                        rawp
                        if strategy == "raw"
                        else protect_raw(
                            rawp,
                            strategy,
                            config["release"]["sigma_max"],
                            deepleak=config.get("baselines", {}).get("deepleak"),
                        )
                        for _ in range(queries)
                    ]
                packets.append((record.key, release_groups[i]))
                raw.append((record.key, rawp))
                scores.append((record.key, float(out["confidence"][i].detach())))
                embeddings.append((record.key, out["embedding"][i].detach().cpu()))
                predictions.append((record.key, out["predicted_subjects"][i]))
                references.append((record.key, ref[i].cpu()))
                risk_values.append((record.key, float(out["risk"][i].detach())))

    # Restore source order explicitly, including boundary batches.
    def ordered(items):
        by_key = dict(items)
        return [by_key[r.key] for r in records]

    predictions = ordered(predictions)
    recognition = recognition_report(records, predictions, config["dataset"])
    first = [g[0] for g in ordered(packets)]
    proportions = {
        q: sum(p["level"] == q for p in first) / len(first)
        for q in ("low", "medium", "high", "raw", "coarse")
    }
    report = {
        "recognition": recognition,
        "release_proportions": proportions,
        "batch_processing_ms_per_sequence": profile_total / profile_count * 1000,
        "processing_includes_local_gradient": True,
        "queries": queries,
        "seed": config["seed"],
        "dataset": config["dataset"],
        "strategy": strategy,
        "release_settings": config["release"],
        "policy_source": config.get("policy_source", "fixed"),
    }
    archive = {
        "config": config,
        "records": [r.metadata() for r in records],
        "releases": ordered(packets),
        "raw_packets": ordered(raw),
        "scores": ordered(scores),
        "embeddings": torch.stack(ordered(embeddings)),
        "predictions": predictions,
        "internal_risk": ordered(risk_values),
        "references": ordered(references),
        "report": report,
    }
    output = resolve(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save(archive, output / "attack_inputs.pt")
    # This file contains only the allowed explanation fields.
    with (output / "released_packets.jsonl").open("w", encoding="utf-8") as f:
        import json

        for group in archive["releases"]:
            for packet in group:
                f.write(json.dumps(packet, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "recognition.json", report)
    print("Recognition Rank-1: %.4f; results: %s" % (recognition["rank1"], output), flush=True)
    return archive


if __name__ == "__main__":
    from run import main

    main(["recognition"] + __import__("sys").argv[1:])
