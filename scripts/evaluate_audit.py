import numpy as np
import torch
from opengait.data.dataset import index_dataset, GaitDataset
from opengait.data.protocols import role
from opengait.modeling.models import SafeGaitER
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.evaluation.retrieval import subject_match
from opengait.utils import get_device, resolve, load_checkpoint, seed_everything, write_json
from scripts.evaluate_recognition import build_galleries
from baselines.deepleak_gait import raw_packet, protect_raw
from attacks.descriptors import raw_summary


def perturb_sequence(x, kind, severity=0.20):
    x = x.clone()
    if kind == "local_occlusion":
        x[:, 16 : 16 + round(64 * severity)] = 0
    elif kind == "lower_body_occlusion":
        x[:, 64 - round(64 * severity) :] = 0
    elif kind == "consecutive_frame_occlusion":
        x[10 : 10 + min(8, round(30 * severity))] = 0
    elif kind == "frame_dropping":
        count = max(1, round(30 * severity))
        x[10 : 10 + count] = x[9]
    elif kind == "silhouette_jitter":
        for t in range(len(x)):
            offset = int(torch.randint(-2, 3, (1,)))
            x[t] = torch.roll(x[t], offset, dims=1)
            if offset > 0:
                x[t, :, :offset] = 0
            elif offset < 0:
                x[t, :, offset:] = 0
    if kind == "combined_disturbance":
        x[:, 43:] = 0
        x[10:16] = 0
    return x


def supported_input(x, spatial, temporal, config):
    regions = config["model"]["spatial_groups"]
    boundaries = config["model"]["temporal_boundaries"]
    spatial_mask = torch.zeros(64, device=x.device)
    dominant_region = int(spatial.argmax())
    for strip in regions[dominant_region]:
        spatial_mask[strip * 8 : (strip + 1) * 8] = 1
    temporal_mask = torch.zeros(30, device=x.device)
    dominant_interval = int(temporal.argmax())
    temporal_mask[boundaries[dominant_interval] : boundaries[dominant_interval + 1]] = 1
    return x * temporal_mask[:, None, None] * spatial_mask[None, :, None]


def evaluate_audit(config, checkpoint, records=None):
    if config["dataset"] != "CASIA-B":
        raise ValueError("The controlled diagnostic categories are defined for CASIA-B.")
    seed_everything(config["seed"])
    device = get_device(config["device"])
    saved = load_checkpoint(checkpoint, device)
    if saved["config"]["dataset"] != config["dataset"]:
        raise ValueError("The checkpoint and audit configuration use different datasets.")
    config["training"]["strategy"] = saved["config"]["training"]["strategy"]
    config["model"] = saved["config"]["model"]
    config["risk"] = saved["config"]["risk"]
    model = SafeGaitER(config).to(device).eval()
    model.load_state_dict(saved["model"])
    transform, policy = PrivateExplanation(config).to(device), ReleasePolicy(config)
    records = records or index_dataset(
        resolve(config["data_root"]), config["dataset"], resolve(config["ou_split"])
    )
    gallery = build_galleries(model, records, config, device)["test"]
    selected = [r for r in records if role(r, config["dataset"]) in ("probe", "gallery")]
    # Protected gallery reference states, processed with the same retrieval/noise pipeline.
    ref_cache = {}
    for r in gallery["records"]:
        x = GaitDataset([r])[0]["x"][None].to(device)
        out = model(x, gallery["embeddings"], gallery["subjects"], [r.key], gallery["keys"])
        ref_cache[r.key] = transform(out)["rho"][0].detach()
    counts = {
        "region_visible": 0,
        "region_correct": 0,
        "temporal_visible": 0,
        "temporal_correct": 0,
        "summary_visible": 0,
        "summary_correct": 0,
        "packets": 0,
    }
    disturbance_counts = dict(counts)
    fid, disturbance_fid = [], []
    categories = {
        "normal_walking": [],
        "carried_object": [],
        "clothing_variation": [],
        "local_occlusion": [],
        "lower_body_occlusion": [],
        "consecutive_frame_occlusion": [],
        "silhouette_jitter": [],
        "frame_dropping": [],
        "combined_disturbance": [],
    }
    for r in selected:
        source = GaitDataset([r])[0]["x"].to(device)
        clean_out = model(
            source[None], gallery["embeddings"], gallery["subjects"], [r.key], gallery["keys"]
        )
        clean_state = transform(clean_out)
        predicted = clean_out["predicted_subjects"][0]
        same_view = [
            g
            for g in gallery["records"]
            if g.subject == predicted and g.view == r.view and g.key != r.key
        ]
        reference = (
            torch.stack([ref_cache[g.key] for g in same_view]).mean(0)
            if same_view
            else torch.tensor(config.get("diagnostic_reference", [0.70] * 3), device=device)
        )
        covariate = {"bg": "carried_object", "cl": "clothing_variation"}.get(
            r.condition.split("-")[0], "normal_walking"
        )
        states = [(source, clean_out, clean_state, covariate, reference)]
        if covariate == "normal_walking":
            for kind in (
                "local_occlusion",
                "lower_body_occlusion",
                "consecutive_frame_occlusion",
                "silhouette_jitter",
                "frame_dropping",
                "combined_disturbance",
            ):
                disturbed = perturb_sequence(source, kind)
                out = model(
                    disturbed[None],
                    gallery["embeddings"],
                    gallery["subjects"],
                    [r.key],
                    gallery["keys"],
                )
                states.append(
                    (disturbed, out, transform(out), kind, clean_state["rho"][0].detach())
                )
        for case_index, (x, out, state, truth, ref) in enumerate(states):
            counter = counts if case_index == 0 else disturbance_counts
            strategy = config["training"]["strategy"]
            if strategy in ("safe", "no_auditor"):
                packet = policy.packets(state, out["risk"], ref)[0]
            else:
                packet = raw_packet(out, 0)
                if strategy != "raw":
                    packet = protect_raw(
                        packet,
                        strategy,
                        config["release"]["sigma_max"],
                        deepleak=config.get("baselines", {}).get("deepleak"),
                    )
                if packet["level"] == "raw":
                    scores, importance = raw_summary(packet)
                else:
                    scores, importance = packet["spatial_scores"], packet["temporal_scores"]
                state["rho"] = torch.tensor([np.asarray(scores).tolist()], device=device)
                state["p"] = torch.tensor([np.asarray(importance).tolist()], device=device)
            raw_region = int(state["rho_raw"][0].argmax())
            raw_interval = int(state["p_raw"][0].argmax())
            counter["packets"] += 1
            if packet["level"] in ("low", "medium", "raw", "coarse"):
                if packet["level"] == "medium":
                    region, interval = packet["dominant_region"], packet["dominant_interval"]
                elif packet["level"] == "low":
                    region, interval = (
                        int(np.argmax(packet["spatial_scores"])),
                        int(np.argmax(packet["temporal_scores"])),
                    )
                else:
                    region, interval = int(state["rho"][0].argmax()), int(state["p"][0].argmax())
                counter["region_visible"] += 1
                counter["temporal_visible"] += 1
                counter["region_correct"] += int(region == raw_region)
                counter["temporal_correct"] += int(interval == raw_interval)
            if "diagnostic" in packet:
                success = int(packet["diagnostic"] == truth)
                counter["summary_visible"] += 1
                counter["summary_correct"] += success
                categories[truth].append(success)
            raw_supported = supported_input(x, state["rho_raw"][0], state["p_raw"][0], config)
            audit_supported = supported_input(x, state["rho"][0], state["p"][0], config)
            with torch.no_grad():
                variants = model(torch.stack((raw_supported, audit_supported)), contribution=False)[
                    "embedding"
                ].detach()
                matched = subject_match(
                    variants,
                    gallery["embeddings"],
                    gallery["subjects"],
                    [r.key, r.key],
                    gallery["keys"],
                    config["evaluation"]["temperature"],
                )
                candidate = sorted(set(gallery["subjects"])).index(out["predicted_subjects"][0])
                drops = (
                    out["confidence"][0].detach() - matched["subject_probabilities"][:, candidate]
                )
                fidelity = (1 - (drops[0] - drops[1]).abs() / (drops[0] + 1e-8)).clamp(0, 1)
                (fid if case_index == 0 else disturbance_fid).append(float(fidelity))
    report = {
        "seed": config["seed"],
        "counts": counts,
        "internal_fidelity": sum(fid) / len(fid),
        "controlled_disturbance_counts": disturbance_counts,
        "controlled_disturbance_fidelity": sum(disturbance_fid) / len(disturbance_fid)
        if disturbance_fid
        else None,
        "region_coverage": counts["region_visible"] / counts["packets"],
        "temporal_coverage": counts["temporal_visible"] / counts["packets"],
        "summary_coverage": counts["summary_visible"] / counts["packets"],
        "region_agreement": counts["region_correct"] / counts["region_visible"]
        if counts["region_visible"]
        else None,
        "temporal_agreement": counts["temporal_correct"] / counts["temporal_visible"]
        if counts["temporal_visible"]
        else None,
        "conditional_summary_agreement": counts["summary_correct"] / counts["summary_visible"]
        if counts["summary_visible"]
        else None,
        "category_counts": {
            k: {"visible": len(v), "correct": sum(v)} for k, v in categories.items()
        },
        "support_rule": "dominant coarse region and interval intersection",
    }
    write_json(resolve(config["output"]) / "audit.json", report)
    return report
