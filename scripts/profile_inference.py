import time
import numpy as np
import torch
from opengait.data.dataset import index_dataset, GaitDataset
from opengait.data.protocols import role
from opengait.modeling.models import SafeGaitER
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.utils import get_device, resolve, load_checkpoint, write_json, seed_everything
from scripts.evaluate_recognition import build_galleries
from baselines.deepleak_gait import raw_packet, protect_raw


def profile(config, checkpoint, repetitions=100, warmup=20):
    if repetitions < 1 or warmup < 0:
        raise ValueError("Profiling requires positive repetitions and nonnegative warm-up.")
    seed_everything(config["seed"])
    device = get_device(config["device"])
    saved = load_checkpoint(checkpoint, device)
    if saved["config"]["dataset"] != config["dataset"]:
        raise ValueError("The checkpoint and profiling configuration use different datasets.")
    config["training"]["strategy"] = saved["config"]["training"]["strategy"]
    config["model"] = saved["config"]["model"]
    config["risk"] = saved["config"]["risk"]
    model = SafeGaitER(config).to(device).eval()
    model.load_state_dict(saved["model"])
    transform, policy = PrivateExplanation(config).to(device), ReleasePolicy(config)
    records = index_dataset(
        resolve(config["data_root"]), config["dataset"], resolve(config["ou_split"])
    )
    gallery = build_galleries(model, records, config, device)["test"]
    probe = next(r for r in records if role(r, config["dataset"]) == "probe")
    x = GaitDataset([probe])[0]["x"][None].to(device)
    values = []
    for iteration in range(warmup + repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        strategy = config["training"]["strategy"]
        safe = strategy in ("safe", "no_auditor")
        out = model(
            x,
            gallery["embeddings"],
            gallery["subjects"],
            [probe.key],
            gallery["keys"],
            include_risk=safe,
        )
        if safe:
            state = transform(out)
            packet = policy.packets(state, out["risk"])
        else:
            packet = raw_packet(out, 0)
            if strategy != "raw":
                packet = protect_raw(
                    packet,
                    strategy,
                    config["release"]["sigma_max"],
                    deepleak=config.get("baselines", {}).get("deepleak"),
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        if iteration >= warmup:
            values.append((time.perf_counter() - start) * 1000)
    report = {
        "seed": config["seed"],
        "dataset": config["dataset"],
        "batch_size": 1,
        "repetitions": repetitions,
        "warmup": warmup,
        "latency_mean_ms": float(np.mean(values)),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "includes_local_gradient": True,
        "includes_single_view_stability": safe,
        "includes_transform_and_release": True,
        "strategy": strategy,
        "device": str(device),
        "parameters_encoder_and_stability": sum(p.numel() for p in model.parameters()),
    }
    write_json(resolve(config["output"]) / "profile.json", report)
    return report
