import csv
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from opengait.data.dataset import GaitDataset, index_dataset
from opengait.data.protocols import role, auxiliary_pool
from opengait.data.sampler import IdentityBatchSampler
from opengait.data.validation import (
    validate_training_pool,
    validate_auxiliary_pool,
    validate_unique_records,
)
from opengait.modeling.models import SafeGaitER
from opengait.modeling.models.private_explanation import PrivateExplanation
from opengait.modeling.models.release_policy import ReleasePolicy
from opengait.modeling.models.soft_routing import SoftRouting
from opengait.modeling.models.privacy_auditors import PrivacyAuditors
from opengait.modeling.models.stability_predictor import stability_target
from opengait.modeling.losses.safegait_losses import objective
from opengait.utils import (
    get_device,
    resolve,
    seed_everything,
    optimizer_groups,
    write_json,
    load_checkpoint,
)


def set_frozen(module, frozen):
    for p in module.parameters():
        p.requires_grad_(not frozen)


def adversarial_step(
    model,
    routing,
    auditors,
    transform,
    policy,
    main_optimizer,
    audit_optimizer,
    x,
    second,
    labels,
    auxiliary,
    config,
    joint=True,
):
    out = model(x)
    state = transform(out)
    reference = config.get("diagnostic_reference", [0.70, 0.70, 0.70])
    adversarial = joint and config["training"]["strategy"] == "safe"
    audit_value = None
    if adversarial:
        auxiliary_out = model(auxiliary)
        auxiliary_state = transform(auxiliary_out)
        branches = policy.branch_vectors(state, out["risk"], reference)
        auxiliary_branches = policy.branch_vectors(
            auxiliary_state, auxiliary_out["risk"], reference
        )
        # First objective: minimize auditor loss on detached encoder outputs.
        audit_optimizer.zero_grad(set_to_none=True)
        audit_rep = routing([z.detach() for z in branches], out["risk"].detach())
        aux_rep = routing([z.detach() for z in auxiliary_branches], auxiliary_out["risk"].detach())
        audit_fit = auditors.loss(
            audit_rep,
            aux_rep,
            labels,
            config["training"]["lambda_y"],
            config["training"]["lambda_l"],
        )
        audit_fit.backward()
        audit_optimizer.step()
        audit_optimizer.zero_grad(set_to_none=True)
        # Second objective: freeze auditor-side parameters, reverse their input gradients.
        set_frozen(routing, True)
        set_frozen(auditors, True)
        audit_value = auditors.loss(
            routing(branches, out["risk"]),
            routing(auxiliary_branches, auxiliary_out["risk"]),
            labels,
            config["training"]["lambda_y"],
            config["training"]["lambda_l"],
        )
    target = None
    if joint:
        # Reference view is a supervision target, so no higher-order graph is retained.
        model.eval()
        with torch.enable_grad():
            reference_out = model(second)
        model.train()
        target = stability_target(out["descriptor"], reference_out["descriptor"])
    losses = objective(out, state, labels, target, config, audit_value, joint)
    main_optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    main_optimizer.step()
    if adversarial:
        set_frozen(routing, False)
        set_frozen(auditors, False)
    return {k: float(v.detach()) for k, v in losses.items()}


def train(config, records=None, auxiliary_records=None, resume=None):
    if config["training"]["epochs"] < 1:
        raise ValueError("Training requires at least one epoch.")
    if config["training"]["recognition_warmup"] < 0 or config["training"]["lr_warmup"] < 0:
        raise ValueError("Training warm-up counts must be nonnegative.")
    seed_everything(config["seed"])
    device = get_device(config["device"])
    split = resolve(config["ou_split"])
    if records is None:
        records = index_dataset(resolve(config["data_root"]), config["dataset"], split)
    train_records = [r for r in records if role(r, config["dataset"]) == "train"]
    if not train_records:
        raise ValueError("No recognition-training sequences match the configured protocol.")
    validate_unique_records(records)
    validate_training_pool(train_records, config["dataset"], config["model"]["num_classes"])
    settings = config["training"]
    if settings["strategy"] == "safe" and settings["epochs"] > settings["recognition_warmup"]:
        if auxiliary_records is None:
            aux_dataset = "OU-MVLP" if config["dataset"] == "CASIA-B" else "CASIA-B"
            auxiliary_records = index_dataset(resolve(config["auxiliary_root"]), aux_dataset, split)
            auxiliary_records = auxiliary_pool(auxiliary_records, config["dataset"])
        if not auxiliary_records:
            raise ValueError(
                "Adversarial training requires the auxiliary silhouette pool from the other dataset."
            )
        validate_auxiliary_pool(auxiliary_records)
    model = SafeGaitER(config).to(device)
    transform, policy = PrivateExplanation(config).to(device), ReleasePolicy(config)
    routing = SoftRouting(config).to(device)
    auditors = PrivacyAuditors(config["model"]["num_classes"]).to(device)
    primary = nn.ModuleDict(
        {name: part for name, part in model.named_children() if name != "stability"}
    )
    main_groups = optimizer_groups(primary, settings["lr_backbone"], settings["weight_decay"])
    main_groups += optimizer_groups(
        model.stability, settings["lr_privacy"], settings["weight_decay"]
    )
    main_optimizer = torch.optim.AdamW(main_groups, betas=(0.9, 0.999))
    side = nn.ModuleDict({"routing": routing, "auditors": auditors})
    audit_optimizer = torch.optim.AdamW(
        optimizer_groups(side, settings["lr_privacy"], settings["weight_decay"]), betas=(0.9, 0.999)
    )
    initial_lrs = [g["lr"] for g in main_optimizer.param_groups]
    audit_lrs = [g["lr"] for g in audit_optimizer.param_groups]
    sampler = IdentityBatchSampler(train_records, settings["p"], settings["k"], config["seed"])
    loader = DataLoader(
        GaitDataset(train_records, dual_view=True, training=True),
        batch_sampler=sampler,
        num_workers=config["workers"],
    )
    aux_loader = None
    if auxiliary_records:
        aux_loader = DataLoader(
            GaitDataset(auxiliary_records, training=True),
            batch_size=settings["p"] * settings["k"],
            shuffle=True,
            num_workers=config["workers"],
        )
    output = resolve(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    start = 0
    if resume:
        checkpoint = load_checkpoint(resume, device)
        if (
            checkpoint["config"]["dataset"] != config["dataset"]
            or checkpoint["config"]["training"]["strategy"] != settings["strategy"]
        ):
            raise ValueError(
                "The resumed checkpoint and configuration use different datasets or strategies."
            )
        model.load_state_dict(checkpoint["model"])
        routing.load_state_dict(checkpoint["routing"])
        auditors.load_state_dict(checkpoint["auditors"])
        main_optimizer.load_state_dict(checkpoint["main_optimizer"])
        audit_optimizer.load_state_dict(checkpoint["audit_optimizer"])
        start = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    write_json(output / "config.json", config)
    write_json(
        output / "training_manifest.json",
        {
            "dataset": config["dataset"],
            "seed": config["seed"],
            "train_sequences": len(train_records),
            "train_identities": config["model"]["num_classes"],
            "auxiliary_sequences": len(auxiliary_records or []),
            "reserved_nonmembers_used": 0,
        },
    )
    aux_iterator = iter(aux_loader) if aux_loader else None
    with (output / "training.csv").open(
        "a" if resume else "w", newline="", encoding="utf-8"
    ) as log:
        writer = csv.DictWriter(
            log,
            fieldnames=[
                "epoch",
                "phase",
                "recognition",
                "stability",
                "auditability",
                "compactness",
                "risk",
                "adversarial",
                "total",
            ],
        )
        if not resume:
            writer.writeheader()
        for epoch in range(start, settings["epochs"]):
            sampler.set_epoch(epoch)
            for optimizer, bases in ((main_optimizer, initial_lrs), (audit_optimizer, audit_lrs)):
                for group, base in zip(optimizer.param_groups, bases):
                    if epoch < settings["lr_warmup"]:
                        group["lr"] = base * (epoch + 1) / settings["lr_warmup"]
                    else:
                        progress = (epoch - settings["lr_warmup"]) / max(
                            1, settings["epochs"] - settings["lr_warmup"] - 1
                        )
                        group["lr"] = (
                            settings["min_lr"]
                            + (base - settings["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2
                        )
            model.train()
            joint = epoch >= settings["recognition_warmup"] and settings["strategy"] in (
                "safe",
                "no_auditor",
            )
            totals = {}
            for step, batch in enumerate(loader):
                x, second, labels = (
                    batch["x"].to(device),
                    batch["view2"].to(device),
                    batch["label"].to(device),
                )
                aux = None
                if joint and settings["strategy"] == "safe":
                    try:
                        aux_batch = next(aux_iterator)
                    except StopIteration:
                        aux_iterator = iter(aux_loader)
                        aux_batch = next(aux_iterator)
                    aux = aux_batch["x"].to(device)
                losses = adversarial_step(
                    model,
                    routing,
                    auditors,
                    transform,
                    policy,
                    main_optimizer,
                    audit_optimizer,
                    x,
                    second,
                    labels,
                    aux,
                    config,
                    joint,
                )
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0) + value
                if (step + 1) % 10 == 0:
                    print(
                        f"epoch {epoch + 1}/{settings['epochs']} step {step + 1}/{len(loader)} loss={losses['total']:.4f}",
                        flush=True,
                    )
            totals = {k: v / len(loader) for k, v in totals.items()}
            writer.writerow(
                dict(epoch=epoch + 1, phase="joint" if joint else "recognition", **totals)
            )
            log.flush()
            checkpoint = {
                "model": model.state_dict(),
                "routing": routing.state_dict(),
                "auditors": auditors.state_dict(),
                "main_optimizer": main_optimizer.state_dict(),
                "audit_optimizer": audit_optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": config,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            temporary = output / "checkpoint.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "checkpoint.pt")
            print(
                f"epoch {epoch + 1}/{settings['epochs']} {'joint' if joint else 'recognition'} loss={totals['total']:.4f}",
                flush=True,
            )
    return output / "checkpoint.pt"


if __name__ == "__main__":
    from run import main

    main(["train"] + __import__("sys").argv[1:])
