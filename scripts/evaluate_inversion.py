import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attacks.inversion_attack import InversionAttack
from attacks.descriptors import inversion_vector
from opengait.data.dataset import SequenceRecord, GaitDataset
from opengait.data.protocols import role
from opengait.evaluation.metrics import reconstruction_metrics
from opengait.evaluation.retrieval import subject_match
from opengait.modeling.models import SafeGaitER
from opengait.utils import seed_everything, get_device, resolve, load_checkpoint, write_json


def evaluate_inversion(config, archive=None, inputs=None, raw=False, reid_checkpoint=None):
    if archive is None:
        archive = load_checkpoint(inputs or resolve(config["output"]) / "attack_inputs.pt")
    if archive["config"]["dataset"] != config["dataset"]:
        raise ValueError("The inversion configuration and target model use different datasets.")
    seed_everything(config["seed"])
    device = get_device(config["device"])
    records = [
        SequenceRecord(**{k: r[k] for k in ("path", "subject", "condition", "view", "index")})
        for r in archive["records"]
    ]
    packets = archive["raw_packets"] if raw else [group[0] for group in archive["releases"]]
    explanations = torch.tensor(np.stack([inversion_vector(p) for p in packets]))
    embeddings = archive["embeddings"]
    members = [i for i, r in enumerate(records) if role(r, config["dataset"]) == "train"]
    test = [i for i, r in enumerate(records) if role(r, config["dataset"]) in ("gallery", "probe")]
    if not members or not test:
        raise ValueError("Inversion requires training and held-out sequence pools.")
    raw_input = packets[0]["level"] == "raw"
    model = InversionAttack(raw_input).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    dataset = GaitDataset([records[i] for i in members])
    settings = config["attacks"]
    loader = DataLoader(
        dataset,
        batch_size=settings["inversion_batch_size"],
        shuffle=True,
        num_workers=config["workers"],
    )
    for epoch in range(settings["inversion_epochs"]):
        model.train()
        for step, batch in enumerate(loader):
            actual = torch.tensor([members[i] for i in batch["record_index"].tolist()])
            pred = model(
                embeddings[actual].to(device), explanations[actual].to(device), logits=True
            )
            loss = F.binary_cross_entropy_with_logits(pred, batch["x"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"Inversion epoch {epoch + 1}/{settings['inversion_epochs']}", flush=True)
    model.eval()
    reid_model, gallery_embeddings, gallery_subjects = None, [], []
    if reid_checkpoint:
        saved = load_checkpoint(reid_checkpoint, device)
        if (
            saved["config"]["dataset"] != config["dataset"]
            or saved["config"]["training"]["strategy"] != "raw"
        ):
            raise ValueError(
                "Reconstructed Re-ID requires an independent raw recognizer on the same dataset."
            )
        if saved["config"]["seed"] != 777:
            raise ValueError(
                "Reconstructed Re-ID requires an independent recognition model trained with seed 777."
            )
        reid_model = SafeGaitER(saved["config"]).to(device).eval()
        reid_model.load_state_dict(saved["model"])
        gallery_records = [r for r in records if role(r, config["dataset"]) == "gallery"]
        with torch.no_grad():
            for batch in DataLoader(
                GaitDataset(gallery_records), batch_size=config["evaluation"]["batch_size"]
            ):
                gallery_embeddings.append(
                    reid_model(batch["x"].to(device), contribution=False)["embedding"].detach()
                )
        gallery_embeddings = torch.cat(gallery_embeddings)
        gallery_subjects = [r.subject for r in gallery_records]
    from opengait.evaluation.perceptual import PerceptualDistance

    perceptual = PerceptualDistance(device)
    totals = {"ssim": 0.0, "iou": 0.0}
    if perceptual.available:
        totals["lpips"] = 0.0
    count, reid_correct = 0, 0
    reconstructed_preview = None
    test_loader = DataLoader(
        GaitDataset([records[i] for i in test]),
        batch_size=min(4, settings["inversion_batch_size"]),
        num_workers=config["workers"],
    )
    with torch.no_grad():
        for batch in test_loader:
            actual = torch.tensor([test[i] for i in batch["record_index"].tolist()])
            pred = model(embeddings[actual].to(device), explanations[actual].to(device))
            target = batch["x"].to(device)
            metrics = reconstruction_metrics(pred, target)
            if perceptual.available:
                metrics["lpips"] = perceptual(pred, target)
            for key, value in metrics.items():
                totals[key] += value * len(pred)
            count += len(pred)
            if reconstructed_preview is None:
                reconstructed_preview = pred.cpu()
            if reid_model is not None:
                rec_embedding = reid_model(pred, contribution=False)["embedding"].detach()
                matched = subject_match(rec_embedding, gallery_embeddings, gallery_subjects)
                reid_correct += sum(
                    p == records[i].subject
                    for p, i in zip(matched["predicted_subjects"], actual.tolist())
                )
    report = {k: v / count for k, v in totals.items()}
    report.update(
        seed=config["seed"],
        dataset=config["dataset"],
        test_sequences=count,
        lpips_available=perceptual.available,
    )
    if reid_model is not None:
        report["reconstructed_reid_rank1"] = reid_correct / count
    output = resolve(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    name = "raw" if raw else "safe"
    torch.save(
        {
            "model": model.state_dict(),
            "raw": raw,
            "raw_input": raw_input,
            "interface": "raw_reference" if raw else "released",
            "config": config,
        },
        output / f"inversion_{name}.pt",
    )
    torch.save(reconstructed_preview, output / f"reconstruction_preview_{name}.pt")
    write_json(output / f"inversion_{name}.json", report)
    return report
