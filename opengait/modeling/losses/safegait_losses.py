import torch
import torch.nn.functional as F
from ..models.private_explanation import distribution


def triplet_loss(embedding, labels, margin=0.2):
    embedding = F.normalize(embedding, dim=1)
    distance = torch.cdist(embedding, embedding)
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    different = labels[:, None] != labels[None, :]
    valid = same.any(1) & different.any(1)
    hardest_positive = distance.masked_fill(~same, -float("inf")).max(1).values
    hardest_negative = distance.masked_fill(~different, float("inf")).min(1).values
    return (
        F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()
        if valid.any()
        else embedding.sum() * 0
    )


def kl(raw, protected):
    u, v = distribution(raw), distribution(protected)
    return (u * (u.log() - v.log())).sum(1).mean()


def auditability_loss(state, lambda_p=0.5, lambda_s=0.5):
    l1 = (state["e_audit"] - state["e_raw"]).abs().sum(1).mean()
    return (
        l1
        + lambda_p * kl(state["p_raw"], state["p"])
        + lambda_s * kl(state["rho_raw"], state["rho"])
    )


def objective(out, state, labels, stability_target, cfg, auditor_loss=None, joint=True):
    t = cfg["training"]
    rec = F.cross_entropy(out["logits"], labels) + t["lambda_tri"] * triplet_loss(
        out["embedding"], labels, t["margin"]
    )
    losses = {"recognition": rec}
    if joint:
        losses["stability"] = F.mse_loss(out["stability"], stability_target)
        losses["auditability"] = auditability_loss(state, t["lambda_p"], t["lambda_s"])
        losses["compactness"] = state["fine"].abs().mean()
        losses["risk"] = (out["local_risk"] * state["fine"].abs()).mean()
        total = (
            rec
            + losses["stability"]
            + t["lambda_a"] * losses["auditability"]
            + t["lambda_c"] * losses["compactness"]
            + t["lambda_r"] * losses["risk"]
        )
        if auditor_loss is not None:
            losses["adversarial"] = auditor_loss
            total = total - t["lambda_adv"] * auditor_loss
    else:
        total = rec
    losses["total"] = total
    return losses
