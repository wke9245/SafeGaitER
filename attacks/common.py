import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from opengait.utils import seed_everything


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def fit_numeric(model, inputs, labels, settings, device="cpu", seed=2025, multiclass=False):
    seed_everything(seed)
    model.to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    x, y = (
        torch.as_tensor(inputs, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long if multiclass else torch.float32),
    )
    for epoch in range(settings["epochs"]):
        order = torch.randperm(len(y))
        for idx in order.split(settings["batch_size"]):
            logits = model(x[idx].to(device))
            loss = (
                F.cross_entropy(logits, y[idx].to(device))
                if multiclass
                else F.binary_cross_entropy_with_logits(logits.squeeze(-1), y[idx].to(device))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def predict_numeric(model, x, device="cpu", batch_size=512):
    with torch.no_grad():
        return torch.cat(
            [
                model(torch.as_tensor(part, dtype=torch.float32, device=device)).cpu()
                for part in np.array_split(x, max(1, int(np.ceil(len(x) / batch_size))))
            ]
        )


def check_identity_disjoint(records, fit, test):
    train_ids = {records[i].subject for i in fit}
    test_ids = {records[i].subject for i in test}
    if train_ids & test_ids:
        raise ValueError("Attacker fitting identities overlap with evaluation identities.")
