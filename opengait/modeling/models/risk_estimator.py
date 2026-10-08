import math
import torch
from torch import nn


class RiskEstimator(nn.Module):
    def __init__(
        self, exponents=(1, 1, 1, 1), top_k_ratio=0.20, aggregation="product", use_confidence=True
    ):
        super().__init__()
        self.exponents = exponents
        self.ratio = top_k_ratio
        self.aggregation = aggregation
        self.use_confidence = use_confidence

    def forward(self, omega, alpha, contribution, stability, confidence):
        factors = [omega, alpha[:, :, None].expand_as(omega), contribution, stability]
        terms = torch.stack(
            [
                factor.clamp_min(0).pow(exp) if exp else torch.ones_like(factor)
                for factor, exp in zip(factors, self.exponents)
            ]
        )
        if self.aggregation == "product":
            joint = terms.prod(0)
        elif self.aggregation == "sum":
            joint = terms.mean(0)
        elif self.aggregation == "max":
            joint = terms.max(0).values
        else:
            raise ValueError("Unknown risk aggregation method.")
        flat = joint.flatten(1)
        low, high = flat.min(1).values[:, None], flat.max(1).values[:, None]
        normalized = (flat - low) / (high - low + 1e-8)
        k = max(1, min(flat.shape[1], math.ceil(flat.shape[1] * self.ratio)))
        values, indices = normalized.topk(k, dim=1)
        return {
            "joint": joint,
            "local_risk": normalized.reshape_as(joint),
            "risk": (confidence if self.use_confidence else torch.ones_like(confidence))
            * values.mean(1),
            "risk_indices": indices,
        }
