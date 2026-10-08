import torch
from torch import nn
from .backbones import HierarchicalEncoder, GaitPartEncoder
from .identity_contribution import identity_contribution
from .stability_predictor import StabilityPredictor, local_descriptor
from .risk_estimator import RiskEstimator


class SafeGaitER(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["model"]
        self.config = config
        self.strips = cfg.get("strips", 8)
        self.backbone = (
            GaitPartEncoder() if cfg.get("backbone") == "gaitpart" else HierarchicalEncoder()
        )
        self.reliability = nn.Linear(128, 1)
        self.projection = nn.Linear(128, cfg.get("embedding_dim", 256))
        self.temporal = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.classifier = nn.Linear(256, cfg["num_classes"])
        self.stability = StabilityPredictor()
        self.risk_estimator = RiskEstimator(**config["risk"])

    def extract_features(self, x):
        if x.ndim == 5 and x.shape[2] == 1:
            x = x[:, :, 0]
        if x.ndim != 4 or tuple(x.shape[1:]) != (30, 64, 44):
            raise ValueError("Input must have shape [B, 30, 64, 44].")
        b, t, height, width = x.shape
        features = self.backbone(x.reshape(b * t, 1, height, width))
        return features.reshape(b, t, *features.shape[1:])

    def evidence(self, features):
        b, t, c, h, w = features.shape
        strips = features.reshape(b, t, c, self.strips, h // self.strips, w)
        q = strips.abs().mean(dim=(4, 5)).permute(0, 1, 3, 2)
        pooled = strips.mean(dim=(4, 5)).permute(0, 1, 3, 2)
        omega = self.reliability(q).sigmoid().squeeze(-1)
        frame = self.projection((omega[..., None] * pooled).sum(2))
        alpha = self.temporal(frame).squeeze(-1).softmax(dim=1)
        embedding = (alpha[..., None] * frame).sum(1)
        logits = self.classifier(embedding)
        return {"q": q, "omega": omega, "alpha": alpha, "embedding": embedding, "logits": logits}

    def forward(
        self,
        x,
        gallery=None,
        gallery_subjects=None,
        query_keys=None,
        gallery_keys=None,
        contribution=True,
        include_risk=True,
    ):
        # Explicit enable_grad allows explanation inference even under a caller's no_grad.
        with torch.enable_grad():
            if self.training:
                features = self.extract_features(x)
            else:
                with torch.no_grad():
                    features = self.extract_features(x)
                features = features.detach().requires_grad_(contribution)
            out = self.evidence(features)
            if not contribution:
                return out
            if gallery is None:
                predicted = out["logits"].detach().argmax(1)
                target = out["logits"].gather(1, predicted[:, None]).squeeze(1)
                confidence = out["logits"].softmax(1).max(1).values
            else:
                from ...evaluation.retrieval import subject_match

                matched = subject_match(
                    out["embedding"],
                    gallery,
                    gallery_subjects,
                    query_keys,
                    gallery_keys,
                    self.config["evaluation"]["temperature"],
                )
                out.update(matched)
                target, confidence = matched["matched_similarity"], matched["confidence"]
            c = identity_contribution(features, target, self.strips, create_graph=self.training)
            descriptor = local_descriptor(out["q"], out["omega"], out["alpha"], c)
            keyframes = torch.zeros_like(out["alpha"])
            keyframes.scatter_(1, out["alpha"].detach().topk(6, dim=1).indices, 1)
            out.update(
                contribution=c, descriptor=descriptor, confidence=confidence, keyframes=keyframes
            )
            if include_risk:
                stability = self.stability(descriptor, confidence)
                out["stability"] = stability
                out.update(
                    self.risk_estimator(out["omega"], out["alpha"], c, stability, confidence)
                )
            return out
