import torch
import torch.nn.functional as F
from .diagnostic_mapper import CATEGORIES, diagnostic_categories


class ReleasePolicy:
    def __init__(self, config):
        self.tau1 = config["release"]["tau1"]
        self.tau2 = config["release"]["tau2"]

    def levels(self, risk):
        r = risk.detach()
        return torch.where(r < self.tau1, 0, torch.where(r < self.tau2, 1, 2))

    def branch_vectors(self, state, risk, reference=None):
        level = self.levels(risk)
        category = diagnostic_categories(state, reference)
        qr = F.one_hot(level, 3).float()
        m = F.one_hot(state["s"].detach().argmax(1), 3).float()
        interval = F.one_hot(state["p_release"].detach().argmax(1), 4).float()
        d = F.one_hot(category, 9).float()
        return [
            torch.cat((state["s"], state["p_release"], qr), 1),
            torch.cat((m, interval, qr, d), 1).detach(),
            torch.cat((qr, d), 1).detach(),
        ]

    def packets(self, state, risk, reference=None):
        level = self.levels(risk).cpu().tolist()
        categories = diagnostic_categories(state, reference).cpu().tolist()
        scores = state["s"].detach().cpu().tolist()
        temporal = state["p_release"].detach().cpu().tolist()
        packets = []
        for i, q in enumerate(level):
            if q == 0:
                packet = {
                    "level": "low",
                    "spatial_scores": scores[i],
                    "temporal_scores": temporal[i],
                }
            elif q == 1:
                packet = {
                    "level": "medium",
                    "dominant_region": max(range(3), key=lambda j: scores[i][j]),
                    "dominant_interval": max(range(4), key=lambda j: temporal[i][j]),
                    "diagnostic": CATEGORIES[categories[i]],
                }
            else:
                packet = {"level": "high", "diagnostic": CATEGORIES[categories[i]]}
            validate_packet(packet)
            packets.append(packet)
        return packets


def validate_packet(packet):
    allowed = {
        "low": {"level", "spatial_scores", "temporal_scores"},
        "medium": {"level", "dominant_region", "dominant_interval", "diagnostic"},
        "high": {"level", "diagnostic"},
    }
    if packet.get("level") not in allowed or set(packet) != allowed[packet["level"]]:
        raise ValueError("Release packet contains unauthorized fields or omits required fields.")
    if packet["level"] == "low":
        if len(packet["spatial_scores"]) != 3 or len(packet["temporal_scores"]) != 4:
            raise ValueError("Invalid release packet dimensions.")
