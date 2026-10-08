"""Optional official LPIPS. Missing dependencies never produce substitute values."""

import importlib.util


class PerceptualDistance:
    def __init__(self, device):
        self.network = None
        if importlib.util.find_spec("lpips") is not None:
            import lpips

            self.network = lpips.LPIPS(net="alex").to(device).eval()
            for parameter in self.network.parameters():
                parameter.requires_grad_(False)

    @property
    def available(self):
        return self.network is not None

    def __call__(self, predicted, target):
        if self.network is None:
            return None
        first = predicted.flatten(0, 1)[:, None].expand(-1, 3, -1, -1) * 2 - 1
        second = target.flatten(0, 1)[:, None].expand(-1, 3, -1, -1) * 2 - 1
        return float(self.network(first, second).mean())
