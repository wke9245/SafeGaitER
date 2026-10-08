import torch
from torch import nn


def distribution(x):
    floored = x + 1e-8
    return floored / floored.sum(-1, keepdim=True)


def masked_average(values, mask, dims):
    count = mask.sum(dims)
    average = (values * mask).sum(dims) / count.clamp_min(1)
    return torch.where(count > 0, average, torch.zeros_like(average)), count


class PrivateExplanation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.groups = config["model"]["spatial_groups"]
        self.bounds = config["model"]["temporal_boundaries"]
        self.settings = config["release"]

    def perturb(self, value, risk, count=None, noise=True):
        sigma = (
            self.settings["sigma_min"]
            + (self.settings["sigma_max"] - self.settings["sigma_min"]) * risk
        )
        result = (value + sigma * torch.randn_like(value) if noise else value).clamp(0, 1)
        return (
            torch.where(count > 0, result, torch.zeros_like(result))
            if count is not None
            else result
        )

    def forward(self, out, noise=True):
        omega, alpha = out["omega"], out["alpha"]
        risk, stability = out["local_risk"], out["stability"]
        # Joint evidence always uses its original four source factors.
        evidence = omega * alpha[:, :, None] * out["contribution"] * stability
        clipped = torch.minimum(
            evidence, self.settings["rho0"] + self.settings["rho1"] * (1 - risk)
        )
        mask = (risk.detach() < self.settings["tau_mask"]).to(risk.dtype)
        fine = clipped * mask
        spatial_risk, rho, rho_raw, spatial, spatial_raw, cell_risk = [], [], [], [], [], []
        for group in self.groups:
            local_mask, local_risk = mask[:, :, group], risk[:, :, group]
            rm = local_risk.mean((1, 2))
            mean_rho, count = masked_average(omega[:, :, group], local_mask, (1, 2))
            joint_mean, cell_count = masked_average(clipped[:, :, group], local_mask, 2)
            rc = local_risk.mean(2)
            rho.append(self.perturb(mean_rho, rm, count, noise))
            rho_raw.append(omega[:, :, group].mean((1, 2)))
            spatial.append(self.perturb(joint_mean, rc, cell_count, noise))
            spatial_raw.append(evidence[:, :, group].mean(2))
            spatial_risk.append(rm)
            cell_risk.append(rc)
        interval_risk, temporal, temporal_raw, interval_reliability = [], [], [], []
        for left, right in zip(self.bounds[:-1], self.bounds[1:]):
            rr = risk[:, left:right].mean((1, 2))
            tmp = alpha[:, left:right].mean(1)
            interval_rho, count = masked_average(omega[:, left:right], mask[:, left:right], (1, 2))
            temporal.append(self.perturb(tmp, rr, noise=noise))
            temporal_raw.append(tmp)
            interval_reliability.append(self.perturb(interval_rho, rr, count, noise))
            interval_risk.append(rr)
        rho = torch.stack(rho, 1)
        p = torch.stack(temporal, 1)
        total = p.sum(1, keepdim=True)
        p = torch.where(total > 0, p / total.clamp_min(1e-8), torch.full_like(p, 0.25))
        p_raw = torch.stack(temporal_raw, 1)
        p_raw = p_raw / p_raw.sum(1, keepdim=True).clamp_min(1e-8)
        rs, rt = torch.stack(spatial_risk, 1), torch.stack(interval_risk, 1)
        psi, _ = masked_average(stability, mask, (1, 2))
        sp = torch.stack(spatial, -1)
        raw_sp = torch.stack(spatial_raw, -1)
        return {
            "fine": fine,
            "audit_spatial": sp,
            "raw_spatial": raw_sp,
            "rho": rho,
            "rho_raw": torch.stack(rho_raw, 1),
            "interval_reliability": torch.stack(interval_reliability, 1),
            "p": p,
            "p_raw": p_raw,
            "psi": psi,
            "spatial_risk": rs,
            "temporal_risk": rt,
            "s": rho * (rs.detach() < self.settings["tau_mask"]),
            "p_release": p * (rt.detach() < self.settings["tau_mask"]),
            "e_raw": torch.cat((raw_sp.flatten(1), p_raw), 1),
            "e_audit": torch.cat((sp.flatten(1), p), 1),
        }
