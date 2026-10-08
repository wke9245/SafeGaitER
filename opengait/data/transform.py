import torch
import torch.nn.functional as F
from .collate_fn import sample_frames


def augment(sequence, count=30):
    keep = torch.rand(len(sequence)) >= 0.10
    if not keep.any():
        keep[0] = True
    x = sample_frames(sequence[keep], count, random=True)
    shifts = torch.randint(-1, 2, (count,))
    x = x[(torch.arange(count) + shifts).clamp(0, count - 1)]
    dilated = F.max_pool2d(x[:, None], 3, 1, 1)[:, 0]
    eroded = -F.max_pool2d(-x[:, None], 3, 1, 1)[:, 0]
    edge = (dilated != eroded) & (torch.rand_like(x) < 0.03)
    x = torch.where(edge, 1 - x, x)
    for strip in range(8):
        if torch.rand(()) < 0.05:
            x[:, strip * 8 : (strip + 1) * 8] = 0
    return x
