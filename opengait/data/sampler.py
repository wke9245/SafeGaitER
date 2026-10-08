import math
from collections import defaultdict
import numpy as np
from torch.utils.data import Sampler


class IdentityBatchSampler(Sampler):
    def __init__(self, records, p=8, k=4, seed=2025, steps=None):
        self.groups = defaultdict(list)
        for i, record in enumerate(records):
            self.groups[record.subject].append(i)
        if len(self.groups) < p:
            raise ValueError("The number of training identities is smaller than training.p.")
        self.p, self.k, self.seed = p, k, seed
        self.steps = steps or math.ceil(len(records) / (p * k))
        self.epoch = 0

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            batch = []
            for identity in rng.choice(sorted(self.groups), self.p, replace=False):
                group = self.groups[identity]
                batch.extend(rng.choice(group, self.k, replace=len(group) < self.k).tolist())
            yield batch
