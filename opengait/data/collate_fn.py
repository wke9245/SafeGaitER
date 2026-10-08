import torch


def sample_frames(sequence, count=30, random=False):
    length = len(sequence)
    if length < 1:
        raise ValueError("The silhouette sequence is empty.")
    if random:
        index = (
            (torch.randperm(length)[:count] if length >= count else torch.randint(length, (count,)))
            .sort()
            .values
        )
    else:
        index = torch.linspace(0, length - 1, count).round().long()
    return sequence[index].clone()
