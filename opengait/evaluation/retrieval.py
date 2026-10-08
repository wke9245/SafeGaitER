import torch
import torch.nn.functional as F


def subject_match(
    query, gallery, gallery_subjects, query_keys=None, gallery_keys=None, temperature=0.05
):
    similarity = F.normalize(query, dim=1) @ F.normalize(gallery.detach(), dim=1).T
    allowed = torch.ones_like(similarity, dtype=torch.bool)
    if query_keys is not None and gallery_keys is not None:
        for i, key in enumerate(query_keys):
            for j, gkey in enumerate(gallery_keys):
                if key == gkey:
                    allowed[i, j] = False
    subjects = sorted(set(gallery_subjects))
    means = []
    for subject in subjects:
        mask = (
            torch.tensor([s == subject for s in gallery_subjects], device=query.device)[None, :]
            & allowed
        )
        count = mask.sum(1)
        average = (similarity * mask).sum(1) / count.clamp_min(1)
        means.append(average.masked_fill(count == 0, -float("inf")))
    means = torch.stack(means, 1)
    if not torch.isfinite(means).any(1).all():
        raise ValueError("No gallery candidates remain after self-match exclusion.")
    probabilities = (means / temperature).softmax(1)
    confidence, predicted = probabilities.max(1)
    predicted_subjects = [subjects[i] for i in predicted.detach().cpu().tolist()]
    top_mask = (
        torch.tensor(
            [[s == p for s in gallery_subjects] for p in predicted_subjects], device=query.device
        )
        & allowed
    )
    matched_similarity, matched_index = similarity.masked_fill(~top_mask, -float("inf")).max(1)
    return {
        "confidence": confidence,
        "predicted_subjects": predicted_subjects,
        "matched_similarity": matched_similarity,
        "matched_index": matched_index,
        "subject_probabilities": probabilities,
    }
