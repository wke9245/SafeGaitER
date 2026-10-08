import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .collate_fn import sample_frames
from .transform import augment


@dataclass(frozen=True)
class SequenceRecord:
    path: str
    subject: str
    condition: str
    view: str
    index: int = -1

    @property
    def key(self):
        return "/".join((self.subject, self.condition, self.view))

    def metadata(self):
        return dict(asdict(self), key=self.key)


def index_dataset(root, dataset="CASIA-B", split_file=None):
    if dataset not in {"CASIA-B", "OU-MVLP"}:
        raise ValueError("Unsupported dataset: " + str(dataset))
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: " + str(root))
    lookup = {}
    if dataset == "OU-MVLP":
        import json

        if split_file is None or not Path(split_file).is_file():
            raise FileNotFoundError("OU-MVLP requires the official identity partition file.")
        split = json.loads(Path(split_file).read_text(encoding="utf-8"))
        train = sorted(str(s).zfill(5) for s in split["TRAIN_SET"])
        test = sorted(str(s).zfill(5) for s in split["TEST_SET"])
        if len(train) != 5153 or len(test) != 5154:
            raise ValueError("OU-MVLP requires 5,153 training and 5,154 held-out identities.")
        if len(set(train + test)) != len(train) + len(test):
            raise ValueError("OU-MVLP identity partitions must be unique and disjoint.")
        if any(not subject.isdigit() for subject in train + test):
            raise ValueError("OU-MVLP identity partitions must use numeric identifiers.")
        lookup = {s: i for i, s in enumerate(train + test)}
    records = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        subject = subject_dir.name.zfill(3 if dataset == "CASIA-B" else 5)
        if dataset == "OU-MVLP" and subject not in lookup:
            continue
        index = int(subject) - 1 if dataset == "CASIA-B" else lookup[subject]
        for cond_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            for view_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                arrays = sorted(
                    p for p in view_dir.iterdir() if p.suffix.lower() in (".pkl", ".npy", ".npz")
                )
                images = sorted(
                    p
                    for p in view_dir.iterdir()
                    if p.suffix.lower() in (".png", ".bmp", ".jpg", ".jpeg")
                )
                if arrays or images:
                    preferred = [p for p in arrays if "sil" in p.stem.lower()]
                    candidates = preferred or arrays
                    if len(candidates) > 1:
                        raise ValueError("Ambiguous silhouette sequence files in " + str(view_dir))
                    source = candidates[0] if candidates else view_dir
                    records.append(
                        SequenceRecord(
                            str(source), subject, cond_dir.name.lower(), view_dir.name, index
                        )
                    )
    if not records:
        raise ValueError(
            "No silhouette sequences found. Expected subject/condition/view/sequence files or image folders."
        )
    from .validation import validate_unique_records

    validate_unique_records(records)
    return records


def normalize_frames(array, resolution=(64, 44)):
    a = np.asarray(array)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 4 and a.shape[-1] in (1, 3, 4):
        a = a.mean(-1)
    if a.ndim != 3 or len(a) == 0:
        raise ValueError("A silhouette sequence must be a nonempty [frames, height, width] array.")
    if tuple(a.shape[1:]) == tuple(resolution):
        return torch.from_numpy((a > (0.5 if a.max() <= 1 else 127)).astype(np.float32))
    output = []
    height, width = resolution
    for frame in a:
        binary = frame > (0.5 if frame.max() <= 1 else 127)
        rows = np.flatnonzero(binary.any(1))
        if not len(rows):
            output.append(np.zeros((height, width), np.float32))
            continue
        crop = binary[rows[0] : rows[-1] + 1]
        target_width = max(1, round(crop.shape[1] * height / crop.shape[0]))
        resized = (
            np.asarray(
                Image.fromarray(crop.astype(np.uint8) * 255).resize(
                    (target_width, height), Image.Resampling.NEAREST
                )
            )
            > 0
        )
        mass = resized.sum(0)
        center = int(np.searchsorted(np.cumsum(mass), mass.sum() / 2))
        left = center - width // 2
        out = np.zeros((height, width), np.float32)
        lo, hi = max(0, left), min(target_width, left + width)
        if hi > lo:
            out[:, lo - left : hi - left] = resized[:, lo:hi]
        output.append(out)
    return torch.from_numpy(np.stack(output))


def read_sequence(path, resolution=(64, 44)):
    p = Path(path)
    if p.is_dir():
        files = sorted(
            q for q in p.iterdir() if q.suffix.lower() in (".png", ".bmp", ".jpg", ".jpeg")
        )
        array = np.stack([np.asarray(Image.open(q).convert("L")) for q in files])
    elif p.suffix.lower() == ".pkl":
        with p.open("rb") as f:
            array = pickle.load(f)
    elif p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as data:
            if "silhouettes" in data:
                array = data["silhouettes"]
            elif len(data.files) == 1:
                array = data[data.files[0]]
            else:
                raise ValueError("NumPy archives with multiple arrays require a silhouettes key.")
    else:
        array = np.load(p, allow_pickle=False)
    return normalize_frames(array, resolution)


class GaitDataset(Dataset):
    def __init__(self, records, frames=30, training=False, dual_view=False):
        self.records = list(records)
        self.frames, self.training, self.dual_view = frames, training, dual_view
        self.labels = {s: i for i, s in enumerate(sorted({r.subject for r in records}))}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        source = read_sequence(record.path)
        if self.dual_view:
            x, second = augment(source, self.frames), augment(source, self.frames)
        else:
            x = sample_frames(source, self.frames, self.training)
            second = x
        return {
            "x": x,
            "view2": second,
            "label": self.labels[record.subject],
            "record_index": index,
        }
