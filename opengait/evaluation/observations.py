"""Read and validate seed-specific frozen CASIA-B inference artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from opengait.checkpoint import sha256_file
from opengait.modeling.models.release_policy import validate_packet


def manifest_digest(records):
    keys = "\n".join(sorted(record.key for record in records)) + "\n"
    return hashlib.sha256(keys.encode("utf-8")).hexdigest()


def metadata_path(path):
    return Path(str(path) + ".metadata.json")


def read_metadata(path, records, seed=None):
    path = Path(path)
    metadata = json.loads(metadata_path(path).read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("dataset") != "CASIA-B"
        or type(metadata.get("seed")) is not int
        or metadata.get("manifest_key_sha256") != manifest_digest(records)
        or metadata.get("observations_sha256") != sha256_file(path.resolve())
    ):
        raise ValueError("Inference artifact metadata or checksum is invalid.")
    if seed is not None and seed != metadata["seed"]:
        raise ValueError("Attack seed does not match the frozen inference seed.")
    if metadata.get("training", {}).get("seed") != metadata["seed"]:
        raise ValueError("Inference and model training seeds must agree.")
    count = metadata.get("release_count", 1)
    if type(count) is not int or not 1 <= count <= 200:
        raise ValueError("Release count must be an integer between 1 and 200.")
    return metadata


def iter_observations(path, records, metadata):
    expected = {record.key: record for record in records}
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry.get("key")
            if key not in expected or key in seen:
                raise ValueError("Inference outputs contain an unexpected or duplicated key.")
            validate_observation(entry, expected[key], metadata.get("release_count", 1))
            seen.add(key)
            yield entry
    if seen != set(expected) or metadata.get("sequences") != len(seen):
        raise ValueError("Inference outputs must contain every manifest sequence exactly once.")


def validate_observation(entry, record, release_count=1):
    if (entry.get("subject"), entry.get("condition"), entry.get("view")) != (
        record.subject,
        record.condition,
        record.view,
    ):
        raise ValueError(f"Inference sequence labels disagree with the manifest: {record.key}")
    embedding = np.asarray(entry["embedding"], dtype=np.float32)
    if embedding.shape != (256,) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid embedding: {record.key}")
    if not 0 <= entry["score"] <= 1 or not 0 <= entry["risk"] <= 1:
        raise ValueError(f"Invalid score or risk: {record.key}")
    predicted = entry.get("predicted_subject")
    candidates = range(1, 75) if int(record.subject) <= 74 else range(75, 125)
    if predicted not in {f"{subject:03d}" for subject in candidates}:
        raise ValueError(f"Invalid predicted identity: {record.key}")
    raw = entry["raw_packet"]
    for name, shape in (
        ("omega", (30, 8)),
        ("alpha", (30,)),
        ("contribution", (30, 8)),
        ("keyframes", (30,)),
    ):
        values = np.asarray(raw[name], dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid raw explanation field: {name}")
    validate_packet(entry["safe_packet"])
    packets = entry.get("release_packets", [entry["safe_packet"]])
    if not isinstance(packets, list) or len(packets) != release_count:
        raise ValueError(f"Release count disagrees with metadata: {record.key}")
    if packets[0] != entry["safe_packet"]:
        raise ValueError("The primary packet must equal the first repeated release.")
    for packet in packets:
        validate_packet(packet)
        if packet["level"] != entry["safe_packet"]["level"]:
            raise ValueError("Repeated releases must retain the fixed risk level.")


def read_observations(path, records, seed=None):
    metadata = read_metadata(path, records, seed)
    return {entry["key"]: entry for entry in iter_observations(path, records, metadata)}, metadata
