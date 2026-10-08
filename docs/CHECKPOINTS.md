# Checkpoints

CASIA-B evaluation requires trained recognition weights compatible with the configured SafeGait-ER architecture. Prepare one checkpoint per experiment seed: 2024, 2025, and 2026.

## Recognition checkpoint contract

| Entry | Contents |
| --- | --- |
| `model` | Complete model state dictionary |
| `config` | Configuration used by the originating training run |
| `training.seed` | Training seed, equal to `config.seed` |
| `training.completed_epochs` | Positive number of completed training epochs |
| `format_version` | Export format version, currently 1 |
| `provenance.source_sha256` | SHA-256 checksum of the source training checkpoint |

For source training archives, `epoch` may record completed epochs, and `config.seed` may record the training seed. The exporter preserves those recorded values. Training records accompany the source checkpoint.

The configured model has 30 frames, resolution 64 by 44, 128 channels, eight spatial strips, a 256-dimensional embedding, and 74 training identities. Hierarchical checkpoints include learned `backbone.sgra.scale` and `backbone.sgra.bias` parameters. Loading validates all parameter names, dimensions, finite values, and the recorded training seed. All inference model parameters are frozen; feature gradients remain available for identity attribution.

## Export

Run on an NVIDIA CUDA GPU from the repository root:

```bash
python -m scripts.export_checkpoint --source path/to/trained_model.pt --output checkpoints/seed2025.pt --device cuda
```

The command requires an existing trained source checkpoint and creates an inference checkpoint plus `seed2025.pt.json`, which records the exported checksum, architecture, training metadata, and source checksum. The destination must be a new file.

An author-controlled full training archive may include serialized NumPy or Python random-state objects. For this format, use:

```bash
python -m scripts.export_checkpoint --source path/to/training_archive.pt --output checkpoints/seed2025.pt --device cuda --trusted-training-source
```

The trusted-source option enables Python object deserialization for the source archive. Use it only for an archive whose origin you control. Exported inference checkpoints use tensor and primitive metadata entries and are loaded with restricted deserialization.

Prepare seeds 2024 and 2026 from their corresponding trained source archives. Weight files remain in the local `checkpoints` directory and are excluded from Git.

## Use

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint checkpoints/seed2025.pt --device cuda --output outputs/release.json
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint checkpoints/seed2025.pt --seed 2025 --device cuda --output outputs/casia_b/seed2025/observations.jsonl
```

See [CASIA-B evaluation](CASIA_B_REPRODUCTION.md) for result computation and membership inference.

## Membership attacker checkpoints

Membership evaluation saves a separate trained MLP checkpoint for each channel and protocol in `attack_checkpoints` next to the result file. Each file contains the attacker state dictionary, input dimension, hidden widths, channels, seed, completed fitting epochs, protocol, evaluation metrics, and originating observation metadata. Use `--checkpoint-dir` to select another artifact directory.

Recognition model checkpoints and membership attacker checkpoints have separate contracts.
