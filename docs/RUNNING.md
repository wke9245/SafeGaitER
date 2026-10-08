# Method pipeline

Run commands from the repository root with the Python environment activated.

## Input

The input must be an actual silhouette sequence:

- A NumPy array in `.npy` format.
- A `.npz` archive with a `silhouettes` array, or a single array.
- A silhouette `.pkl` file.
- A directory of PNG, BMP, JPEG, or JPG frames, sorted by filename.

Sequence arrays use shape `[frames, height, width]`. Single-channel sequence arrays are also accepted. Frames are normalized to 64 by 44 and sampled deterministically to 30 frames.

## Execution

```bash
python -m scripts.pipeline --input path/to/sequence.npy
```

Without a checkpoint, the network uses untrained parameters. This mode exposes the computation order and emits a notice on standard error. Its reliability scores, risk levels, and diagnostic categories are not trained model predictions.

To supply learned weights:

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/checkpoint.pt
```

Checkpoints must contain `model` and `config` entries. Weights are loaded strictly. The hierarchical encoder requires the `backbone.sgra.scale` and `backbone.sgra.bias` parameters. The checkpoint configuration is used unless `--config` explicitly selects another compatible configuration.

The entry point uses classifier confidence and contribution from the predicted identity logit. It does not construct a recognition gallery. The model's gallery-matching interface remains available for integration.

## Options

| Option | Meaning |
| --- | --- |
| `--input` | Required silhouette file or image directory |
| `--checkpoint` | Compatible trained model checkpoint |
| `--config` | Explicit YAML configuration |
| `--device cpu` | CPU execution, the default |
| `--device cuda` | CUDA execution |
| `--device auto` | CUDA when available, otherwise CPU |
| `--output` | Optional output JSON file |

Input and output arguments resolve from the current working directory. Relative configuration and checkpoint paths resolve from the repository root.

## Output

Standard output contains only the permitted explanation packet. The optional JSON file contains the same packet.

| Level | Released fields |
| --- | --- |
| Low | `level`, `spatial_scores`, `temporal_scores` |
| Medium | `level`, `dominant_region`, `dominant_interval`, `diagnostic` |
| High | `level`, `diagnostic` |

The entry point does not export internal risk values, identity contributions, key-frame indices, or embeddings. It does not perform training, attacker fitting, or table generation.
