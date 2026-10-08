# GPU method pipeline

Run commands from the repository root with the Python environment activated.

## GPU environment

The installation uses PyTorch 2.14.1 with the official CUDA 13.2 GPU wheels. Python 3.12 or newer is required; Python 3.14 is recommended.

Use a Windows 11 or Linux x86-64 workstation with an NVIDIA CUDA GPU and a current driver compatible with CUDA 13.2. The recommended workstation has an RTX 4090 with 24 GB VRAM or an RTX 5090 with 32 GB VRAM, at least 12 CPU cores, 64 GB system RAM, and 1 TB NVMe storage.

On Windows:

```powershell
powershell -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

The setup script creates a Python 3.14 environment through the Windows Python launcher, installs the CUDA dependencies, and checks GPU availability. Use `-PythonVersion 3.12`, `3.13`, or `3.15` when selecting another supported installation.

On Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file selects the CUDA 13.2 build explicitly. The NVIDIA driver is required. The PyTorch distribution supplies its CUDA runtime dependencies; a separate CUDA Toolkit installation is not required for this pipeline.

## Input

The input must be an actual silhouette sequence:

- A NumPy array in `.npy` format.
- A `.npz` archive with a `silhouettes` array, or a single array.
- A silhouette `.pkl` file.
- A directory of PNG, BMP, JPEG, or JPG frames, sorted by filename.

Sequence arrays use shape `[frames, height, width]`. Single-channel sequence arrays are also accepted. Frames are normalized to 64 by 44 and sampled deterministically to 30 frames.

## Execution

```bash
python -m scripts.pipeline --input path/to/sequence.npy --device cuda
```

Model inference, gradient attribution, risk estimation, and explanation protection execute on the selected NVIDIA GPU. The entry point stops with an error if CUDA is unavailable.

Without a checkpoint, the network uses untrained parameters. Its reliability scores, risk levels, and diagnostic categories are not trained model predictions.

To supply learned weights:

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/checkpoint.pt --device cuda
```

Checkpoints must contain `model` and `config` entries. Weights are loaded strictly. The hierarchical encoder requires the `backbone.sgra.scale` and `backbone.sgra.bias` parameters. The checkpoint configuration is used unless `--config` explicitly selects another compatible configuration.

The entry point uses classifier confidence and contribution from the predicted identity logit. It does not construct a recognition gallery. The model's gallery-matching interface remains available for integration.

## Options

| Option | Meaning |
| --- | --- |
| `--input` | Required silhouette file or image directory |
| `--checkpoint` | Compatible trained model checkpoint |
| `--config` | Explicit YAML configuration |
| `--device cuda` | Default NVIDIA GPU |
| `--device cuda:1` | Second visible NVIDIA GPU |
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
