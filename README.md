# SafeGait-ER

PyTorch implementation of **SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition**.

Submitted to *Applied Soft Computing Journal*, manuscript **ASOC-D-26-11494**.

The repository provides the principal SafeGait-ER components and CASIA-B evaluation with explicitly defined dataset and membership-attack partitions. It includes subject-level gait recognition, stability prediction, structured privacy risk, risk-adaptive protection, and three-level explanation release. Evaluation uses a frozen trained recognition model and independently fitted post-hoc attackers.

## Architecture

1. Extract hierarchical frame features with global and local convolutional branches.
2. Refine features using activation-guided reliability at 1, 2, and 4 horizontal regions.
3. Construct separate eight-region evidence and aggregate frame importance.
4. Estimate identity contribution, single-view stability, and structured privacy risk.
5. Apply clipping, masking, coarsening, and Gaussian perturbation.
6. Release the fields permitted by the low, medium, or high risk policy.

The hierarchical refinement follows the reliability-weighting principles of IHGR-Net. The eight-region evidence stage supplies the fine-grained factors used by SafeGait-ER.

## GPU environment and installation

Dependencies are pinned to stable releases verified on 2026-10-08: PyTorch 2.14.1, NumPy 2.5.3, PyYAML 6.0.3, and Pillow 12.3.0. PyTorch uses the official CUDA 13.2 GPU build.

| Component | Recommended configuration |
| --- | --- |
| Operating system | Windows 11 or Linux x86-64 |
| NVIDIA GPU | RTX 4090, 24 GB VRAM; or RTX 5090, 32 GB VRAM |
| Processor | 12 or more CPU cores |
| System memory | 64 GB RAM |
| Storage | 1 TB NVMe SSD |
| Python | 3.14; dependency minimum 3.12 |
| Framework | PyTorch 2.14.1 with CUDA 13.2 |
| GPU driver | Current NVIDIA driver compatible with CUDA 13.2 |

On Windows, install Python 3.14 with its launcher, then run:

```powershell
powershell -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

On Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file explicitly selects the CUDA 13.2 wheel. Model computation requires an accessible NVIDIA GPU. The PyTorch distribution provides the CUDA runtime dependencies.

Official sources: [PyTorch 2.14.1 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.1), [installation guide](https://pytorch.org/get-started/locally/), and [CUDA 13.2 wheels](https://download.pytorch.org/whl/cu132/torch/).

## Input and method pipeline

Provide an actual silhouette sequence as a NumPy array, silhouette pickle, or image directory. The reader normalizes silhouettes to 64 by 44 and samples 30 frames.

Run the pipeline from the repository root with a compatible trained checkpoint:

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/checkpoint.pt --device cuda --output outputs/release.json
```

The single-sequence entry point uses identity-classifier confidence and requires a trained checkpoint. The CASIA-B evaluation entry point uses subject-level gallery confidence and seed-specific frozen weights.

## CASIA-B evaluation

The evaluation package defines the recognition, identity-matched membership, and identity-disjoint stress partitions. It computes NM/BG/CL Rank-1, independent membership AUC and TPR at 1% FPR, release-level proportions, and three-seed statistics.

```bash
python -m scripts.casia_b_index --data-root data/CASIA-B --manifest outputs/casia_b/manifest.csv
python -m scripts.casia_b_partitions --manifest outputs/casia_b/manifest.csv --seeds 2024 2025 2026
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint checkpoints/seed2025.pt --seed 2025 --device cuda --output outputs/casia_b/seed2025/observations.jsonl
python -m scripts.casia_b_results --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --device cuda --output outputs/casia_b/seed2025/principal_results.json
python -m scripts.casia_b_mia --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --device cuda --output outputs/casia_b/seed2025/membership.json
```

Numerical evaluation requires licensed CASIA-B silhouettes and the corresponding trained checkpoints. See the [complete CASIA-B protocol and run order](docs/CASIA_B_REPRODUCTION.md) and [checkpoint preparation](docs/CHECKPOINTS.md).

## Repository scope

| Path | Contents |
| --- | --- |
| `opengait/modeling/models` | Encoding, evidence, stability, risk, protection, release, and auditor components |
| `opengait/modeling/losses` | Recognition and privacy-aware objective functions |
| `opengait/data` | Silhouette reading, sampling, augmentation, and exact CASIA-B protocols |
| `opengait/evaluation/retrieval.py` | Subject-level gallery matching interface |
| `scripts/pipeline.py` | Single-sequence explanation release |
| `scripts/casia_b_*.py` | Indexing, partitions, frozen inference, metrics, and three-seed aggregation |
| `attacks/descriptors.py` | 36-dimensional observable-channel descriptor |
| `opengait/checkpoint.py`, `scripts/export_checkpoint.py` | Frozen-model validation and checkpoint export |
| `configs/safegait_er/casia_b.yaml` | Model, protection, and objective parameters |

Soft routing, privacy auditors, and objective functions are included as training components. This evaluation package uses externally supplied trained recognition checkpoints. Full recognition training and supplementary experiments are managed separately.

See [method details](docs/METHOD.md), [pipeline instructions](docs/RUNNING.md), and [third-party references](THIRD_PARTY_NOTICES.md).

## References

SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition. Submitted to Applied Soft Computing Journal.

IHGR-Net: Towards Trustworthy Gait Recognition: Visualizing and Refining Hierarchical Motion Dependencies. *Complex & Intelligent Systems*, 2026. https://doi.org/10.1007/s40747-026-02381-6

CASIA-B and OU-MVLP are subject to their respective access and distribution policies. Dataset silhouettes and pretrained recognition weights are not redistributed.
