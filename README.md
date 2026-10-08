# SafeGait-ER

Method-oriented PyTorch implementation of **SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition**.

Submitted to *Applied Soft Computing Journal*, manuscript **ASOC-D-26-11494**.

The repository contains the principal algorithmic components and a single-sequence explanation pipeline. It covers hierarchical gait encoding, spatial reliability, temporal importance, identity contribution, stability prediction, structured privacy risk, risk-adaptive protection, and three-level explanation release.

## Architecture

1. Extract hierarchical frame features with global and local convolutional branches.
2. Refine features using activation-guided reliability at 1, 2, and 4 horizontal regions.
3. Construct separate eight-region evidence and aggregate frame importance.
4. Estimate identity contribution, single-view stability, and structured privacy risk.
5. Apply clipping, masking, coarsening, and Gaussian perturbation.
6. Release the fields permitted by the low, medium, or high risk policy.

The hierarchical refinement follows the reliability-weighting principles of IHGR-Net. The eight-region evidence stage supplies the fine-grained factors used by SafeGait-ER.

## Installation

Python 3.9 or later is required.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Windows, `powershell -File setup.ps1` creates a CPU environment. CUDA 12.4 wheels can be selected with `powershell -File setup.ps1 -TorchIndex cu124`.

## Input and method pipeline

Provide an actual silhouette sequence as a NumPy array, silhouette pickle, or image directory. The reader normalizes silhouettes to 64 by 44 and samples 30 frames.

Run the pipeline from the repository root:

```bash
python -m scripts.pipeline --input path/to/sequence.npy
```

Use a compatible trained checkpoint and optionally save the permitted release packet:

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/checkpoint.pt --output outputs/release.json
```

The default pipeline uses the identity-classifier confidence path. No gallery is supplied by this entry point. Without a checkpoint, parameters are randomly initialized and numerical outputs describe the computational structure only; they are not trained predictions.

## Repository scope

| Path | Contents |
| --- | --- |
| `opengait/modeling/models` | Encoding, evidence, stability, risk, protection, release, and auditor components |
| `opengait/modeling/losses` | Recognition and privacy-aware objective functions |
| `opengait/data` | Silhouette reading, sampling, and augmentation |
| `opengait/evaluation/retrieval.py` | Subject-level gallery matching interface |
| `scripts/pipeline.py` | Single-sequence method entry point |
| `configs/safegait_er/casia_b.yaml` | Model, protection, and objective parameters |

Soft routing, privacy auditors, and objective functions are retained as independent training-time components. The pipeline does not fit these modules or generate benchmark result tables. Training weights, full experimental procedures, and datasets are managed separately.

See [method details](docs/METHOD.md), [pipeline instructions](docs/RUNNING.md), and [third-party references](THIRD_PARTY_NOTICES.md).

## References

SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition. Submitted to Applied Soft Computing Journal.

IHGR-Net: Towards Trustworthy Gait Recognition: Visualizing and Refining Hierarchical Motion Dependencies. *Complex & Intelligent Systems*, 2026. https://doi.org/10.1007/s40747-026-02381-6

CASIA-B and OU-MVLP are subject to their respective access and distribution policies. Dataset silhouettes and pretrained recognition weights are not redistributed.
