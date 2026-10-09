# SafeGait-ER

PyTorch components for **SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition**.

Submitted to *Applied Soft Computing Journal*, manuscript **ASOC-D-26-11494**.

The source includes hierarchical gait evidence, identity attribution, stability prediction, structured privacy risk, explanation protection and release, CASIA-B partitions, and independent membership auditing.

## Installation

Requires Python 3.12 or newer, CUDA-enabled PyTorch, and an NVIDIA GPU. Dependencies are pinned in `requirements.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `powershell -File setup.ps1`, then activate `.venv`.

## Usage

Inputs are silhouette sequences sampled to 30 frames at 64 by 44 resolution. Supported formats include NumPy arrays, NPZ silhouettes, silhouette pickle files, and image directories.

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/model.pt --device cuda --output outputs/release.json
```

CASIA-B entry points: `casia_b_index`, `casia_b_partitions`, `casia_b_inference`, `casia_b_results`, `casia_b_mia`, and `casia_b_summary`. Run `python -m scripts.<command> --help` for arguments. Evaluation uses the subject and condition partitions defined in `opengait/data/protocols.py`.

## Checkpoint

Download `safegait_er_casia_b_seed2025.pt` from [Releases](https://github.com/wke9245/SafeGaitER/releases) into `checkpoints/`. The archive retains model, optimizer, scheduler, configuration, and training metadata. Its saved architecture is `hierarchical_full`, with 512 feature channels and a 256-dimensional embedding.

```bash
python -m scripts.inspect_checkpoint --checkpoint checkpoints/safegait_er_casia_b_seed2025.pt --device cuda
```

Forward inference requires a model implementation and settings matching the checkpoint configuration.

See [third-party references](THIRD_PARTY_NOTICES.md) for algorithmic and engineering sources.
