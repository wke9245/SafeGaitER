# SafeGait-ER

PyTorch implementation of **SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition**.

The repository provides gait evidence encoding, explanation stability prediction, identity-sensitive risk estimation, risk-adaptive protection, three-level explanation release, adversarial privacy learning, and independent privacy evaluation.

## Installation

Python 3.9 or later is required. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For CPU-only PyTorch, install the CPU wheel before the other dependencies:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows users can run `powershell -File setup.ps1`. CUDA 12.4 wheels are available through `powershell -File setup.ps1 -TorchIndex cu124`.

## Data preparation

Provide the CASIA-B and OU-MVLP silhouette datasets in the following layout:

```text
data/
  CASIA-B/
    001/nm-01/000/silhouettes.pkl
  OU-MVLP/
    00001/00/000/silhouettes.pkl
```

The reader accepts silhouette pickle files, NumPy sequence files, and image folders. Dataset access and distribution follow the respective benchmark terms. The OU-MVLP subject partition is included in `configs/splits/OUMVLP.json`.

See [data protocols](docs/DATA_PROTOCOLS.md) for recognition, auxiliary, and attacker partitions.

## CASIA-B experiments

```bash
python run.py train --config configs/safegait_er/casia_b.yaml
python run.py recognition --queries 20
python run.py privacy --protocol matched --repeated
python run.py privacy --protocol stress
python run.py audit
python run.py profile
python run.py inversion
python run.py inversion --raw
python run.py tables --results-root outputs
```

Reconstructed Re-ID uses an independent recognition model:

```bash
python run.py train --strategy raw --seed 777 --output outputs/reid_777
python run.py inversion --reid-checkpoint outputs/reid_777/checkpoint.pt
```

## OU-MVLP experiments

Use `--config configs/safegait_er/ou_mvlp.yaml` for training and evaluation. Membership evaluation uses `--protocol stress`.

```bash
python run.py train --config configs/safegait_er/ou_mvlp.yaml
python run.py recognition --config configs/safegait_er/ou_mvlp.yaml --queries 20
python run.py privacy --config configs/safegait_er/ou_mvlp.yaml --protocol stress --repeated
```

## Experiment suites

```bash
python run.py sweep --suite main
python run.py sweep --suite factors
python run.py sweep --suite thresholds
python run.py sweep --suite backbones
python run.py sweep --suite transfer
python run.py sweep --suite all --plan-only
```

The suites use seeds 2024, 2025, and 2026. Attack models are fitted independently for each interface and release budget. Summaries compute the per-seed best-of envelope before the mean and sample standard deviation.

## Repository structure

| Directory | Contents |
| --- | --- |
| `opengait/data` | Silhouette preprocessing, sampling, augmentation, and protocols |
| `opengait/modeling/models` | Recognition, risk, protection, release, and privacy auditors |
| `opengait/modeling/losses` | Recognition and privacy objectives |
| `opengait/evaluation` | Retrieval and evaluation metrics |
| `attacks` | Membership, linkage, identity, inversion, and repeated-release attacks |
| `baselines` | Explanation protection controls |
| `configs` | Benchmark experiment configurations and identity partitions |
| `scripts` | Training, evaluation, profiling, and aggregation |

Additional documentation: [running experiments](docs/RUNNING.md), [method and parameters](docs/METHOD.md), and [third-party references](THIRD_PARTY_NOTICES.md).

Datasets, model checkpoints, and generated experiment outputs are stored locally and are excluded from version control.
