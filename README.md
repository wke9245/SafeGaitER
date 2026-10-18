# SafeGait-ER

PyTorch components for **SafeGait-ER: From Faithful Explanations to Privacy-Aware Release in Gait Recognition**.

Submitted to *Applied Soft Computing Journal*, manuscript **ASOC-D-26-11494**.

The source includes hierarchical gait evidence, identity attribution, stability prediction, structured privacy risk, explanation protection and release, CASIA-B partitions, and independent membership auditing. Explanations are represented over 30 frames and eight spatial strips, then protected and released according to the estimated privacy risk.

## Installation

Requires Python 3.12 or newer, CUDA-enabled PyTorch, and an NVIDIA GPU. Dependencies are specified in `requirements.txt`. The independent triage auditor uses scikit-learn's deterministic `lbfgs` solver; recognizer inference and privacy-attack training use CUDA.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows:

```powershell
powershell -File setup.ps1
.\.venv\Scripts\Activate.ps1
```

Run all commands from the repository root with the environment activated. Use `--device cuda:0` to select a particular GPU.

## Usage

Inputs are silhouette sequences sampled to 30 frames at 64 by 44 resolution. Supported formats include NumPy arrays, NPZ silhouettes, silhouette pickle files, and image directories. Sequence arrays use `[frames, height, width]`; multi-array NPZ files require a `silhouettes` key. Image filenames must sort in frame order.

For a single sequence, supply a checkpoint whose model definition and saved configuration match the implementation:

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/model.pt --device cuda --output outputs/release.json
```

The command writes a risk-conditioned explanation packet to JSON. Its `--config` option accepts a YAML configuration matching the saved checkpoint configuration.

## CASIA-B evaluation

Organize the licensed silhouettes as `data/CASIA-B/<subject>/<condition>/<view>/`. Each sequence directory contains either one array file or its image frames. The index expects all 13,640 sequences: 124 identities, 11 views, and ten sequence conditions.

Run indexing and partition export first:

```bash
python -m scripts.casia_b_index --data-root data/CASIA-B --manifest outputs/casia_b/manifest.csv
python -m scripts.casia_b_partitions --manifest outputs/casia_b/manifest.csv --seeds 2024 2025 2026 --output outputs/casia_b/partitions
```

Next, use a compatible seed-2025 checkpoint for inference, recognition statistics, and independent membership inference. Replace `path/to/casia_b_seed2025.pt` with its location:

```bash
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint path/to/casia_b_seed2025.pt --seed 2025 --device cuda --output outputs/casia_b/seed2025/observations.jsonl
python -m scripts.casia_b_results --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --device cuda --output outputs/casia_b/seed2025/principal_results.json
python -m scripts.casia_b_mia --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --protocol identity_matched --device cuda --output outputs/casia_b/seed2025/membership_matched.json --checkpoint-dir outputs/casia_b/seed2025/attackers_matched
```

Recognition uses identities 075-124 with 2,200 gallery and 3,300 probe sequences. Release statistics use all 5,500 sequences from these identities. Membership inference fits separate attackers for score, raw explanation, and protected explanation inputs.

See [CASIA-B evaluation](docs/CASIA_B_REPRODUCTION.md) for exact partitions, the identity-disjoint protocol, output files, and three-seed aggregation. Each seed requires its corresponding model checkpoint and inference artifacts. All entry points provide `--help`.

## Independent audit and repeated releases

The independent recognition-error audit uses subjects 075-099 for fitting and 100-124 for evaluation, with 1,650 recognition probes in each partition. Five independently fitted logistic-regression auditors compare retrieval confidence, release level, raw explanations, and the protected packet. Feature scaling is fitted exclusively on the audit-fit partition.

```bash
python -m scripts.casia_b_audit --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --output outputs/casia_b/seed2025/triage.json --model-dir outputs/casia_b/seed2025/audit_models
```

For repeated-release evaluation, collect 200 stochastic packets per source sequence under a frozen recognizer and release policy, then fit independent membership and native-field linkage attackers at each query budget:

```bash
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint path/to/casia_b_seed2025.pt --seed 2025 --releases 200 --device cuda --output outputs/casia_b/seed2025/repeated_observations.jsonl
python -m scripts.casia_b_repeated --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/repeated_observations.jsonl --seed 2025 --device cuda --output outputs/casia_b/seed2025/repeated.json --checkpoint-dir outputs/casia_b/seed2025/repeated_attackers
```

Query budgets are 1, 5, 10, 20, 50, 100, and 200. Repeated packets are pooled within each source sequence. The AUC-selected membership attack also supplies the interpolated TPR at 1% FPR. See [independent audit and query budgets](docs/AUDIT_AND_QUERY_BUDGETS.md) for exact settings, saved artifacts, and CSV exports of Tables 11, 12, and D.11.

## Checkpoint

Download `safegait_er_casia_b_seed2025.pt`, its JSON metadata, and `SHA256SUMS.txt` from [Releases](https://github.com/wke9245/SafeGaitER/releases/tag/v1.0.0) into `checkpoints/`. The checkpoint contains model weights, configuration, training metadata, optimizer state, and scheduler state.

```bash
python -m scripts.inspect_checkpoint --checkpoint checkpoints/safegait_er_casia_b_seed2025.pt --device cuda --output outputs/checkpoint_inspection.json
```

The inspection report records the file checksum, tensor checks, optimizer checks, and model binding. Forward inference requires a model implementation and settings matching the checkpoint configuration; review `architecture_binding` before evaluation. See [checkpoint usage](docs/CHECKPOINTS.md) for the file contract and optional sequence inspection.

See [third-party references](THIRD_PARTY_NOTICES.md) for algorithmic and engineering sources.
