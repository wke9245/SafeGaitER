# Running experiments

Run all commands from the repository root with the environment activated.

## Configuration and paths

The benchmark configurations are:

- `configs/safegait_er/casia_b.yaml`
- `configs/safegait_er/ou_mvlp.yaml`

Paths in configurations are resolved from the repository root. Absolute dataset paths are supported:

```bash
python run.py train --data-root /datasets/CASIA-B --auxiliary-root /datasets/OU-MVLP
```

The `auto` device uses CUDA when available and otherwise uses CPU. Use `--device cpu` or `--device cuda` to choose explicitly.

Without an explicit configuration, evaluation reads the configuration stored in the supplied checkpoint or attack archive. The release strategy defaults to the strategy recorded in the recognition checkpoint. An explicit `recognition --strategy` can evaluate another observable interface on the same frozen recognizer.

## Training

CASIA-B uses 120 epochs, consisting of 60 recognition epochs and 60 joint epochs. OU-MVLP uses 100 epochs, split into two stages of 50 epochs each. The complete auxiliary pool is required for adversarial optimization.

Each epoch writes a checkpoint containing the model, auditor-side modules, optimizers, and random states. Resume at the next epoch with:

```bash
python run.py train --resume outputs/casia_b/checkpoint.pt
```

Release-strategy comparisons use `--strategy` with `raw`, `clipping`, `noise`, `masking`, `coarsening`, `generic`, `deepleak`, `no_auditor`, or `safe`. The frame feature extractor is selected with `--backbone hierarchical` or `--backbone gaitpart`.

## Recognition and explanation release

```bash
python run.py recognition --checkpoint outputs/casia_b/checkpoint.pt --queries 20
```

Recognition uses subject-level gallery similarity, gallery self-match exclusion, and probe-level Rank-1 aggregation. Explanation inference includes local feature-gradient attribution and one stability-predictor forward pass.

The output files are:

| File | Contents |
| --- | --- |
| `released_packets.jsonl` | Fields visible through the configured explanation interface |
| `recognition.json` | Recognition outcomes and packet-level counts |
| `attack_inputs.pt` | Local inputs for independent attacker fitting |
| `training_manifest.json` | Recognition-training and auxiliary pool counts |
| `training.csv` | Per-epoch objectives |
| `checkpoint.pt` | Recognition and training-state checkpoint |

The local attack archive also contains recognition embeddings and internal evaluation state. It is separate from the recipient-visible explanation packet file.

## Privacy evaluation

```bash
python run.py privacy --protocol matched --repeated
python run.py privacy --protocol stress
```

The descriptor and native-field attacks are fitted independently. Score-only, packet-only, and score-plus-packet interfaces are evaluated separately. AUC and TPR at 1% FPR use the per-seed best-of rule. Linkage includes ROC AUC and EER. The closed-set identity probe uses disjoint fitting and evaluation sequences.

Repeated-release experiments require recognition exports with at least the largest configured query budget. Perturbation is resampled independently for each release.

## Auditability

```bash
python run.py audit
```

CASIA-B auditability evaluation reports natural-sequence coverage and conditional agreement separately from controlled disturbance outcomes. Category labels are used only when computing evaluation agreement. They do not enter the diagnostic mapper.

## Reconstruction

```bash
python run.py inversion
python run.py inversion --raw
python run.py inversion --reid-checkpoint outputs/reid_777/checkpoint.pt
```

The decoder reconstructs all 30 frames. SSIM and IoU are computed over complete sequences. Reconstructed Re-ID requires the independent seed-777 raw recognizer.

LPIPS is enabled when its optional dependencies are installed:

```bash
python -m pip install -r requirements-metrics.txt
```

LPIPS uses the official AlexNet perceptual network. Its initial execution obtains the pretrained network weights. The result file records whether the metric was available.

## Profiling and tables

```bash
python run.py profile --repetitions 100
python run.py tables --results-root outputs
```

Profiling uses a batch size of one, warm-up iterations, device synchronization, and the complete configured explanation path. Table aggregation exports CSV and JSON summaries from measured experiment outputs.
