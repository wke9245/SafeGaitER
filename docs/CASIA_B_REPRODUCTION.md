# CASIA-B Evaluation

Run the commands from the repository root in the CUDA environment described in the [README](../README.md). Evaluation requires licensed CASIA-B silhouettes and a checkpoint whose network definition, tensor dimensions, and method settings match the implementation. The inference seed must equal the saved training seed.

## Data and manifest

Use the directory layout `data/CASIA-B/<subject>/<condition>/<view>/`, for example `data/CASIA-B/001/nm-01/000/`. Each directory contains one `.npy`, `.npz`, or `.pkl` sequence file, or a sequence of `.png`, `.bmp`, `.jpg`, or `.jpeg` frames. Array files and image frames must be stored in separate sequence directories. Image filenames must sort in temporal order.

Arrays represent `[frames, height, width]`. NPZ archives may contain a single array or a named `silhouettes` array. The reader normalizes silhouettes to 64 by 44 pixels, and inference deterministically samples 30 frames.

```bash
python -m scripts.casia_b_index --data-root data/CASIA-B --manifest outputs/casia_b/manifest.csv
python -m scripts.casia_b_partitions --manifest outputs/casia_b/manifest.csv --seeds 2024 2025 2026 --output outputs/casia_b/partitions
```

The manifest has `subject,condition,view,path` columns and one row per sequence. Subjects are `001` through `124`; conditions are `nm-01` through `nm-06`, `bg-01`, `bg-02`, `cl-01`, and `cl-02`; views are `000,018,036,054,072,090,108,126,144,162,180`. Relative paths are resolved against the manifest directory. The index validates all 13,640 unique subject/condition/view keys.

## Recognition partitions

| Role | Identities | Conditions at all 11 views | Sequences |
| --- | --- | --- | ---: |
| Recognition training members | 001-074 | nm-01 through nm-04, bg-01, cl-01 | 4,884 |
| Reserved nonmembers | 001-074 | nm-05, nm-06, bg-02, cl-02 | 3,256 |
| Recognition gallery | 075-124 | nm-01 through nm-04 | 2,200 |
| Recognition probes | 075-124 | nm-05, nm-06, bg-01, bg-02, cl-01, cl-02 | 3,300 |

Gallery retrieval uses normalized embeddings and averages cosine similarity over the gallery sequences of each candidate identity. All gallery views are included. Rank-1 accuracy is reported separately for NM, BG, and CL; the average is the arithmetic mean of these three percentages. Each held-out identity has 44 gallery sequences.

Low, medium, and high release counts and fractions use the 5,500 held-out gallery and probe sequences as their denominator. The report also includes field coverage and mean privacy risk.

## Inference and recognition results

Replace `path/to/casia_b_seed2025.pt` with the compatible checkpoint for seed 2025:

```bash
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint path/to/casia_b_seed2025.pt --seed 2025 --device cuda --output outputs/casia_b/seed2025/observations.jsonl
python -m scripts.casia_b_results --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --device cuda --output outputs/casia_b/seed2025/principal_results.json
```

Inference builds separate galleries for identities 001-074 and 075-124 and excludes exact sequence self-matches. It writes one observation for every manifest sequence, including the embedding, retrieval confidence, privacy risk, predicted identity, raw explanation, and protected release packet.

Keep `observations.jsonl` together with `observations.jsonl.metadata.json`. The metadata records the seed, training metadata, checkpoint checksum, manifest-key checksum, observation checksum, sequence count, and execution environment. Result and attack readers validate these artifacts before computing metrics. Use a new inference output path for each run.

## Membership inference

The attack programs fit independent post-hoc attackers using the frozen observations. Membership labels are `1` for members and `0` for nonmembers. Fitting and evaluation identities are separated in both protocols.

### Identity-matched protocol

| Phase | Identities | Member instances | Nonmember instances |
| --- | --- | ---: | ---: |
| Fit | 001-049 | 2,156 | 2,156 |
| Evaluate | 050-074 | 1,100 | 1,100 |

For each identity and view, member inputs comprise two normal-walking sequences sampled without replacement from `nm-01` through `nm-04`, plus `bg-01` and `cl-01`. Nonmember inputs are `nm-05`, `nm-06`, `bg-02`, and `cl-02`. The seed controls the normal-walking selection.

```bash
python -m scripts.casia_b_mia --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --protocol identity_matched --device cuda --output outputs/casia_b/seed2025/membership_matched.json --checkpoint-dir outputs/casia_b/seed2025/attackers_matched
```

### Identity-disjoint stress protocol

| Phase | Member identities | Nonmember identities | Candidate member sequences | Sampled instances per class |
| --- | --- | --- | ---: | ---: |
| Fit | 001-049 | 075-099 | 3,234 | 2,750 |
| Evaluate | 050-074 | 100-124 | 1,650 | 2,750 |

Member candidates use the six recognition training conditions. Nonmember candidates use all ten conditions and supply 2,750 unique sequences in each phase. Fitting members are sampled without replacement; evaluation members are sampled with replacement to obtain 2,750 instances. Repeated evaluation sequence keys are therefore retained in the exported partition CSV.

```bash
python -m scripts.casia_b_mia --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --protocol identity_disjoint_stress --device cuda --output outputs/casia_b/seed2025/membership_stress.json --checkpoint-dir outputs/casia_b/seed2025/attackers_stress
```

### Attack inputs and metrics

| Report key | Input | Dimension |
| --- | --- | ---: |
| `score_only` | Retrieval confidence | 1 |
| `raw_only` | Statistical descriptor of raw explanation | 36 |
| `score_plus_raw` | Raw descriptor and retrieval confidence | 37 |
| `safe_only` | Statistical descriptor of protected release | 36 |
| `score_plus_safe` | Protected descriptor and retrieval confidence | 37 |

The descriptor encodes observable explanation fields, field visibility, release level, categorical values, and summary statistics. Each channel uses a separate MLP with hidden dimensions 128, 64, and 32, fitted for 50 epochs with Adam and binary cross-entropy. Training-time privacy auditor weights are not reused.

Reports contain ROC AUC on a 0-1 scale and TPR at 1% FPR as a percentage. `raw_interface_best_of` and `safe_interface_best_of` take the maximum of each metric over the corresponding score-only and explanation channels. Attacker weights are saved to the selected `--checkpoint-dir`; choose a new directory for another run.

## Three-seed aggregation

Repeat inference and evaluation for seeds 2024, 2025, and 2026 using the corresponding independently trained checkpoints, the same manifest, and separate output directories. Aggregate recognition and membership reports separately, and keep the two membership protocols in separate summaries.

```bash
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/principal_results.json outputs/casia_b/seed2025/principal_results.json outputs/casia_b/seed2026/principal_results.json --output outputs/casia_b/principal_summary.json
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/membership_matched.json outputs/casia_b/seed2025/membership_matched.json outputs/casia_b/seed2026/membership_matched.json --output outputs/casia_b/membership_matched_summary.json
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/membership_stress.json outputs/casia_b/seed2025/membership_stress.json outputs/casia_b/seed2026/membership_stress.json --output outputs/casia_b/membership_stress_summary.json
```

The summarizer requires one report per seed, matching metric schemas, and the same manifest-key checksum. It reports the arithmetic mean and sample standard deviation (`ddof=1`) for numeric fields. A seed-2025 checkpoint supplies the seed-2025 run; three-seed statistics require all three sets of artifacts.

## Output files

| Artifact | Contents |
| --- | --- |
| `manifest.csv` | Sequence labels and input paths |
| `partitions/recognizer_*.csv` | Recognition sequence roles |
| `partitions/<protocol>_seed<seed>_<phase>.csv` | Indexed attack instances and membership labels |
| `partitions/dataset_roles.json` | Recognition partition counts |
| `partitions/partition_summary.json` | Partition counts, unique sequence counts, and checksums |
| `seed<seed>/observations.jsonl` and its metadata JSON | Frozen-model observations and artifact provenance |
| `seed<seed>/principal_results.json` | Recognition and release statistics |
| `seed<seed>/membership_*.json` | Attack metrics and partition metadata |
| `seed<seed>/attackers_*/*.pt` | Fitted channel-specific attacker weights |
| `*_summary.json` | Three-seed means and sample standard deviations |

Protocol definitions are maintained in [opengait/data/protocols.py](../opengait/data/protocols.py). Checkpoint loading and inspection are described in [CHECKPOINTS.md](CHECKPOINTS.md).

The independent recognition-error audit, repeated-release budgets through 200 queries, native-field linkage, and manuscript table exports are described in [AUDIT_AND_QUERY_BUDGETS.md](AUDIT_AND_QUERY_BUDGETS.md).
