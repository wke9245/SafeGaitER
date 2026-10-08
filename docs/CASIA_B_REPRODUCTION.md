# CASIA-B evaluation

The evaluation uses a frozen SafeGait-ER recognition model, subject-level gallery retrieval, independently fitted membership attackers, and risk-conditioned explanation release.

## Inputs

Use the licensed CASIA-B silhouettes and the compatible trained checkpoint for each seed: 2024, 2025, and 2026. See [checkpoint preparation](CHECKPOINTS.md).

The indexer accepts the following sequence layout:

```text
CASIA-B/
  001/nm-01/000/silhouettes.npy
  001/nm-01/018/silhouettes.npy
  ...
  124/cl-02/180/silhouettes.npy
```

Each view folder contains one complete sequence array in NPY, NPZ, or PKL format, or ordered silhouette image frames. NPZ files contain a `silhouettes` array or a single array. Sequence arrays use shape `[frames, height, width]`. Silhouettes are normalized to 64 by 44 and sampled deterministically to 30 frames.

A CSV manifest has columns `subject,condition,view,path`. Relative sequence paths resolve from the manifest directory. Each of its 13,640 rows identifies one complete sequence. View angles are 000 through 180 in steps of 18 degrees.

## Recognition and release partitions

| Role | Identities | Conditions | Sequences |
| --- | --- | --- | ---: |
| Recognition training members | 001-074 | NM 01-04, BG 01, CL 01 | 4,884 |
| Reserved nonmembers | 001-074 | NM 05-06, BG 02, CL 02 | 3,256 |
| Recognition gallery | 075-124 | NM 01-04 | 2,200 |
| Recognition probes | 075-124 | NM 05-06, BG 01-02, CL 01-02 | 3,300 |
| Held-out release pool | 075-124 | All conditions | 5,500 |

Each held-out subject has 44 gallery sequences. Retrieval averages cosine similarity over that subject's gallery sequences. The subject with the highest mean similarity supplies the recognition decision. Confidence uses a subject softmax with temperature 0.05. A query present in the gallery is excluded from itself.

Attribution targets the highest-matching gallery sequence within the predicted subject. Diagnostic reference values average protected spatial reliability over the predicted subject's same-view NM gallery sequences, excluding the query.

## Membership partitions

Identity-matched fitting uses subjects 001-049; evaluation uses 050-074. For each subject and view, members include two seed-selected NM 01-04 sequences, BG 01, and CL 01. Nonmembers include NM 05-06, BG 02, and CL 02.

| Phase | Members | Nonmembers |
| --- | ---: | ---: |
| Fitting | 2,156 | 2,156 |
| Evaluation | 1,100 | 1,100 |

The identity-disjoint stress protocol uses the following source pools:

| Phase | Member identities | Member source sequences | Nonmember identities | Instances per class |
| --- | --- | ---: | --- | ---: |
| Fitting | 001-049 | 3,234 | 075-099 | 2,750 |
| Evaluation | 050-074 | 1,650 | 100-124 | 2,750 |

Stress fitting members are sampled without replacement. Stress evaluation members are sampled with replacement from 1,650 available sequences. Each nonmember pool contains 2,750 distinct sequences. Exported partition CSV files retain an instance index so repeated selections remain visible.

## Execution

Activate the [GPU environment](RUNNING.md) and run from the repository root.

1. Index the silhouettes and export the recognition and attack partitions:

```bash
python -m scripts.casia_b_index --data-root data/CASIA-B --manifest outputs/casia_b/manifest.csv
python -m scripts.casia_b_partitions --manifest outputs/casia_b/manifest.csv --seeds 2024 2025 2026
```

2. Produce frozen-model observations for seed 2025:

```bash
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint checkpoints/seed2025.pt --seed 2025 --device cuda --output outputs/casia_b/seed2025/observations.jsonl
```

The checkpoint training seed must match the inference seed. The output path must be new.

3. Calculate recognition and release statistics:

```bash
python -m scripts.casia_b_results --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --device cuda --output outputs/casia_b/seed2025/principal_results.json
```

4. Fit and evaluate the identity-matched attackers:

```bash
python -m scripts.casia_b_mia --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --device cuda --output outputs/casia_b/seed2025/membership.json
```

5. Evaluate the identity-disjoint stress protocol:

```bash
python -m scripts.casia_b_mia --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --protocol identity_disjoint_stress --device cuda --output outputs/casia_b/seed2025/stress.json
```

Repeat steps 2-5 for seeds 2024 and 2026 with their corresponding checkpoints and output directories.

Aggregate recognition and release results:

```bash
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/principal_results.json outputs/casia_b/seed2025/principal_results.json outputs/casia_b/seed2026/principal_results.json --output outputs/casia_b/principal_summary.json
```

Aggregate membership results:

```bash
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/membership.json outputs/casia_b/seed2025/membership.json outputs/casia_b/seed2026/membership.json --output outputs/casia_b/membership_summary.json
```

Apply the same aggregation command to the three stress reports for the stress summary. Summaries contain the mean and sample standard deviation with one degree-of-freedom correction.

## Attacker configuration

Each interface has a separate MLP with hidden widths 128, 64, and 32, ReLU activations, and one membership logit. Fitting uses 50 epochs, batches of 256, Adam with learning rate 0.001 and weight decay 0.0001, and the corresponding experiment seed. Evaluation sequences do not enter optimization.

| Channel | Features | Dimension |
| --- | --- | ---: |
| Score-only | Recognition confidence | 1 |
| Raw-only | Statistical raw-explanation descriptor | 36 |
| Score + Raw | Raw descriptor and confidence | 37 |
| Safe-only | Statistical release-packet descriptor | 36 |
| Score + Safe | Safe descriptor and confidence | 37 |

The descriptor contains three level indicators, three spatial scores, four temporal scores, three region indicators, four interval indicators, nine diagnostic indicators, six visibility or high-level indicators, and four entropy or maximum statistics. Fields absent from a recipient packet have zero values and zero visibility indicators.

AUC uses midranks for tied scores. TPR at 1% FPR uses the highest empirical TPR satisfying the FPR bound, with tied scores processed together. Raw-interface and safe-interface best-of values are the maximum over their three relevant channels.

## Artifacts

| Artifact | Contents |
| --- | --- |
| `partitions/*.csv` | Recognition roles and exact seed-specific attack instance keys |
| `partition_summary.json` | Partition counts and file checksums |
| `observations.jsonl` | Embeddings, confidence, risk, raw fields, and release packets |
| `observations.jsonl.metadata.json` | Seed, checkpoint checksum, manifest-key checksum, and observation checksum |
| `principal_results.json` | NM/BG/CL Rank-1, their mean, and held-out release statistics |
| `membership.json`, `stress.json` | Independent attack metrics and partition counts |
| `attack_checkpoints/*.pt` | Fitted post-hoc attacker weights and configuration |
| `*_summary.json` | Three-seed aggregate statistics and artifact provenance |

Recipient packets expose only their policy fields: low releases spatial and temporal scores; medium releases dominant region, dominant interval, and diagnostic; high releases diagnostic. Every packet includes its level.

Recognition and TPR outputs use percentages; AUC and release fractions use values in [0, 1]. Generated observations contain identity-associated biometric derivatives and belong in a controlled experiment directory. Datasets, generated observations, and weight files are excluded from source control.
