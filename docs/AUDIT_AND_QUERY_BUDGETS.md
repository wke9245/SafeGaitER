# Independent Audit and Query Budgets

This workflow implements the CASIA-B evaluation settings in Sections 5.3.1-5.3.4, 5.7.3, 5.8, and Appendix B. Run commands from the repository root using the environment described in [README.md](../README.md). Start with the licensed data manifest and checkpoint associated with the selected seed.

## Exact partitions

```bash
python -m scripts.casia_b_partitions --manifest outputs/casia_b/manifest.csv --seeds 2024 2025 2026 --include-linkage --output outputs/casia_b/partitions
```

Alongside the recognition and membership CSVs, this exports `audit_fit.csv`, `audit_evaluate.csv`, and seed-specific linkage pair CSVs. Audit CSVs contain sequence keys and labels identifying the source sequence; recognition-error labels are derived from the actual frozen-model predictions. Linkage pairs contain two source keys and a same-identity indicator. Pairs exclude self-pairs and duplicate unordered pairs within each phase.

| Evaluation | Fit partition | Evaluation partition |
| --- | --- | --- |
| Recognition-error triage | 1,650 probes, identities 075-099 | 1,650 probes, identities 100-124 |
| Identity-matched membership | 2,156 members and 2,156 nonmembers, identities 001-049 | 1,100 members and 1,100 nonmembers, identities 050-074 |
| Native-field linkage | 50,000 same-identity and 50,000 different-identity pairs, identities 075-099 | 20,000 pairs per class, identities 100-124 |

Triage includes NM#5-6, BG#1-2, and CL#1-2 over all 11 views. It excludes gallery sequences. Linkage draws from all ten conditions over all views within its identity partitions. The 50-subject, 2,200-sequence NM gallery remains shared across audit conditions. Membership follows the condition- and view-matched construction in [CASIA_B_REPRODUCTION.md](CASIA_B_REPRODUCTION.md).

## Independent recognition-error triage

Generate frozen observations using `scripts.casia_b_inference`, then run:

```bash
python -m scripts.casia_b_audit --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/observations.jsonl --seed 2025 --output outputs/casia_b/seed2025/triage.json --model-dir outputs/casia_b/seed2025/audit_models
```

The positive target is an incorrect top-ranked identity, computed by comparing `predicted_subject` with the sequence subject. Each input condition uses an independent model:

| Condition | Dimension |
| --- | ---: |
| Retrieval confidence only | 1 |
| Release-level one-hot only | 3 |
| Confidence and release level | 4 |
| Confidence and raw-explanation descriptor | 37 |
| Confidence and protected-packet descriptor | 37 |

Descriptors use only recipient-visible fields and their visibility indicators. Recognition embeddings, internal risk maps, predicted stability, and suppressed fields are excluded from the audit features.

`StandardScaler` is fitted exclusively on the 1,650 audit-fit probes and applied unchanged to final-test probes. Zero-variance features retain a scale factor of one. Each L2 logistic-regression model uses `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=5000`, and `tol=1e-4`. Positive-class probabilities rank errors directly, without threshold or hyperparameter selection on final-test labels. The audit solver runs on the CPU as specified by scikit-learn; it does not alter the frozen CUDA recognizer.

The report contains AP, AUROC, and Recall@10%, including the integer number of errors found among the top 165 probes. Tied scores use a fixed seed-specific random ordering for the review budget; AP and AUROC retain score ties. Paired AP differences are computed within each seed at full precision. Within-level AP and per-seed sample/error counts support Table D.11. A subgroup without positive targets has an undefined AP represented by JSON `null`.

`--model-dir` receives an NPZ file for each fitted scaler and logistic-regression coefficient set, `settings.json`, and `test_predictions.csv`. The prediction CSV retains sequence identities, error targets, observed release levels, and all five audit scores for numerical verification.

## Repeated stochastic releases

```bash
python -m scripts.casia_b_inference --manifest outputs/casia_b/manifest.csv --checkpoint path/to/casia_b_seed2025.pt --seed 2025 --releases 200 --device cuda --output outputs/casia_b/seed2025/repeated_observations.jsonl
python -m scripts.casia_b_repeated --manifest outputs/casia_b/manifest.csv --observations outputs/casia_b/seed2025/repeated_observations.jsonl --seed 2025 --device cuda --output outputs/casia_b/seed2025/repeated.json --checkpoint-dir outputs/casia_b/seed2025/repeated_attackers
```

The recognizer, identity contribution, risk map, retrieval confidence, release thresholds, source sequence, and diagnostic reference are fixed across requests. The private transformation independently resamples its stochastic perturbation for each packet. `safe_packet` is the first packet; `release_packets` stores the ordered requests. The metadata records `release_count` and the fixed release settings. The same source-level instance and fitting/evaluation partitions are retained at every budget.

The evaluator uses the first Q packets at Q = 1, 5, 10, 20, 50, 100, and 200. `--budgets` can select a subset of these values. It validates observation checksums, seed agreement, source keys, packet field permissions, release counts, and fixed release levels before attacker fitting. Streaming observation reads retain compact numeric features; input JSON objects are not retained for the entire repeated-release collection.

The statistical attack averages the 36-dimensional descriptors within each sequence. The native-field attack first encodes every visible packet, then averages the resulting 64-dimensional embeddings. Low-, medium-, and high-level encoders have widths 10-64-64, 19-64-64, and 12-32-64. Missing fields are absent from each branch's input. Score-fusion attacks append the unchanged confidence once after pooling. Linkage pools releases for each source separately, then constructs `[abs(z1-z2), z1*z2]` for its 128-64-1 pair classifier.

Each Q and seed receives five independent membership attackers: score-only, descriptor packet-only, descriptor score-fusion, native packet-only, and native score-fusion. Membership uses 50 epochs, batches of 256, Adam with learning rate 0.001 and weight decay 0.0001. Native linkage uses the same optimizer settings for 50 epochs with batches of 512 pairs. Final-epoch weights are used throughout. Repeated requests are never counted as additional independent test sequences.

Best-of membership selects the highest-AUC channel within each seed and budget. TPR at 1% FPR is linearly interpolated on that selected attacker's ROC curve. Channel selection is retained in report provenance. Membership and linkage manifests are shared across budgets, and their checksums are recorded. The checkpoint directory stores fitted attacker states, membership-instance CSVs, linkage-pair CSVs, and test logits. Each test-logit array follows the row order of its corresponding `membership_evaluate.csv` or `linkage_evaluate.csv`.

## Three-seed summaries and manuscript tables

Run the corresponding workflow independently for seeds 2024, 2025, and 2026, using the checkpoint and observation artifacts associated with each seed. Then aggregate the computed reports:

```bash
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/triage.json outputs/casia_b/seed2025/triage.json outputs/casia_b/seed2026/triage.json --output outputs/casia_b/triage_summary.json --table-dir outputs/casia_b/triage_tables
python -m scripts.casia_b_summary --inputs outputs/casia_b/seed2024/repeated.json outputs/casia_b/seed2025/repeated.json outputs/casia_b/seed2026/repeated.json --output outputs/casia_b/repeated_summary.json --table-dir outputs/casia_b/repeated_tables
```

The first command writes `table11.csv` and `tableD11.csv`; the second writes `table12.csv`. All table statistics are derived from the supplied reports. Means, sample standard deviations (`ddof=1`), best-of choices, and paired differences are computed before display rounding. CSV summaries use `+/-` between mean and standard deviation. An undefined subgroup metric leaves its table cell empty; a three-seed summary with any undefined value retains a null aggregate and reports the number of available runs.

Preserve the per-seed JSON reports, metadata, fitted model states, predictions, and partition CSVs together. Choose new report, model, and inference paths when rerunning a workflow.
