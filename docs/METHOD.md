# Method and implementation

## Hierarchical gait evidence

The input is a 30-frame silhouette sequence at resolution 64 by 44. Frame features have 128 channels and are partitioned into eight horizontal strips. Reliability-weighted strip features produce 256-dimensional frame descriptors, which are aggregated by learned temporal importance.

The hierarchical trunk combines shared frame convolutions with global and local feature branches. The GaitPart trunk uses plain and focal frame convolutions while retaining the SafeGait-ER evidence heads.

## Stability prediction and identity contribution

A local explanation descriptor has 131 dimensions: 128 normalized regional features, spatial reliability, temporal importance, and identity contribution. Confidence is appended for the 132-dimensional predictor input.

The stability predictor uses widths 132, 704, 352, and 1, with ReLU hidden activations and a sigmoid output. Two independently augmented views supply the cosine-consistency target during training. Inference uses the predictor on a single view.

Identity contribution is the positive gradient-times-activation response, averaged within each strip and normalized over the sequence. Training uses the predicted identity logit. Open-set explanation inference uses similarity to the highest-matching gallery sequence of the predicted subject.

## Risk estimation

Local risk combines spatial reliability, temporal importance, identity contribution, and predicted stability. The local map is min-max normalized within each sequence. Sequence risk is the top-20-percent mean, multiplied by confidence. With 30 frames and eight strips, 48 risk units are selected.

Training confidence comes from the identity classifier. Retrieval confidence uses mean gallery similarity per subject with temperature 0.05.

## Private transformation and release

The transformation applies risk-adaptive clipping, fine-level masking, branch-specific coarsening, Gaussian perturbation, and release-level masking.

The spatial groups are:

- Upper: strips 0, 1, 2
- Middle: strips 3, 4
- Lower: strips 5, 6, 7

The temporal intervals are [0, 8), [8, 15), [15, 23), and [23, 30).

| Parameter | Value |
| --- | --- |
| Clipping parameters | 0.30, 0.70 |
| Fine and coarse mask threshold | 0.70 |
| Maximum noise standard deviation | 0.10 |
| Low/medium threshold | 0.35 |
| Medium/high threshold | 0.65 |
| Soft-routing sharpness | 20 |

The released fields are:

| Level | Visible fields |
| --- | --- |
| Low | Three spatial scores, four temporal scores, level flag |
| Medium | Dominant region, dominant interval, level flag, diagnostic category |
| High | Level flag, diagnostic category |

The diagnostic mapper operates on the protected state before release-level masking. Its nine categories are normal walking, carried object, clothing variation, local occlusion, lower-body occlusion, consecutive-frame occlusion, silhouette jitter, frame dropping, and combined disturbance.

For natural sequences, reliability references use protected same-view NM gallery states from the predicted identity, excluding self-matches. Controlled disturbances use the paired clean source. The configured fallback reference is [0.70, 0.70, 0.70].

## Baseline protection

The individual controls apply clipping, Gaussian perturbation, random masking, or spatial-temporal coarsening. Generic hardening combines clipping, perturbation, and random masking. The DeepLeak gait adaptation applies Gaussian perturbation, range clipping, precision reduction, and small-attribution threshold masking to the continuous fields. Its settings are recorded under `baselines.deepleak` in the configuration. The key-frame indicator remains observable for continuous-field controls.

## Adversarial learning

Separate branch encoders map the 10-, 19-, and 12-dimensional release vectors into 128-dimensional training representations. Soft routing makes the continuous scores and release weights differentiable. Discrete categories, indices, and masking decisions are held fixed.

Auditor-side encoders and privacy auditors minimize the auxiliary privacy objective. Recognition and stability modules receive the opposite privacy-gradient direction together with recognition, stability, auditability, compactness, and risk-suppression objectives.

Independent post-hoc attackers consume hard release packets and do not share parameters or optimizer states with training-time auditors.

## Auditability and attacks

Internal fidelity uses confidence drops from retaining the dominant coarse spatial region and temporal interval intersection. Natural-sequence coverage and controlled-disturbance measurements have separate denominators.

Statistical packet descriptors have 36 dimensions. Native-field attack encoders map each visible interface into 64 dimensions. Reconstruction concatenates a 256-dimensional recognition embedding and a 256-dimensional visible-explanation encoding before 3D upsampling to 30 frames.

All training constants and attacker schedules are specified in the YAML configurations. Table summaries are calculated from the saved evaluation outputs.
