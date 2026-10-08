# Method components

## Hierarchical gait evidence

The feature encoder uses a lightweight hierarchical convolutional architecture inspired by IHGR-Net. Shared convolutional stages feed global and local branches. Activation-guided reliability then refines their combined feature map at 1, 2, and 4 horizontal regions.

For each region, the mean absolute activation supplies an energy statistic. A learned scale and bias map this statistic through a sigmoid gate. The gate weights the region's feature map, and the three spatial granularities are averaged. The refined frame output has 128 channels and spatial dimensions 16 by 11.

SafeGait-ER subsequently forms a separate evidence representation with eight horizontal regions per frame. Regional reliability weights pooled features to produce 256-dimensional frame descriptors. A two-layer temporal MLP supplies frame importance, followed by softmax and weighted sequence aggregation.

## Stability prediction and identity contribution

The local descriptor contains 128 normalized regional features, spatial reliability, temporal importance, and identity contribution, giving 131 dimensions. Confidence extends the stability-predictor input to 132 dimensions.

The predictor uses widths 132, 704, 352, and 1, with ReLU hidden activations and a sigmoid output. The retained stability-target function computes consistency between two augmented descriptors. The method pipeline estimates stability from one view.

Identity contribution uses positive gradient-times-activation responses, averaged within each strip and normalized over the sequence. The pipeline targets the predicted identity logit. When a caller supplies a gallery to the model, contribution instead targets the highest-matching gallery sequence of the predicted subject.

## Risk estimation

Local risk combines spatial reliability, temporal importance, identity contribution, and predicted stability. The local map is min-max normalized within each sequence. Sequence risk is the top-20-percent mean multiplied by confidence. With 30 frames and eight regions, the aggregation selects 48 units.

The pipeline obtains confidence from the identity classifier. The retained gallery interface computes mean cosine similarity per subject with temperature 0.05.

## Private transformation and release

The transformation applies risk-adaptive clipping, fine-level masking, branch-specific coarsening, Gaussian perturbation, and release-level masking.

Spatial groups use strips 0-2, 3-4, and 5-7. Temporal intervals are [0, 8), [8, 15), [15, 23), and [23, 30).

| Parameter | Value |
| --- | --- |
| Clipping parameters | 0.30, 0.70 |
| Fine and coarse mask threshold | 0.70 |
| Maximum noise standard deviation | 0.10 |
| Low/medium threshold | 0.35 |
| Medium/high threshold | 0.65 |
| Soft-routing sharpness | 20 |

| Level | Visible fields |
| --- | --- |
| Low | Three spatial scores, four temporal scores, level flag |
| Medium | Dominant region, dominant interval, level flag, diagnostic category |
| High | Level flag, diagnostic category |

The diagnostic mapper reads the protected state before release-level masking. Its nine categories are normal walking, carried object, clothing variation, local occlusion, lower-body occlusion, consecutive-frame occlusion, silhouette jitter, frame dropping, and combined disturbance.

The pipeline uses the configured reliability reference [0.70, 0.70, 0.70]. Dataset conditions and disturbance labels are not supplied to the diagnostic mapper.

## Adversarial learning components

Independent branch encoders map 10-, 19-, and 12-dimensional release vectors into 128-dimensional training representations. Soft routing retains gradients through continuous scores and release weights. Discrete categories, indices, and masking decisions are held fixed.

Privacy auditors provide auxiliary membership, identity, and linkage objectives. The objective functions combine recognition, stability, auditability, compactness, risk suppression, and reversed privacy loss. These modules are retained for method inspection and integration; the single-sequence entry point does not optimize them.

All parameters used by the retained method components are specified in the YAML configuration. Outputs from untrained parameters do not represent empirical recognition or privacy results.
