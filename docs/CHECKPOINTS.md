# Checkpoint Usage

## Release files

Download the checkpoint assets from [release v1.0.0](https://github.com/wke9245/SafeGaitER/releases/tag/v1.0.0) and place them in `checkpoints/`:

| File | Purpose |
| --- | --- |
| `safegait_er_casia_b_seed2025.pt` | Model weights, configuration, training metadata, and optimizer/scheduler states |
| `safegait_er_casia_b_seed2025.pt.json` | Saved model dimensions, metadata, and checkpoint checksum |
| `SHA256SUMS.txt` | SHA-256 checksums for the release assets |

The checkpoint records CASIA-B and seed 2025, with 512 feature channels, eight spatial strips, a 256-dimensional embedding, and 74 identity classes. The file size is 102,634,227 bytes, approximately 97.9 MiB.

## Saved entries and loading requirements

| Entry | Contents |
| --- | --- |
| `model` | Named model tensors in a PyTorch state dictionary |
| `config` | Dataset, seed, input dimensions, model dimensions, and recorded settings |
| `training` | Training seed, completed epochs, and recorded optimization metadata |
| `optimizer` | Optimizer parameter groups, update counters, and moment tensors |
| `scheduler` | Saved learning-rate scheduler state |

Inference uses the model weights and their configuration. Optimizer and scheduler states retain the training state and are not used to calculate inference outputs. Training metadata links the checkpoint to its evaluation seed.

The loading implementation must match the saved network definition, parameter names, tensor shapes, and method settings. The model loader validates the configuration and loads weights with `strict=True`. Where a YAML configuration is supplied to `scripts.pipeline`, it must equal the saved configuration. Changing dimension values in a configuration does not convert the stored tensors to another network.

## CUDA inspection

Run from the repository root:

```bash
python -m scripts.inspect_checkpoint --checkpoint checkpoints/safegait_er_casia_b_seed2025.pt --device cuda --output outputs/checkpoint_inspection.json
```

The report includes the checkpoint size and SHA-256 checksum, recorded architecture, training metadata, model tensor count, tensor finiteness, optimizer state checks, model binding, and CUDA environment. Compare `checkpoint_sha256` with the entry in `SHA256SUMS.txt` before using the file.

Read these fields before inference:

| Report field | Interpretation |
| --- | --- |
| `tensor_loading` and `model_finiteness` | Tensor transfer and numerical checks |
| `optimizer` | Parameter-state counts, update counters, moment shapes, and finiteness checks |
| `architecture_binding` | Whether the supplied implementation supports strict model loading, or requires the originating network definition and method settings |
| `forward_pass` | Sequence output checks when `--input` is supplied |

Tensor and optimizer checks establish file-state consistency. Model execution additionally requires matching architecture binding and a sequence forward pass. The inspection command can produce a file-state report when the originating network definition is required; use `architecture_binding` to determine the next loading step.

## Sequence inspection and release

For a checkpoint supported by the supplied network implementation, request a forward pass with a silhouette sequence:

```bash
python -m scripts.inspect_checkpoint --checkpoint path/to/model.pt --input path/to/sequence.npy --device cuda --output outputs/checkpoint_forward.json
```

This checks finite outputs and the expected shapes: embedding `[1, 256]`, identity logits `[1, 74]`, contribution/stability/local-risk maps `[1, 30, 8]`, and sequence risk `[1]`.

To obtain the protected explanation packet:

```bash
python -m scripts.pipeline --input path/to/sequence.npy --checkpoint path/to/model.pt --device cuda --output outputs/release.json
```

Dataset recognition and membership metrics are computed using the [CASIA-B evaluation workflow](CASIA_B_REPRODUCTION.md). Use the checkpoint associated with the selected seed, preserve observation metadata files, and keep run outputs in separate directories. Forward inspection checks sequence execution; recognition accuracy is measured by the dataset evaluation commands.
