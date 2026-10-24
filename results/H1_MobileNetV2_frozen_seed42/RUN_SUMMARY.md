# H1_MobileNetV2_frozen_seed42 - measured run summary

## Scope
Frozen ImageNet MobileNetV2 baseline with a three-class head. No fine-tuning,
stain normalization, extra online augmentation, fusion, Grad-CAM, or final test evaluation.
The source archive already includes augmented images.

## Data and protocol
Training images: 10,394; validation images: 2,273.
Held-out test images: 2,333; not evaluated.
Existing source-group split retained; not patient-level or external-hospital separation.
Class mapping: {"lung_aca": 0, "lung_n": 1, "lung_scc": 2}.
RGB images are resized with padding to 224 x 224 outside the model.
The saved model rescales [0, 255] to [-1, 1] internally.
Batch size: 16; seed: 42; Adam learning rate: 0.001.
Training loss is class-weighted; validation loss is unweighted.

## Validation results
- Epochs completed: 5 (maximum requested: 6).
- Best epoch by validation loss: 3 (one-based).
- Validation loss: 0.078156.
- Validation accuracy: 0.970963.
- Validation macro-F1: 0.970826.
- Validation weighted-F1: 0.970920.

Confusion-matrix rows are true classes; columns are predicted classes.
Per-class metrics and predictions are in the accompanying CSV/TXT files.

## Limits and interpretation
These are image-level validation metrics on related images within reconstructed
source-image groups. They do not establish independent patient-level performance,
clinical reliability, or external generalization. Softmax scores are not calibrated
clinical probabilities. Source-group annotations were reused, not reconstructed here.
The CSV epoch log is zero-based; figures and this summary are one-based.

## Provenance
Checkpoint SHA-256: 503559b9333124d8be0b665a9fd9538220a750db2274ca55a9e61dff503dd6bf
Manifest SHA-256: d931c9b11d03716c3c426f251bc75eaa2051002acca99d3ba66d59d964285dea
Archive SHA-256: 382c98315435cbbf559be5da94cc0b62f696434e30e15feefebe9c892de01728
Notebook revision: 2026-10-17-v1

The executed notebook is retained separately from the evidence archive.
