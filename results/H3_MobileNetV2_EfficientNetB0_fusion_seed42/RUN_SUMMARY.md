# H3_MobileNetV2_EfficientNetB0_fusion_seed42 - measured experiment summary

## Study scope
Pooled feature-level fusion of the *frozen* ImageNet MobileNetV2 and EfficientNetB0
backbones saved as H1/H2 checkpoints. Their trained classification heads were NOT
averaged or fused. H1 produces 1,280 features; H2 produces 1,280 features;
concatenation creates 2,560 features. New H3 head: Dropout(0.2) -> Dense(3, softmax).

## Dataset and split
Original LC25000 lung-only source-image manifest: d931c9b11d03716c3c426f251bc75eaa2051002acca99d3ba66d59d964285dea.
Train: 10,394; validation: 2,273; held-out test: 2,333 (untouched).
Groups are *reconstructed source-image groups*, NOT patient/slide/hospital identifiers.
Pixel decode/resize uses float32 RGB [0,255] with resize-with-pad; both saved
backbones retain their respective internal scaling. Cache keyed to source/checkpoint hashes.

## Training protocol
Head-only learning with seed 42, batch size 16, Adam 0.001,
training class weights {0: 0.9973133755517175, 1: 1.0048337200309359, 2: 0.9978878648233487}, maximum 6 epochs,
early stopping patience 2, and best checkpoint by minimum validation loss.
Epochs completed: 6; selected best epoch: 5 (one-based).

## Measured validation performance (not test)
- Validation loss: 0.048743
- Accuracy: 0.980202
- Macro-F1: 0.980139
- Weighted-F1: 0.980192
- Correct: 2228 / 2273

## Limits
Feature fusion does not guarantee improved accuracy, and these experiments do not
prove clinical utility or external-site generalization. Source-group separation
reduces augmented-derivative leakage but is not patient-level separation.
No stain normalization, stain-aware augmentation, Grad-CAM, robustness analysis,
external validation, or held-out test evaluation was done in this notebook.
Softmax scores are not calibrated clinical probabilities.

## Reproducibility
Best H3 head checkpoint SHA-256: 77deea98faef81db0fd8e6574598f677a18f9368fde3edccc67c3549b5098b75
H1 baseline checkpoint SHA-256: 503559b9333124d8be0b665a9fd9538220a750db2274ca55a9e61dff503dd6bf
H2 baseline checkpoint SHA-256: fcd95f3198c1a63641462b3a72e6b21cf501a4ad0928186bd91c2362c53fa5f6
Image-input composite H3 model SHA-256: 9e1a5bbf49da7e88a529f5ac08303c65e138d7c9f818959aaddd46f609890c95
Feature-cache key: d87930d4f5c72dcffa8d2c65b333413707812fc3b59f79bdcdff5eae9496dcf6
Configuration fingerprint: 74e5b9b8fd9da1084f15e128093b29bba9488fd95efb3989b2acc9fb92b6575b
The executed notebook is saved separately from this evidence ZIP.
