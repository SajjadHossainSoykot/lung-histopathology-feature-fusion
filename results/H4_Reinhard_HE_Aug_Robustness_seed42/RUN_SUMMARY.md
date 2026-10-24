# Lung H4: Reinhard stain normalization and H&E-inspired augmentation

## Method and source data

Fixed LC25000 lung-only manifest SHA-256: `d931c9b11d03716c3c426f251bc75eaa2051002acca99d3ba66d59d964285dea`. Frozen saved
H1 MobileNetV2 and H2 EfficientNetB0 pre-dropout pooled features (1,280 each) were
concatenated into 2,560-dimensional features, followed by two separately trained
Dropout(0.2) -> Dense(3,softmax) heads. Original H3 remained unchanged as control.

Train: 10,394 images (514 reconstructed source-image groups);
validation: 2,273 images (113 reconstructed groups);
held-out test: 2,333 images (not accessed or evaluated).
Reconstructed source groups are NOT independent patients, slides or hospitals.

Train-reference Reinhard normalization used 96 original training images
(32 per class), pooled masked LAB mean/std, and a bounded LAB
standard-deviation rescaling. Every H4-NA training image received one deterministic
offline training view, with H&E-inspired OD concentration factors; approximately
80% of rows were modified. This is NOT epochwise augmentation.
H4-N and H4-NA used identical numbers of training rows and identical normalized
validation features. H&E synthetic stress shifts were predeclared and applied before
normalization at inference (except H3, which uses its original raw pipeline).

## Validation results (not held-out test)

Clean validation macro-F1: H3 0.980139, H4-N 0.972577,
H4-NA 0.971261. See `validation_condition_metrics.csv`,
`validation_robustness_summary.csv` and `validation_details.json` for
per-shift classification results, accuracy, class-wise metrics and confusion matrices.
Values are descriptive; the clean/shifted data are images of the SAME validation
source groups, not independent new hospital cases. The use of validation data for
checkpoint selection and model comparison may induce selection optimism.

## Limits and reproducibility

The two predefined perturbations are synthetic, not real stain/scanner/hospital
shift. Reinhard color statistics do not explicitly deconvolve stains. The chosen
H&E basis is fixed and inspired by conventional H&E optical-density vectors;
its concentration scaling and the resulting transformed images are approximations.
Results do not demonstrate patient-level, hospital-level, clinical, or external
stain robustness. Any raw-image H4 inference must reproduce this notebook's
resize-with-pad, training-only target reference statistics, transform order,
and H1/H2 backbone preprocessing; the exported H4 image-input Keras graphs
accept *already normalized* RGB and DO NOT include Reinhard normalization.

H1, H2 and H3 checkpoints remain unchanged. No held-out test images were read
or evaluated in H4. Final test evaluation must be separate and performed only
after validation-based pipeline selection is frozen.
