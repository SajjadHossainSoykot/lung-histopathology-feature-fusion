# Original checkpoints and evidence locations

The repository contains the executed notebook snapshots and small derived figures/tables. It intentionally does not contain large Google Drive caches, training images, archived datasets, or model weights.

Original Drive root: `/content/drive/MyDrive/ICTIU_Final_Projects/lung/`.

```text
archives/LC25000.zip
splits/manifest_candidate.csv
splits/class_mapping.json
metadata/lc25000_image_groups.csv
results/audit.json
results/H1_MobileNetV2_frozen_seed42/
results/H2_EfficientNetB0_frozen_seed42/
results/H3_MobileNetV2_EfficientNetB0_fusion_seed42/
results/H4_Reinhard_HE_Aug_Robustness_seed42/
results/H_GradCAM_H3_validation_seed42/
results/H_Final_Test_H3_seed42/
feature_cache/H3_fusion_d87930d4f5c72dcf/
feature_cache/H4_stain_a28fa20606419f6c/
selection/H3_model_selection_freeze_2026-10-19.json
selection/H3_final_test_protocol_2026-10-19.json
```

**Selected model fingerprints:**

| Artifact | Original SHA-256 |
|---|---|
| `H3_MobileNetV2_EfficientNetB0_fusion_seed42/fusion_image_model.keras` | `9e1a5bbf49da7e88a529f5ac08303c65e138d7c9f818959aaddd46f609890c95` |
| `H3_MobileNetV2_EfficientNetB0_fusion_seed42/best_head.keras` | `77deea98faef81db0fd8e6574598f677a18f9368fde3edccc67c3549b5098b75` |

The selected full-image model passed an entire clean-validation inference consistency check before the final TEST. The final test notebook's saved verification reported all 13 test artifacts present and the model hash unchanged. The actual Drive artifact bytes have **not** been copied into this GitHub package.

**Artifact preservation rule:** Do not replace saved checkpoints or remove their historical data to match renamed notebook filenames. H1 and H2 feature extractors remain required for H3 reproducibility. H4 normalized-input models require the training-derived preprocessing external to the saved Keras files.
