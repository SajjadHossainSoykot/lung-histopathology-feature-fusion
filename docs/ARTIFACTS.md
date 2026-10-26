# Original Checkpoints and Evidence Locations

This document outlines the local and historical Google Drive evidence artifacts for the Lung Histopathology Feature Fusion research project.

The GitHub repository contains the executed notebook snapshots, dataset splits, class mappings, group metadata, Grad-CAM overlays, and lightweight derived figures and metric tables. Large Google Drive caches, raw histology tiles, zip archives, and large model binary weights (`*.keras`) remain intentionally excluded from git via `.gitignore`.

Original Google Drive root: `/content/drive/MyDrive/ICTIU_Final_Projects/lung/`.

```text
archives/LC25000.zip                               # Excluded from git (large archive)
splits/manifest_candidate.csv                      # Preserved in repository
splits/class_mapping.json                          # Preserved in repository
metadata/lc25000_image_groups.csv                  # Preserved in repository
results/audit.json                                 # Preserved in repository
results/class_counts.csv                           # Preserved in repository
results/training_previews/                         # Preserved in repository
results/H1_MobileNetV2_frozen_seed42/              # Metrics, configs, and training curves preserved
results/H2_EfficientNetB0_frozen_seed42/           # Metrics, configs, and training curves preserved
results/H3_MobileNetV2_EfficientNetB0_fusion_seed42/ # Metrics, summaries, and curves preserved
results/H4_Reinhard_HE_Aug_Robustness_seed42/      # Metrics, configs, and curves preserved
results/H_GradCAM_H3_validation_seed42/            # Saliency maps, panels, and overview preserved
results/H_Final_Test_H3_seed42/                    # Test report, predictions, and matrix preserved
feature_cache/H3_fusion_d87930d4f5c72dcf/          # Google Drive cache (excluded from git)
feature_cache/H4_stain_a28fa20606419f6c/           # Google Drive cache (excluded from git)
selection/H3_model_selection_freeze_2026-10-19.json # Model selection freeze record
selection/H3_final_test_protocol_2026-10-19.json   # Pre-registered test protocol
```

## Selected Model Fingerprints

| Artifact | Original SHA-256 Checksum |
| :--- | :--- |
| `H1_MobileNetV2_frozen_seed42/best_model.keras` | `503559b9333124d8be0b665a9fd9538220a750db2274ca55a9e61dff503dd6bf` |
| `H2_EfficientNetB0_frozen_seed42/best_model.keras` | `fcd95f3198c1a63641462b3a72e6b21cf501a4ad0928186bd91c2362c53fa5f6` |
| `H3_MobileNetV2_EfficientNetB0_fusion_seed42/best_head.keras` | `77deea98faef81db0fd8e6574598f677a18f9368fde3edccc67c3549b5098b75` |
| `H3_MobileNetV2_EfficientNetB0_fusion_seed42/fusion_image_model.keras` | `9e1a5bbf49da7e88a529f5ac08303c65e138d7c9f818959aaddd46f609890c95` |

The selected full-image model passed an entire clean-validation inference consistency check before the final TEST. The final test notebook's saved verification reported all 13 test artifacts present and the model hash unchanged.

## Provenance Documentation Links

- [Dataset and Splits Specification](DATASET_AND_SPLITS.md)
- [Grad-CAM Image Provenance Specification](GRADCAM_IMAGE_PROVENANCE.md)
- [Limitations and Clinical Scope](LIMITATIONS.md)
- [Notebook Provenance Metadata](NOTEBOOK_PROVENANCE.md)
- [Reproducibility Guidelines](REPRODUCIBILITY.md)
- [Experimental Results and Summary Tables](RESULTS.md)

**Artifact preservation rule:** Do not replace saved checkpoints or remove their historical data to match renamed notebook filenames. H1 and H2 feature extractors remain required for H3 reproducibility. H4 normalized-input models require the training-derived preprocessing external to the saved Keras files.
