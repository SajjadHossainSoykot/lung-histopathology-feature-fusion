# Stain-Robust Explainable Deep Feature Fusion for Lung Histopathology Classification

A research implementation of three-class lung histopathology image classification on the **lung subset of LC25000**, using frozen ImageNet-pretrained feature extractors, feature concatenation, stain-aware ablations, and Grad-CAM. This repository contains the **executed** experiment notebooks and lightweight measured results from the undergraduate project at the Department of Information and Communication Technology, Islamic University, Bangladesh.

**Academic project author:** S M Sajjad Hossain Soykot. **Supervisor:** Dr. Md. Zahidul Islam. Code and analysis contributions should be attributed according to their actual provenance. Repository ownership and commit authorship are separate from academic authorship.

## Research questions and scope

The work compares MobileNetV2 and EfficientNetB0 single-backbone baselines with pooled-feature fusion, and examines whether training-derived Reinhard stain normalization and H&E-inspired training augmentation change classification performance under predetermined **synthetic** stain shifts. Grad-CAM produces qualitative class-discriminative heatmaps from both fusion branches.

This is **research on a public image dataset**, not a validated clinical diagnostic system. Synthetic stain shifts do not establish real cross-hospital generalization.

## Dataset and fixed partitions

| Class ID | Class | Meaning |
|---:|---|---|
| 0 | `lung_aca` | Lung adenocarcinoma |
| 1 | `lung_n` | Benign lung tissue |
| 2 | `lung_scc` | Lung squamous cell carcinoma |

| Split | Images | Reconstructed source-image groups |
|---|---:|---:|
| Training | 10,394 | 514 |
| Validation | 2,273 | 113 |
| Held-out TEST | 2,333 | 113 |
| **Total** | **15,000** | **740** |

The split is **source-image-group-separated**, not patient-, slide-, or institution-separated. The saved split seed was 42. Keep the original `manifest_candidate.csv` and `class_mapping.json` unchanged. See [Dataset and splits](docs/DATASET_AND_SPLITS.md).

## Verified results

The final model was selected using validation results and formally frozen **before** the one-time held-out test. Model architecture is not changed in response to test performance.

| Model/condition | Validation accuracy | Validation macro-F1 |
|---|---:|---:|
| H1: MobileNetV2 | 97.0963% | 0.970826 |
| H2: EfficientNetB0 | 97.2283% | 0.972158 |
| **H3: MobileNetV2 + EfficientNetB0 pooled-feature fusion** | **98.0202%** | **0.980139** |
| H4-N: Reinhard normalization | 97.2723% | 0.972577 |
| H4-NA: Reinhard plus H&E-inspired training augmentation | 97.1403% | 0.971261 |

**H3 FINAL held-out TEST (2,333 images):** Accuracy **96.2280%**, macro-F1 **0.961961**, macro-precision **0.962412**, macro-recall **0.961992**.

H3 maintained the highest **absolute** macro-F1 on clean and both synthetic-shift validation sets. H4-NA had the smallest *relative drop from its own clean baseline* under the predefined shifts. These are different outcomes and neither constitutes external-site testing. See [Results](docs/RESULTS.md).

## Method at a glance

1. H1: frozen MobileNetV2 feature extractor, global average pooling, dropout, three-way softmax.
2. H2: frozen EfficientNetB0 feature extractor with its appropriate model-native preprocessing, global average pooling, dropout, three-way softmax.
3. H3: concatenate the two 1,280-dimensional **pooled features** into a 2,560-dimensional vector; train a small classification head. This is **feature fusion**, not probability averaging. H3's full-image model implements branch-specific preprocessing internally.
4. H4: compare H3 with newly extracted stain-normalized or normalized-plus-augmented training features; use a training-only Reinhard reference and two predefined synthetic validation shifts. H4 normalized-input models require **external** Reinhard preprocessing and are not the selected final TEST pipeline.
5. Grad-CAM: illustrate validation-only evidence from both H3 branches for a fixed set of examples; heatmaps are not proof of pathologic causal localization.

Input size is 224 x 224 x 3, seed 42, with frozen pretrained backbones. Check the executed notebooks and their saved run configurations for exact optimizer, preprocessing, checkpoint, and evaluation details.

## Repository structure

```text
notebooks/
  00_Lung_Setup_and_Data_Checks.ipynb                 # New, not yet executed
  H1_Lung_MobileNetV2_Baseline.ipynb                  # Executed original
  H2_Lung_EfficientNetB0_Baseline.ipynb               # Executed original
  H3_Lung_Deep_Feature_Fusion.ipynb                   # Executed original
  H4_Lung_Stain_Normalization_Augmentation_Robustness.ipynb  # Executed original
  H_GradCAM_H3_Deep_Feature_Fusion.ipynb              # Executed original
  H_Final_Test_H3_Deep_Feature_Fusion.ipynb           # Executed original

docs/                                                 # Data, methods, reproduction, provenance
results/figures/                                      # Exported from embedded notebook outputs
results/tables/                                       # Derived from measured notebook outputs
report/                                               # Status; report source to be added separately
```

## Start here

- Read [Reproducibility](docs/REPRODUCIBILITY.md) **before running any experiment code**. The completed experiment notebooks are a historical record, not a command to re-train or re-test.
- Open `notebooks/00_Lung_Setup_and_Data_Checks.ipynb` in Colab to verify the five original saved Drive inputs. It never modifies a source split or uses TEST image pixels. This new notebook is **not yet executed** in the released ZIP.
- Saved models, archives, generated caches, and complete test evidence remain in Google Drive and are deliberately excluded from Git.

**Source datasets:** [LC25000 mirror on Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) and [LC25000-clean source grouping](https://github.com/GeorgeBatch/LC25000-clean). Dataset licenses and pretrained-weight redistribution permissions must be checked separately before any public distribution. This repository has no automatic grant to redistribute third-party content.

See [Limitations](docs/LIMITATIONS.md), [Artifact locations](docs/ARTIFACTS.md), and [Notebook provenance](docs/NOTEBOOK_PROVENANCE.md).
