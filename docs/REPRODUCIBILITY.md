# Reproducibility guide

## What is provided

All **six original executed notebooks** are included with their saved output cells. Their bytes match the original submitted notebook snapshots. The new `00_Lung_Setup_and_Data_Checks.ipynb` is a distinct, unexecuted, independent verification notebook.

Two forms of reproducibility should be kept separate:

- **Artifact verification:** compare saved file fingerprints, inspect executed notebooks, plots and tabular metrics, and verify the original result directories in Drive without repeating training or the held-out TEST.
- **Fresh training reproduction:** on a separately authorized run, restore the exact archive and split manifest, use the recorded environment and configurations, and write new checkpoints to **new result folders**. GPU/software nondeterminism may prevent byte-identical weights.

## Required original files

Your Drive account must expose `/content/drive/MyDrive/ICTIU_Final_Projects/lung/` with these original saved files:

```text
archives/LC25000.zip
splits/manifest_candidate.csv
splits/class_mapping.json
metadata/lc25000_image_groups.csv
results/audit.json
```

Also retain H1/H2/H3/H4 checkpoints, feature caches, selection freeze, and final TEST report under the saved Drive structure described in [Artifacts](ARTIFACTS.md). Dataset archives, saved checkpoints and medical images are not bundled in this repository.

## Safe inspection order

1. Open `notebooks/00_Lung_Setup_and_Data_Checks.ipynb` in Colab and execute its data-verification cells sequentially. It creates only a temporary local workspace, verifies original files and reads a deterministic training/validation-only image sample. No newly generated split is permitted.
2. Read H1/H2 baseline saved outputs and configurations. They validate preprocessing differences between MobileNetV2 and EfficientNetB0.
3. Read H3 saved feature concatenation and validation evidence. H3 depends on both H1 and H2 pooled features.
4. Read H4 saved validation stain-normalization, stain-augmentation and synthetic-shift comparisons. H4 models require external Reinhard normalization at inference; H3 does **not**.
5. Inspect the validation-only Grad-CAM notebook and its frozen H3 checkpoint provenance.
6. **Read, do not rerun**, the final TEST notebook and preserved evidence. The single planned held-out TEST was completed already.

## Execution modes and cautions

The executed research notebooks store historical paths and mode switches from the original Colab runs, sometimes with `MODE = 'train_new'`. Do not use `Run all` on these archived versions. Re-running them without a separate protocol risks overwriting stored results or repeating a finalized test. The original H4 training feature extraction was computationally expensive and does not need repeating for packaging.

The original H3 model uses 224 x 224 RGB inputs and two distinct frozen backbones with appropriate branch preprocessing embedded in the exported `fusion_image_model.keras`. Its 2,560-D pooled-feature concatenation passes to a trained dropout/dense classification head. Do not apply external H4 Reinhard normalization to the selected H3 model.

## Recorded environment

Historical H1 Colab runtime recorded Python `3.13.15`, TensorFlow `2.21.0`, Keras `3.13.2`, NumPy `2.1.3`, pandas `2.2.3`, scikit-learn `1.6.1`, matplotlib `3.10.0`, Pillow `11.3.0`, tqdm `4.67.3`, and a Tesla T4 GPU. H4 also used OpenCV (`cv2`) and reported TensorFlow/Keras versions. The exact H4 OpenCV package version is not established in this repository. See `requirements.txt`; package availability and Colab GPU allocation may differ today.

Seed 42, RGB inputs, and the fixed manifest are recorded. A notebook with saved outputs is evidence of previous execution, not proof that rerunning it will produce identical GPU weights.

## Separating validation from TEST

- `H1`, `H2`, `H3`, `H4`: all reported model selection and stain-shift comparisons use **validation**.
- `H_GradCAM`: fixed **validation-only** examples (6 cases, 2 per class).
- `H_Final_Test`: **one-time held-out TEST** for previously selected and frozen H3 (2,333 source-group-separated images). It is complete and must not be rerun to refresh screenshots, outputs, or results.

No performance metric in this repository should be interpreted as clinical reliability or external hospital generalization.
