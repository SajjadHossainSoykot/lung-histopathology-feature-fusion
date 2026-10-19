# Executed notebook provenance

The six supplied research notebooks were copied **byte-for-byte** from the original executed project source snapshots. All contain saved outputs in each code cell, with no saved Python `error` outputs. No training was repeated and no saved results, run configurations, execution cells or original fingerprints were rewritten for this repository.

| Executed notebook | Code cells with outputs | Original `.ipynb` SHA-256 |
|---|---:|---|
| `H1_Lung_MobileNetV2_Baseline.ipynb` | 10/10 | `ef3b600c720d4533740ff2d6370c947d0c309e5a01f877fc93be3b9251595c96` |
| `H2_Lung_EfficientNetB0_Baseline.ipynb` | 10/10 | `c1779f0f776b0a8d1c9075b4c08d9fb5ed3d80052e1ac91b83e0d16e6b9446b9` |
| `H3_Lung_Deep_Feature_Fusion.ipynb` | 10/10 | `6a0c35492b1474fbdc896571e77f3f5b4602c3a81d8d7e1d3976d63e15460efc` |
| `H4_Lung_Stain_Normalization_Augmentation_Robustness.ipynb` | 13/13 | `884e159367422faeff9a5289736591c3ed3bf67135b5199e2328f8335204ec16` |
| `H_GradCAM_H3_Deep_Feature_Fusion.ipynb` | 8/8 | `6208748dfdbdd9f8e250245d5fe349538e58c35e6ca9e98e6131a0488297e5b6` |
| `H_Final_Test_H3_Deep_Feature_Fusion.ipynb` | 8/8 | `4d9844af57e242317832ead573cb12eff4d9330508c41039dfd96807a737d034` |

The new `00_Lung_Setup_and_Data_Checks.ipynb` is a **separately authored project-only setup verifier** with no execution outputs at packaging time. It does not produce a new candidate split or claim to reproduce completed training runs.

Some original notebook JSON configuration values mention earlier numeric filename aliases, including `02_Lung_MobileNetV2_Baseline.ipynb` for H1 and `04_Lung_EfficientNetB0_Baseline.ipynb` for H2. The renamed notebook filenames represent the original experiments. The historical provenance fields are deliberately kept unchanged.

The notebooks may refer to earlier experiment configuration documents as historical citations. Such provenance does not alter this repository's project scope. New documentation here concerns only lung histopathology classification.

Updating notebook Markdown or saved output metadata changes the `.ipynb` byte hashes, even without retraining. Treat future edited portability copies as distinct versions rather than silently replacing this archive.
