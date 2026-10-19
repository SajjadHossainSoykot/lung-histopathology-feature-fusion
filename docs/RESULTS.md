# Recorded experimental results

All figures and data here are derived from the executed lung notebooks or the saved results those notebooks report. Validation and held-out TEST results are deliberately separated.

## Validation-only model comparisons

| Experiment | Model | N | Accuracy | Macro-F1 | Best epoch selected by val loss |
|---|---|---:|---:|---:|---:|
| H1 | Frozen MobileNetV2 | 2,273 | 0.970963 | 0.970826 | 3 |
| H2 | Frozen EfficientNetB0 | 2,273 | 0.972283 | 0.972158 | 6 |
| H3 | Frozen pooled MobileNetV2 + EfficientNetB0 features | 2,273 | 0.980202 | 0.980139 | 5 |
| H4-N | Reinhard-normalized fused features, new head | 2,273 | 0.972723 | 0.972577 | 6 |
| H4-NA | Reinhard plus H&E-inspired augmented training features | 2,273 | 0.971403 | 0.971261 | 6 |

H3's clean validation confusion matrix, in class order `lung_aca`, `lung_n`, `lung_scc` (rows: true, columns: predicted):

| True / Predicted | lung_aca | lung_n | lung_scc |
|---|---:|---:|---:|
| lung_aca | 739 | 6 | 13 |
| lung_n | 2 | 762 | 0 |
| lung_scc | 24 | 0 | 727 |

## H4: Synthetic stain-shift validation study

The H4 reference was fitted using training images only. The two synthetic H&E variations were fixed before the stress evaluation. The shifts do **not** reproduce an independent hospital, scanner or stain-lab distribution.

| Condition | Clean macro-F1 | H-up/E-down macro-F1 | H-down/E-up macro-F1 | Mean shifted drop from clean |
|---|---:|---:|---:|---:|
| Original H3 | **0.980139** | **0.976148** | **0.971283** | 0.006423 |
| H4-N Reinhard | 0.972577 | 0.969955 | 0.968144 | 0.003527 |
| H4-NA Reinhard + H&E train augmentation | 0.971261 | 0.970407 | 0.969954 | **0.001081** |

H3 achieved the highest **absolute** macro-F1 across the clean and both synthetic shifted conditions. H4-NA showed the smallest average change *relative to its own clean baseline*. This does not imply that H4-NA outperformed H3 in absolute shifted performance. There is no augmentation-only-without-normalization ablation, so those effects cannot be isolated.

## Final held-out TEST: selected frozen H3 only

The final TEST occurred **once** after saving the H3 selection freeze and test protocol. TEST images were separated by reconstructed source-image group, not by biological patient.

| Metric | Final TEST result |
|---|---:|
| Images | 2,333 |
| Accuracy | **96.2280%** |
| Macro-F1 | **0.961961** |
| Macro-precision | **0.962412** |
| Macro-recall | **0.961992** |

The per-class TEST report, confusion matrix and prediction CSV are retained in the user's original Drive `results/H_Final_Test_H3_seed42/`. Do not fill in per-class values without retrieving those original saved records. No H1, H2 or H4 final TEST was conducted.

## Grad-CAM: fixed validation examples

The frozen H3 model was explained on 6 fixed validation images, two per class, representing six source groups. Each branch's heatmap targets the **same H3 predicted-class logit**; the experiment's six sampled predictions agreed with saved historical inference. These small illustrative samples are not a measured external accuracy or clinical explanation benchmark.

See the preserved `H_GradCAM_H3_Deep_Feature_Fusion.ipynb` and `results/figures/H3_GradCAM_validation_example.png` for the original generated visual output.
