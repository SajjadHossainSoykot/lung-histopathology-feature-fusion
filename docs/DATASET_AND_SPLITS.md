# Dataset, source groups, and immutable splits

This project uses the **lung-only** classes of LC25000, with 5,000 image rows per class. The source archive was originally retrieved from the LC25000 Kaggle mirror. The reconstructed source-image grouping metadata came from [LC25000-clean](https://github.com/GeorgeBatch/LC25000-clean).

## Fixed labels and split

| Class name | ID | Full dataset count |
|---|---:|---:|
| `lung_aca` | 0 | 5,000 |
| `lung_n` | 1 | 5,000 |
| `lung_scc` | 2 | 5,000 |

| Partition | Images | Reconstructed source-image groups | Usage |
|---|---:|---:|---|
| Training | 10,394 | 514 | Model fitting and train-only stain reference |
| Validation | 2,273 | 113 | Model selection, Grad-CAM examples, synthetic robustness |
| Held-out TEST | 2,333 | 113 | One final evaluation of frozen H3, already completed |

**Not patient-level:** Source-image groups were inferred from image relationships using the LC25000-clean metadata, not from patient identities, slide IDs, hospitals, or scanner sites. Group separation reduces known augmentation-copy leakage, but it cannot prove complete biological independence. The initial audit found no unreadable images, no group IDs spanning splits, and no exact cross-split duplicate-byte hashes.

## Immutable source files in the original Google Drive

```text
/content/drive/MyDrive/ICTIU_Final_Projects/lung/
  archives/LC25000.zip
  splits/manifest_candidate.csv
  splits/class_mapping.json
  metadata/lc25000_image_groups.csv
  results/audit.json
```

The full source archive is approximately 1.8 GiB and is **not included in Git**. Existing files are reused. If a saved manifest is missing, recover the original version rather than recomputing a random split. The original archive includes non-lung content; the three lung classes are selected through the saved manifest.

| Saved input | SHA-256 |
|---|---|
| `archives/LC25000.zip` | `382c98315435cbbf559be5da94cc0b62f696434e30e15feefebe9c892de01728` |
| `splits/manifest_candidate.csv` | `d931c9b11d03716c3c426f251bc75eaa2051002acca99d3ba66d59d964285dea` |
| `splits/class_mapping.json` | `4d14782118e3a438c13b209d7a594f2f9676d09cec25125b599d8c6d6c26a6a8` |
| `metadata/lc25000_image_groups.csv` | `2ff9e992f759f265dfc9aa2363de0dfcc8e6559884d8bb02a4c925214fdf7e23` |
| `results/audit.json` | `cd9aecfb8bfe64ca956b205e333ce9c9697dbfa88d4c46dc89c3ac088115fac0` |

These hashes originate from completed experiment source-file checks. The new setup notebook compares against them but its code has not yet been executed by the project owner.

## Reproduction boundary

The original partition assignments and label IDs must remain unchanged when reproducing any saved result. Training augmentation and fitted stain normalization use **training only**. The one-time H3 held-out TEST has already been performed and should not be rerun for documentation. Report validation and TEST results distinctly.
