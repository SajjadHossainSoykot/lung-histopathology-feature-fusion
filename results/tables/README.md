# Aggregate table provenance

The lightweight CSV tables are faithful summaries of already reported saved notebook outputs. They contain no individual training/test image paths or raw predictions. They were **not** produced by re-running models.

- `validation_model_comparison.csv`: H1/H2/H3 and the two H4 ablation heads, all **validation**.
- `h4_synthetic_shift_validation.csv`: H3/H4-N/H4-NA on original and predetermined synthetic-shift **validation** images.
- `h3_clean_validation_confusion_matrix.csv`: H3 clean **validation** classification counts.
- `final_test_selected_h3.csv`: measured macro metrics from the **one-time** H3 held-out TEST. Per-class TEST scores have not been invented here.

Check `notebooks/` and the original Google Drive evidence folders for the full machine-generated histories and predictions.
