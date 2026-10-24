# Lung H3 Grad-CAM (validation-only)

Six examples (two per class) were selected by a fixed SHA-256 ranking from the original source-group-separated validation partition, before model inference. The frozen MobileNetV2 `out_relu` and EfficientNetB0 `top_activation` maps (both 7x7) were differentiated separately with respect to the **same predicted-class pre-softmax logit** in the trained H3 fusion head. Heatmaps were individually normalized and overlaid on 224x224 padded images. Independent branch colors cannot be used as quantitative contribution comparisons. Results are descriptive; no test images were examined and no tuning or model training occurred. The original lung groups are reconstructed source-image groups, not patient/slide/hospital groups. Grad-CAM does not establish histological localization correctness or clinical validity.

The ratio of correct predictions within these deliberately small fixed examples is **not** the full-validation accuracy or a held-out-test metric.
