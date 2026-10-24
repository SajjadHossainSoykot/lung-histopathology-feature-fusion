# Frozen H3 Deep Feature Fusion: Final Held-out TEST

**Evaluation:** One planned pass on 2,333 held-out lung histopathology images from 113 reconstructed source-image groups (not patients).

- TEST accuracy: **96.2280%**
- TEST macro-precision: **0.962412**
- TEST macro-recall: **0.961992**
- TEST macro-F1: **0.961961**
- TEST correct: **2245/2333**
- Model: frozen H3 concatenated pooled-feature fusion selected on validation.
- No test-based model selection, stain-method changes, or threshold optimization.
- Earlier H1/H2/H3/H4 comparisons are VALIDATION only.
- No external-hospital or patient-level independence is demonstrated.

Softmax scores are not calibrated clinical probabilities.
