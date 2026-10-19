# Interpretation and limitations

1. The LC25000 image dataset contains augmented and related images. Reconstructed source-image groups were separated across train, validation and TEST, but the available groups are **not patient, slide or institution identifiers**. Biological and deployment leakage risks cannot be ruled out entirely.
2. The three-class model is not tested against an independent clinical center, scanner population or biopsy workflow. Reported model scores should not be interpreted as clinical probabilities or diagnostic safety guarantees.
3. Validation comparisons are based on a fixed dataset split and a fixed seed. No claim of statistically significant model superiority is made.
4. H4 stain shifts are synthetic optical-density perturbations with a canonical H&E-inspired basis. They do not prove generalization to real laboratory stain variation. No isolated augmentation-only condition was measured.
5. H4-NA's smaller relative performance drop does not mean it was the highest absolute-performing model under shifts. The frozen H3 pipeline was selected using validation-based evidence.
6. Grad-CAM highlights model-sensitive regions for selected examples, not causal medical localization or histopathological ground truth.
7. The held-out test was conducted once for selected H3. Retuning after TEST or repeatedly querying TEST would compromise the original evaluation design.
8. Reproducing training on a different TensorFlow, Keras, GPU or CUDA environment may produce differing trained weight bytes or scores despite the same immutable split and seed.
