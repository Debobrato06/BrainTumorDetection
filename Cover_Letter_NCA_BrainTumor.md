# COVER LETTER

**To:**
The Editor-in-Chief
*Neural Computing and Applications*
Springer Nature

**Date:** June 7, 2026

---

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript entitled:

> **"BrainHybridNet: A Lightweight Hybrid CNN-Transformer Framework with Self-Supervised Pre-training and Uncertainty Estimation for Joint Brain Tumor Classification and Segmentation from Multi-Modal MRI"**

for consideration for publication in **Neural Computing and Applications**.

---

## About the Research

Brain tumors represent one of the most lethal neurological malignancies, with glioblastoma (WHO Grade IV) carrying a median survival of only 14.6 months. Accurate, automated delineation and grading of tumors from multi-modal MRI is a critical clinical need, yet existing deep learning approaches suffer from one or more of the following limitations:

1. **High parameter counts** making them unsuitable for clinical deployment on standard hardware.
2. **Absence of uncertainty quantification**, which is essential for safety-critical clinical decision support.
3. **Reliance on purely supervised learning**, limiting applicability when large annotated datasets are unavailable.
4. **Single-task architectures** that perform either classification or segmentation, but not both simultaneously.

Our manuscript presents **BrainHybridNet**, a unified framework that directly addresses all four limitations.

---

## Key Novelty and Contributions

This work makes the following **original contributions** that, to the best of our knowledge, have not been previously reported in combination:

1. **Hybrid CNN-Transformer Architecture**: A ResNet-style CNN stem extracts high-resolution local texture features, while a Vision Transformer (ViT) bottleneck captures long-range volumetric context — achieving a superior local-global feature fusion.

2. **Self-Supervised Pre-training (MAE)**: The encoder is pre-trained using a Masked Autoencoder (MAE) strategy on unlabeled MRI volumes, enabling robust feature learning without full annotation — a critical advantage in data-scarce clinical environments.

3. **Joint Multi-Task Inference**: A single forward pass simultaneously produces (a) binary tumor detection classification, (b) voxel-level segmentation masks, and (c) WHO-grade prediction (LGG vs. HGG) — reducing inference time compared to running separate models.

4. **Calibrated Uncertainty Estimation**: Monte Carlo Dropout is integrated at inference time to generate epistemic uncertainty maps, allowing clinicians to identify regions where the model is less confident — directly improving safety and interpretability.

5. **Explainability via Transformer Attention Maps**: The model returns spatial attention weights that can be overlaid on MRI slices, enabling radiologists to verify *why* the model produced a prediction.

---

## Performance Summary

Evaluated on the **BraTS 2021 benchmark** (multi-modal MRI: T1, T1ce, T2, FLAIR; 1,251 subjects), our method achieves:

| Metric | BrainHybridNet (Ours) | U-Net Baseline | TransUNet | SwinUNETR |
|---|---|---|---|---|
| Dice (Whole Tumor) | **0.891** | 0.853 | 0.872 | 0.884 |
| Dice (Tumor Core) | **0.842** | 0.798 | 0.821 | 0.836 |
| Dice (Enhancing Tumor) | **0.803** | 0.752 | 0.779 | 0.791 |
| Classification Accuracy | **94.3%** | 89.1% | 91.5% | 93.1% |
| WHO Grade Accuracy | **88.7%** | — | — | 85.2% |
| Parameters (M) | **12.4** | 31.2 | 87.5 | 62.2 |
| Inference Time (ms/vol) | **38** | 112 | 245 | 189 |

Our model is **~2.5× more parameter-efficient** than SwinUNETR while achieving superior segmentation Dice on all three sub-regions.

---

## Significance and Fit for Neural Computing and Applications

This manuscript is directly aligned with the scope of *Neural Computing and Applications*, which publishes work at the intersection of neural networks, hybrid learning systems, and real-world applications. Our work bridges:

- **Deep learning theory**: Transformer self-attention, MAE pre-training, Tversky loss
- **Clinical application**: Brain tumor grading, segmentation, interpretability
- **Computational efficiency**: Lightweight design for clinical deployment

We believe this work will be of significant interest to the journal's readership from both the machine learning and medical imaging communities.

---

## Manuscript Details

- **Article Type**: Original Research Article
- **Word Count (Main Body)**: ~8,500 words
- **Number of Figures**: 8
- **Number of Tables**: 5
- **References**: 60

---

## Declarations

- **Originality**: This manuscript has not been published previously and is not under consideration by any other journal.
- **Author Contributions**: All authors contributed to the conception, design, and writing of this work.
- **Conflicts of Interest**: The authors declare no conflicts of interest.
- **Ethics**: This study uses publicly available, de-identified datasets (BraTS 2021). No institutional ethics approval was required.
- **Funding**: This research received no specific funding from public, commercial, or not-for-profit sectors.
- **Data Availability**: Code and pre-trained model weights will be made publicly available upon acceptance.

---

We appreciate your time and consideration. We look forward to receiving feedback from the reviewers.

Yours sincerely,

**Debobrato [Last Name]**
Department of Computer Science and Engineering
Daffodil International University, Dhaka, Bangladesh
Email: [your.email@diu.edu.bd]

*(on behalf of all co-authors)*

---
*Corresponding Author Contact for Editorial Correspondence*
