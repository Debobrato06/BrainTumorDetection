# BrainHybridNet: A Lightweight Hybrid CNN-Transformer Framework with Self-Supervised Pre-training and Uncertainty Estimation for Joint Brain Tumor Classification and Segmentation from Multi-Modal MRI

**Debobrato [Last Name]¹**, [Co-author Name]¹, [Co-author Name]²

¹ Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh
² [Second Affiliation if applicable]

**Corresponding Author:** Debobrato [Last Name] — [your.email@diu.edu.bd]

---

> **DRAFT — NOT FOR SUBMISSION.** All experimental numbers are marked [TBD] and must be filled from the outputs of `src/evaluate.py` and `src/stats.py`. Earlier result tables were removed because they were not produced by experiments in this repository.

## Abstract

Brain tumor detection and grading from multi-modal Magnetic Resonance Imaging (MRI) is a critical yet challenging task in clinical neuro-oncology. Existing deep learning approaches predominantly adopt single-task architectures, often have large parameter counts, and rarely quantify uncertainty, which limits their clinical deployability. We propose **BrainHybridNet**, a lightweight hybrid Convolutional Neural Network–Vision Transformer (CNN-ViT) that jointly performs (1) voxel-level 3D segmentation of whole tumor, tumor core and enhancing tumor, (2) tumor detection, and (3) low- versus high-grade glioma prediction in a single forward pass. The encoder is pre-trained with 3D masked image modeling on unlabeled MRI, Monte Carlo (MC) Dropout provides voxel-wise epistemic uncertainty, and transformer attention provides spatial explanations. On the BraTS 2021 benchmark (1,251 subjects; T1, T1ce, T2, FLAIR), evaluated on a held-out test set and with 5-fold cross-validation against 3D U-Net, SegResNet, UNETR, SwinUNETR and nnU-Net re-trained on identical splits, BrainHybridNet achieves Dice scores of [TBD] (WT), [TBD] (TC) and [TBD] (ET) with [TBD] million parameters and [TBD] ms per crop. [TBD: grading, calibration and external-validation results.]

**Keywords:** Brain tumor detection; Multi-modal MRI; Hybrid CNN-Transformer; Vision Transformer; Self-supervised learning; Masked Autoencoder; Uncertainty estimation; Monte Carlo Dropout; Tumor segmentation; WHO grading

---

## 1. Introduction

Brain tumors are among the most aggressive and life-threatening forms of cancer, with glioblastoma multiforme (GBM, WHO Grade IV) exhibiting a median overall survival of only 14.6 months despite current standard-of-care treatment [1]. Globally, approximately 300,000 new cases of primary brain and central nervous system tumors are diagnosed annually [2], with the burden disproportionately affecting individuals in their most productive years. Early, accurate detection and precise delineation of tumor sub-regions — whole tumor (WT), tumor core (TC), and enhancing tumor (ET) — are prerequisites for surgical planning, radiotherapy targeting, and treatment response monitoring [3].

Multi-modal MRI has emerged as the gold standard for non-invasive brain tumor characterization. Each MRI modality provides complementary information: T1-weighted images highlight anatomy, T1 contrast-enhanced (T1ce) images reveal the enhancing active tumor boundary, T2-weighted images highlight peritumoral edema, and FLAIR (Fluid-Attenuated Inversion Recovery) images are sensitive to infiltrative tumor regions [4]. Manual delineation of these regions by radiologists is not only highly time-consuming — typically requiring 1–4 hours per case — but is also subject to substantial inter-observer variability (Dice agreement of 74–85%) [3]. Automated, reliable, and fast deep learning-based systems are therefore urgently needed.

Convolutional Neural Networks (CNNs), particularly U-Net and its 3D variants, have dominated medical image segmentation since 2015 [5]. While CNNs excel at capturing local spatial features through hierarchical convolution, their inherent inductive biases (translation equivariance, local receptive fields) limit their ability to model long-range spatial dependencies — crucial for capturing the global volumetric context of irregular brain tumors [6]. Transformers, originally proposed for natural language processing [7], address this limitation via the multi-head self-attention mechanism, which attends to all spatial positions simultaneously. However, pure Vision Transformer (ViT) architectures [6] demand large pre-training datasets and are computationally prohibitive for 3D volumetric medical imaging when used naively.

Furthermore, despite impressive segmentation performance, existing methods share several critical gaps:

1. **Single-task designs:** Most methods address either segmentation or classification independently, requiring separate models for grading and delineation [8].
2. **Data hunger:** Pure transformer approaches require massive labeled datasets, which are impractical in clinical settings [9].
3. **Limited uncertainty quantification:** In safety-critical clinical applications, a model must communicate *how confident* it is, yet most published segmentation methods report point predictions only [10].
4. **Lack of interpretability:** Black-box predictions are unacceptable in clinical radiology, demanding explainability mechanisms [11].
5. **Computational inefficiency:** Models with 50–100M+ parameters cannot be deployed on standard hospital-grade workstations [12].

To address all of these gaps simultaneously, we propose **BrainHybridNet**. The core insight of our design is the complementarity of CNNs (local features) and Transformers (global context) when combined in a principled hybrid architecture, coupled with self-supervised pre-training to overcome data scarcity, uncertainty estimation for safety, and multi-task learning for clinical completeness.

### 1.1 Summary of Contributions

The principal contributions of this paper are as follows:

1. **Novel Hybrid CNN-Transformer Architecture**: A ResNet-style CNN stem provides local feature extraction and serves as a high-resolution skip-connection path; a Vision Transformer bottleneck captures volumetric long-range dependencies. This design achieves a favorable local-global feature balance at a fraction of the parameter cost of pure transformer approaches.

2. **Self-Supervised Pre-training via Masked Image Modeling**: Following masked autoencoders [13] and SimMIM [14], we mask 75% of 3D patches of the input and train the hybrid encoder to reconstruct them from unlabeled MRI before supervised fine-tuning. The effect on downstream accuracy and label efficiency is measured in Section 6.

3. **Joint Multi-Task Learning**: A single model simultaneously outputs (i) binary tumor presence classification, (ii) voxel-level segmentation masks for WT/TC/ET sub-regions, and (iii) WHO grade prediction (LGG/HGG) via dedicated task-specific heads. This multi-task design improves both efficiency and individual task performance through shared representation learning.

4. **Uncertainty Estimation via MC Dropout**: At inference time, the model performs *T* stochastic forward passes with dropout active and BatchNorm frozen. The variance across predictions constitutes a voxel-wise epistemic uncertainty map; we test whether it tracks segmentation error and how it affects calibration.

5. **Transformer Attention-Based Explainability**: Spatial attention weights from the final transformer block are aggregated and overlaid on MRI slices, providing a class-agnostic saliency map that explains model focus — directly supporting radiologist trust and acceptance.

6. **Rigorous, Reproducible Comparison**: All baselines are re-trained on identical splits with identical preprocessing, results are reported with confidence intervals and paired significance tests, and code, splits and weights are released. [TBD: one sentence on the measured accuracy–efficiency trade-off.]

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the proposed BrainHybridNet framework in detail. Section 4 describes the experimental setup. Section 5 presents quantitative and qualitative results. Section 6 provides ablation studies. Section 7 discusses limitations and future work. Section 8 concludes the paper.

---

## 2. Related Work

### 2.1 CNN-Based Methods for Brain Tumor Segmentation

The U-Net architecture [5], with its symmetric encoder-decoder design and skip connections, established the foundation for medical image segmentation. Çiçek et al. [15] extended U-Net to 3D, demonstrating effectiveness on volumetric MRI. SegResNet [16], the winner of the BraTS 2018 challenge, employs 3D residual encoder blocks with variational autoencoder regularization. MONAI [17], a domain-adapted 3D CNN framework, further optimized training pipelines for medical imaging. While these methods perform competitively, they inherently cannot model long-range spatial dependencies due to CNN's local receptive field limitation.

### 2.2 Transformer-Based Methods

The seminal Vision Transformer (ViT) [6] demonstrated that pure attention-based models can match or exceed CNNs when trained at scale. TransUNet [9] incorporated ViT as a bottleneck encoder in a U-Net-like segmentation framework, achieving significant improvements on multi-organ segmentation. Swin-Transformer [18], with its shifted-window attention mechanism, improved computational efficiency and enabled hierarchical feature extraction. SwinUNETR [19] applied Swin Transformers to 3D brain tumor segmentation, achieving top-tier performance on BraTS benchmarks, and TransBTS [20] combined a 3D CNN encoder with a transformer bottleneck for BraTS. Such models require substantial GPU memory, limiting clinical accessibility.

### 2.3 Hybrid CNN-Transformer Architectures

Several works have combined CNN and transformer components to leverage their complementary strengths. MedT [21] introduced a gated axial-attention mechanism combined with CNN for medical image segmentation. UCTransNet [22] replaced standard skip connections with Channel Transformer modules. D-Former [23] proposed a dilated efficient 3D transformer for volumetric segmentation. Our BrainHybridNet differs from all these by combining (a) an explicit ResNet-style CNN stem, (b) a lightweight ViT encoder with learned 3D positional embeddings, and (c) a U-Net decoder with skip connections — in a unified multi-task framework.

### 2.4 Self-Supervised Learning in Medical Imaging

Self-supervised learning (SSL) reduces reliance on expensive annotations. Contrastive methods such as SimCLR [24] and DINO [25] have been adapted to medical imaging [26]. The Masked Autoencoder (MAE) [13] showed that high masking ratios (75%) force the encoder to learn contextually meaningful representations. Zhou et al. [27] applied MAE to 3D CT for downstream organ segmentation. Our work applies masked image modeling to multi-modal 3D MRI for brain tumor analysis, followed by multi-task fine-tuning.

### 2.5 Uncertainty Estimation in Medical AI

Uncertainty quantification (UQ) in deep learning has been studied through Bayesian methods [28], deep ensembles [29], and approximate inference via Monte Carlo Dropout (MCDO) [10]. In medical imaging, Nair et al. [30] demonstrated MCDO for uncertainty in multiple sclerosis lesion segmentation, showing that high-uncertainty regions correlate with radiologist disagreement. Roy et al. [31] applied uncertainty maps to improve MRI segmentation trustworthiness. Our work integrates MCDO into the hybrid architecture and evaluates calibration and the uncertainty–error relationship on BraTS data.

### 2.6 Brain Tumor Grading and WHO Classification

WHO classifies gliomas into four grades based on histopathological and molecular markers [32]. MRI-based non-invasive grading using deep learning has been explored by [33, 34]. Existing approaches typically treat grading as a separate classification pipeline. Our joint grading head enables simultaneous detection, segmentation, and grading within a single inference, reducing computational overhead in clinical workflows.

---

## 3. Proposed Methodology

### 3.1 Overview

Figure 1 presents an overview of BrainHybridNet. The framework consists of four components: (1) a **sequence attention and CNN stem** for local feature extraction, (2) a **Vision Transformer encoder** operating on CNN features for global context, (3) a **U-Net-style decoder** that upsamples the transformer tokens and fuses the CNN skip features, and (4) **task heads** for segmentation, tumor detection and grading. An optional **masked image modeling** stage pre-trains the encoder before supervised fine-tuning.

```
Input: multi-modal MRI crop (4 channels: T1, T1ce, T2, FLAIR), 128 x 128 x 128
  │
  ▼
[Squeeze-and-Excitation over sequences]
  ▼
[CNN stem, stage 1] ── c1 (32 ch, full resolution) ───────────────┐
  ▼                                                               │
[CNN stem, stage 2] ── c2 (64 ch, 1/2 resolution) ─────────┐      │
  ▼                                                        │      │
[Patch embedding p=8] → 512 tokens → [ViT x L=4]           │      │
  │                         │                              │      │
  │                    mean pooling                        │      │
  │                    ├─► Detection head                  │      │
  │                    └─► Grade head (LGG/HGG)            │      │
  ▼                                                        │      │
[Reshape to 8^3 grid] → [Up x2] → [Up x2] → [Up x2 + c2] ◄─┘      │
  ▼                                                               │
[Up x2 + c1] ◄────────────────────────────────────────────────────┘
  ▼
[Segmentation head] → WT / TC / ET probability maps

[MC Dropout at inference] → voxel-wise uncertainty map
[Attention weights]       → saliency for explainability
```
*Figure 1: BrainHybridNet architecture.*

### 3.2 Input Preprocessing

The input is a multi-modal MRI volume **X** ∈ ℝ^(B × 4 × D × H × W). BraTS 2021 volumes are distributed skull-stripped, co-registered to the SRI24 atlas and resampled to 1 mm isotropic resolution [35], so no additional registration is applied. For each subject and sequence, intensities are z-score normalized over non-zero (brain) voxels, and each volume is cropped to the bounding box of the brain. During training, random 128 × 128 × 128 crops are sampled; with probability 0.66 the crop is centered on a randomly chosen tumor voxel, otherwise it is placed uniformly. Inference on whole volumes uses sliding windows of the same size with 50% overlap and Gaussian weighting.

### 3.3 Sequence Attention and CNN Stem

A Squeeze-and-Excitation block re-weights the four input sequences with learned, input-dependent weights. The CNN stem then extracts local features in two stages:

```
c₁ = ReLU(BN(Conv3D(X, out=32, k=3, s=1, p=1)))    shape: (B, 32, D, H, W)
c₂ = ReLU(BN(Conv3D(c₁, out=64, k=3, s=2, p=1)))   shape: (B, 64, D/2, H/2, W/2)
```

The stem avoids max-pooling; c₁ and c₂ are kept as skip connections for the decoder.

### 3.4 Vision Transformer Encoder

#### 3.4.1 Patch Tokenization and Positional Encoding

The feature map c₂ is partitioned into non-overlapping 8 × 8 × 8 patches by a strided 3D convolution, giving N = (D/16)(H/16)(W/16) tokens of dimension E = 128 (N = 512 for a 128³ crop). Learnable positional embeddings **P** ∈ ℝ^(1 × N × E) are added:

```
z₀ = Flatten(Conv3D(c₂, out=E, k=8, s=8))ᵀ + P
```

#### 3.4.2 Transformer Blocks

The encoder stacks L = 4 pre-norm transformer blocks [7, 36] with H = 4 heads, an MLP ratio of 4 with GELU activation [37], and dropout 0.1 in attention and MLP layers:

```
zₗ' = zₗ + MHSA(LN(zₗ))
zₗ₊₁ = zₗ' + MLP(LN(zₗ'))
```

A final LayerNorm yields **z_L** ∈ ℝ^(B × N × E).

#### 3.4.3 Masked Image Modeling Pre-training

For self-supervised pre-training we use masked image modeling in the style of SimMIM [14], adapted to 3D. A fraction ρ = 0.75 of the N token positions is masked at random. The corresponding 16 × 16 × 16 regions of the input are set to zero before the CNN stem, so masked content cannot leak through the convolutions, and the masked tokens are replaced by a learned [MASK] embedding before the transformer. A linear head maps each output token to the 4 × 16³ voxels of its input region. The loss is the mean squared error between predictions and per-patch normalized targets, averaged over masked patches only:

```
L_MIM = (1 / |M|) Σ_{i ∈ M} ‖x̂ᵢ − norm(xᵢ)‖² / (4 · 16³)
```

Pre-training uses only the training folds of each cross-validation split; the linear reconstruction head is discarded before fine-tuning.

### 3.5 Decoder

The tokens are reshaped into a volume z_vol ∈ ℝ^(B × E × D/16 × H/16 × W/16) and passed through 3D dropout (rate 0.1). Three upsampling stages (transposed convolution, stride 2, followed by a residual SE block) restore the resolution of c₂, where c₂ is concatenated before the third residual block. A fourth stage upsamples to full resolution and fuses c₁:

```
u = Up(z_vol); u = Up(u); u = ResSE(cat[Up(u), c₂])     →  (B, 64, D/2, H/2, W/2)
d = ResSE(cat[Up(u), c₁])                               →  (B, 32, D, H, W)
```

### 3.6 Task Heads

- **Segmentation:** a 1 × 1 × 1 convolution outputs three channels (WT, TC, ET), each with an independent sigmoid.
- **Detection:** mean-pooled tokens z_global = mean(z_L) feed Linear(E, 128) → ReLU → Dropout(0.1) → Linear(128, 2). Because BraTS contains no tumor-free subjects, detection is trained at crop level: a crop is labeled positive when it contains at least 100 whole-tumor voxels.
- **Grading:** Linear(E, 64) → ReLU → Dropout(0.1) → Linear(64, 2) predicts LGG versus HGG. Subjects without a grade label are ignored by this loss.

### 3.7 Uncertainty Estimation via Monte Carlo Dropout

At inference, BatchNorm layers stay in evaluation mode while all dropout layers are kept active [10]. T = 20 stochastic sliding-window passes give

```
μ_seg = (1/T) Σₜ σ(Ŷ_seg⁽ᵗ⁾),     σ²_seg = (1/T) Σₜ (σ(Ŷ_seg⁽ᵗ⁾) − μ_seg)²
```

where σ²_seg is the voxel-wise epistemic uncertainty map used to flag regions for review.

### 3.8 Training Objective

```
L_total = λ₁ · L_seg + λ₂ · L_det + λ₃ · L_grade,   λ₁ = 0.6, λ₂ = 0.2, λ₃ = 0.2
```

L_det and L_grade are cross-entropy losses. L_seg is the Tversky loss [38], computed per region channel and averaged:

```
TL = 1 − (TP + ε) / (TP + α·FP + β·FN + ε),   α = 0.3, β = 0.7
```

Since β > α, false negatives are penalized more than false positives, which favors recall of tumor tissue. The loss weights were fixed before training; their sensitivity is reported in Section 6 [TBD].

### 3.9 Optimization

- **Optimizer:** AdamW [39], learning rate 1 × 10⁻⁴, weight decay 1 × 10⁻⁴, gradient clipping at norm 1.0
- **Schedule:** 10 warm-up epochs, then cosine decay [40]; 300 fine-tuning epochs, 100 pre-training epochs (learning rate 1.5 × 10⁻⁴)
- **Batch size:** 2 crops of 128³ per GPU, mixed precision (FP16)
- **Augmentation:** random flips along each axis (p = 0.5), per-sequence intensity scaling (0.9–1.1) and shift (±0.1), Gaussian noise (σ = 0.01, p = 0.2)
- **Model selection:** the checkpoint with the highest mean validation Dice of each fold

---

## 4. Experimental Setup

### 4.1 Datasets

**BraTS 2021 training set** [35]: 1,251 subjects with expert segmentations and four co-registered sequences (T1, T1ce, T2, FLAIR), 240 × 240 × 155 voxels at 1 mm³. Labels are background (0), necrotic core (1), peritumoral edema (2) and enhancing tumor (4), from which whole tumor (WT = 1 + 2 + 4), tumor core (TC = 1 + 4) and enhancing tumor (ET = 4) are derived.

**Split.** A fixed, subject-level hold-out test set of 15% of subjects ([TBD: n] subjects) is used only once, for the final results. The remaining subjects are divided into 5 cross-validation folds; every model, baseline and ablation uses the same folds (split file released with the code).

**Grading data.** BraTS 2021 does not provide histological grade labels. Grading is trained and evaluated on [TBD: UCSF-PDGM [41], which reports WHO grade for preoperative diffuse gliomas, with its own subject-level split], and UCSF-PDGM also serves as an external test set for segmentation without retraining.

### 4.2 Evaluation Metrics

- **Segmentation:** Dice similarity coefficient and 95th-percentile Hausdorff distance (HD95, mm) per region, with sensitivity and specificity. Following BraTS conventions, Dice = 1 and HD95 = 0 when prediction and reference are both empty, and HD95 = 373.13 mm when only one is empty.
- **Detection and grading:** AUC-ROC, accuracy, balanced accuracy, sensitivity and specificity.
- **Calibration and uncertainty:** voxel-level expected calibration error (ECE, 15 bins) and reliability diagrams [42]; Spearman correlation between case-level mean uncertainty and segmentation error (1 − Dice).
- **Efficiency:** parameters, multiply-accumulate operations, median latency per 128³ crop after warm-up, and peak GPU memory, all on [TBD: GPU model].
- **Statistics:** mean ± standard deviation across cases and 95% bootstrap confidence intervals (10,000 resamples). Paired comparisons with the strongest baseline use the Wilcoxon signed-rank test over test subjects with Holm correction across metrics.

The reporting follows the CLAIM checklist [43] (supplementary material).

### 4.3 Implementation Details

All models were implemented in PyTorch [44] with PyTorch Lightning [45]; baselines use their MONAI [17] implementations. Experiments ran on [TBD: hardware]. Pre-training took [TBD] hours and fine-tuning [TBD] hours per fold. Code, configuration files, split files and trained weights are available at [TBD: repository URL].

### 4.4 Baselines

All baselines were re-trained by us with the same splits, crop size, preprocessing, augmentation, loss and training schedule as BrainHybridNet; no numbers are copied from other publications.

- **3D U-Net** [15]: CNN-only encoder-decoder
- **SegResNet** [16]: residual encoder, BraTS 2018 winner
- **UNETR** [46]: pure ViT encoder with a CNN decoder
- **SwinUNETR** [19]: Swin Transformer encoder
- **nnU-Net v2** [47]: self-configuring U-Net, trained with its default pipeline on the same folds

---

## 5. Results

> All values in this section are placeholders ([TBD]) to be filled from `src/evaluate.py` and `src/stats.py` outputs.

### 5.1 Segmentation

**Table 1: Segmentation on the BraTS 2021 hold-out test set ([TBD] subjects), mean ± SD**

| Method | WT Dice↑ | TC Dice↑ | ET Dice↑ | WT HD95↓ | TC HD95↓ | ET HD95↓ | Params (M) | GMACs | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 3D U-Net [15] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SegResNet [16] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| UNETR [46] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SwinUNETR [19] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| nnU-Net v2 [47] | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| BrainHybridNet (ours) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*Significance versus the strongest baseline (Wilcoxon, Holm-corrected): [TBD].*

**Table 2: Five-fold cross-validation, mean ± SD of fold means** — [TBD]

### 5.2 Detection and Grading

**Table 3: Crop-level tumor detection (BraTS test set) and LGG/HGG grading ([TBD] grading test set)**

| Method | Detection AUC | Sensitivity | Specificity | Grade AUC | Grade balanced accuracy |
|---|---|---|---|---|---|
| BrainHybridNet (ours) | TBD | TBD | TBD | TBD | TBD |

### 5.3 Calibration and Uncertainty

[TBD: voxel ECE of BrainHybridNet with and without MC Dropout, compared with the baselines; reliability diagram; Spearman ρ between case-level uncertainty and 1 − Dice with p-value.]

### 5.4 External Validation

[TBD: segmentation results on UCSF-PDGM without retraining.]

### 5.5 Qualitative Results

[TBD: overlays of predictions and references, uncertainty maps and attention maps for representative good and failure cases.]

---

## 6. Ablation Study

**Table 4: Ablations, 5-fold cross-validation on the development set**

| Configuration | Config file | WT Dice | TC Dice | ET Dice | Params (M) |
|---|---|---|---|---|---|
| CNN only (no transformer blocks) | `abl_cnn_only.yaml` | TBD | TBD | TBD | TBD |
| Transformer path only (no CNN skips) | `abl_no_skips.yaml` | TBD | TBD | TBD | TBD |
| No sequence attention | `abl_no_se.yaml` | TBD | TBD | TBD | TBD |
| Segmentation only (no multi-task) | `abl_no_multitask.yaml` | TBD | TBD | TBD | TBD |
| No self-supervised pre-training | `hybrid.yaml` without pre-training | TBD | TBD | TBD | TBD |
| Full BrainHybridNet | `hybrid.yaml` + pre-training | TBD | TBD | TBD | TBD |

[TBD: masking ratio (0.25 / 0.5 / 0.75 / 0.9), number of transformer layers (2 / 4 / 6 / 8) and loss-weight sensitivity, each with three seeds.]

---

## 7. Discussion

### 7.1 Clinical Implications

[TBD after results.] A joint model that segments, detects and grades in one pass could shorten reading time if its accuracy matches dedicated models; the uncertainty map is intended to direct radiologist review to ambiguous regions, which must be shown by the correlation analysis in Section 5.3.

### 7.2 Self-Supervised Pre-training and Data Efficiency

[TBD after results: whether masked image modeling improves fine-tuning, and by how much, including a reduced-label experiment (e.g. 10% and 25% of labeled subjects).]

### 7.3 Explainability

[TBD: attention maps must be evaluated quantitatively against reference masks (e.g. pointing game or overlap), not only shown.]

### 7.4 Limitations

1. **External validation:** generalization beyond BraTS is limited to [TBD] external data.
2. **Molecular markers:** WHO CNS5 classification uses IDH and other molecular markers [32]; our grading head uses imaging only.
3. **Pre-training scale:** pre-training uses only BraTS training folds; larger unlabeled corpora may help.
4. **Detection labels:** BraTS has no tumor-free subjects, so detection is evaluated at crop level, not on healthy scans.
5. **Annotation quality:** BraTS labels have known inter-rater variability.

### 7.5 Future Work

Future directions include federated training across hospitals, longitudinal response assessment, pediatric tumors, robustness to missing sequences, fusion with molecular data, and a prospective reader study.

---

## 8. Conclusion

We presented BrainHybridNet, a lightweight hybrid CNN-Transformer for joint brain tumor segmentation, detection and grading from multi-modal MRI, with masked image modeling pre-training and MC Dropout uncertainty. [TBD: one-sentence summary of the measured accuracy, efficiency and calibration relative to the baselines.] Code, configurations, splits and weights are publicly available to reproduce every reported number.

---

## Declarations

**Author Contributions:** D.L.: Conceptualization, Methodology, Software, Validation, Writing — Original Draft. [Co-authors]: Review & Editing, Visualization.

**Conflicts of Interest:** The authors declare no conflicts of interest.

**Ethics Approval:** This study uses publicly available, fully de-identified datasets (BraTS 2021). No institutional ethics approval was required.

**Informed Consent:** Not applicable (retrospective analysis of public dataset).

**Funding:** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Data Availability:** The BraTS 2021 dataset is publicly available at [https://www.synapse.org/#!Synapse:syn27046444](https://www.synapse.org/#!Synapse:syn27046444). [TBD: UCSF-PDGM access statement.]

**Code Availability:** Code, configuration files, data splits and trained weights are available at [TBD: repository URL].

---

## References

[1] Stupp R, Mason WP, van den Bent MJ, et al. (2005) Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma. *N Engl J Med* 352:987–996. https://doi.org/10.1056/NEJMoa043330

[2] Ostrom QT, Cioffi G, Gittleman H, et al. (2019) CBTRUS statistical report: primary brain and other central nervous system tumors diagnosed in the United States in 2012–2016. *Neuro-Oncology* 21:v1–v100. https://doi.org/10.1093/neuonc/noz150

[3] Menze BH, Jakab A, Bauer S, et al. (2015) The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE Trans Med Imaging* 34(10):1993–2024. https://doi.org/10.1109/TMI.2014.2377694

[4] Gordillo N, Montseny E, Sobrevilla P (2013) State of the art survey on MRI brain tumor segmentation. *Magn Reson Imaging* 31(8):1426–1438. https://doi.org/10.1016/j.mri.2013.05.002

[5] Ronneberger O, Fischer P, Brox T (2015) U-Net: Convolutional networks for biomedical image segmentation. In: *MICCAI 2015*. LNCS 9351:234–241. Springer. https://doi.org/10.1007/978-3-319-24574-4_28

[6] Dosovitskiy A, Beyer L, Kolesnikov A, et al. (2021) An image is worth 16×16 words: Transformers for image recognition at scale. In: *ICLR 2021*. arXiv:2010.11929

[7] Vaswani A, Shazeer N, Parmar N, et al. (2017) Attention is all you need. In: *NeurIPS 2017*. arXiv:1706.03762

[8] Havaei M, Davy A, Warde-Farley D, et al. (2017) Brain tumor segmentation with deep neural networks. *Med Image Anal* 35:18–31. https://doi.org/10.1016/j.media.2016.05.004

[9] Chen J, Lu Y, Yu Q, et al. (2021) TransUNet: Transformers make strong encoders for medical image segmentation. arXiv:2102.04306

[10] Gal Y, Ghahramani Z (2016) Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In: *ICML 2016*. arXiv:1506.02142

[11] Litjens G, Kooi T, Bejnordi BE, et al. (2017) A survey on deep learning in medical image analysis. *Med Image Anal* 42:60–88. https://doi.org/10.1016/j.media.2017.07.005

[12] Ker J, Wang L, Rao J, Lim T (2018) Deep learning applications in medical image analysis. *IEEE Access* 6:9375–9389. https://doi.org/10.1109/ACCESS.2017.2788044

[13] He K, Chen X, Xie S, et al. (2022) Masked autoencoders are scalable vision learners. In: *CVPR 2022*. arXiv:2111.06377

[14] Xie Z, Zhang Z, Cao Y, et al. (2022) SimMIM: A simple framework for masked image modeling. In: *CVPR 2022*. arXiv:2111.09886

[15] Çiçek Ö, Abdulkadir A, Lienkamp SS, Brox T, Ronneberger O (2016) 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In: *MICCAI 2016*. LNCS 9901:424–432. https://doi.org/10.1007/978-3-319-46723-8_49

[16] Myronenko A (2018) 3D MRI brain tumor segmentation using autoencoder regularization. In: *BraTS Challenge at MICCAI 2018*. arXiv:1810.11654

[17] Cardoso MJ, Li W, Brown R, et al. (2022) MONAI: An open-source framework for deep learning in healthcare. arXiv:2211.02701

[18] Liu Z, Lin Y, Cao Y, et al. (2021) Swin transformer: Hierarchical vision transformer using shifted windows. In: *ICCV 2021*. arXiv:2103.14030

[19] Tang Y, Yang D, Li W, et al. (2022) Self-supervised pre-training of Swin transformers for 3D medical image analysis. In: *CVPR 2022*. arXiv:2111.14791

[20] Wang W, Chen C, Ding M, et al. (2022) TransBTS: Multimodal brain tumor segmentation using transformer. In: *MICCAI 2021*. arXiv:2103.04430

[21] Valanarasu JMJ, Oza P, Hacihaliloglu I, Patel VM (2021) Medical transformer: Gated axial-attention for medical image segmentation. In: *MICCAI 2021*. arXiv:2102.10662

[22] Wang H, Cao P, Wang J, Zaiane OR (2022) UCTransNet: Rethinking the skip connections in U-Net from a channel-wise perspective with transformer. In: *AAAI 2022*. arXiv:2109.04335

[23] Zhang Y, et al. (2023) D-Former: A U-shaped dilated transformer for 3D medical image segmentation. *Neural Comput Appl* 35:1931–1942. https://doi.org/10.1007/s00521-022-07859-1

[24] Chen T, Kornblith S, Norouzi M, Hinton G (2020) A simple framework for contrastive learning of visual representations. In: *ICML 2020*. arXiv:2002.05709

[25] Caron M, Touvron H, Misra I, et al. (2021) Emerging properties in self-supervised vision transformers. In: *ICCV 2021*. arXiv:2104.14294

[26] Sowrirajan H, Yang J, Ng AY, Rajpurkar P (2021) MoCo-CXR: MoCo pretraining improves representation and transferability of chest X-ray models. arXiv:2101.02149

[27] Zhou HY, Lu C, Yang S, Han X, Yu Y (2022) Preservational learning improves self-supervised medical image models by reconstructing diverse contexts. In: *ICCV 2021*. arXiv:2109.04379

[28] Neal RM (1996) *Bayesian Learning for Neural Networks*. Springer. https://doi.org/10.1007/978-1-4612-0745-0

[29] Lakshminarayanan B, Pritzel A, Blundell C (2017) Simple and scalable predictive uncertainty estimation using deep ensembles. In: *NeurIPS 2017*. arXiv:1612.01474

[30] Nair T, Precup D, Arnold DL, Arbel T (2020) Exploring uncertainty measures in deep networks for multiple sclerosis lesion detection and segmentation. *Med Image Anal* 59:101557. https://doi.org/10.1016/j.media.2019.101557

[31] Roy AG, Conjeti S, Navab N, Wachinger C (2019) Inherent brain segmentation quality control from fully ConvNet Monte Carlo sampling. In: *MICCAI 2018*. LNCS 11070. https://doi.org/10.1007/978-3-030-00928-1_46

[32] Louis DN, Perry A, Wesseling P, et al. (2021) The 2021 WHO classification of tumors of the central nervous system: A summary. *Neuro-Oncology* 23(8):1231–1251. https://doi.org/10.1093/neuonc/noab106

[33] Zhou M, Scott J, Chaudhuri B, et al. (2018) Radiomics in brain tumor: Image assessment, quantitative feature descriptors, and machine-learning approaches. *Am J Neuroradiol* 39(2):208–216. https://doi.org/10.3174/ajnr.A5391

[34] Chang K, Bai HX, Zhou H, et al. (2018) Residual convolutional neural network for the determination of IDH status in low- and high-grade gliomas from MR imaging. *Clin Cancer Res* 24(5):1073–1081. https://doi.org/10.1158/1078-0432.CCR-17-2236

[35] Baid U, Ghodasara S, Mohan S, et al. (2021) The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv:2107.02314

[36] Ba JL, Kiros JR, Hinton GE (2016) Layer normalization. arXiv:1607.06450

[37] Hendrycks D, Gimpel K (2016) Gaussian error linear units (GELUs). arXiv:1606.08415

[38] Salehi SSM, Erdogmus D, Gholipour A (2017) Tversky loss function for image segmentation using 3D fully convolutional deep networks. In: *MICCAI Workshop 2017*. arXiv:1706.05721

[39] Loshchilov I, Hutter F (2019) Decoupled weight decay regularization. In: *ICLR 2019*. arXiv:1711.05101

[40] Loshchilov I, Hutter F (2017) SGDR: Stochastic gradient descent with warm restarts. In: *ICLR 2017*. arXiv:1608.03983

[41] Calabrese E, Villanueva-Meyer JE, Rudie JD, et al. (2022) The University of California San Francisco preoperative diffuse glioma MRI dataset. *Radiology AI* 4(6):e220058. https://doi.org/10.1148/ryai.220058

[42] Guo C, Pleiss G, Sun Y, Weinberger KQ (2017) On calibration of modern neural networks. In: *ICML 2017*. arXiv:1706.04599

[43] Mongan J, Moy L, Kahn CE Jr (2020) Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A guide for authors and reviewers. *Radiol Artif Intell* 2(2):e200029. https://doi.org/10.1148/ryai.2020200029

[44] Paszke A, Gross S, Massa F, et al. (2019) PyTorch: An imperative style, high-performance deep learning library. In: *NeurIPS 2019*. arXiv:1912.01703

[45] Falcon W (2019) PyTorch Lightning. GitHub. https://github.com/Lightning-AI/pytorch-lightning

[46] Hatamizadeh A, Tang Y, Nath V, et al. (2022) UNETR: Transformers for 3D medical image segmentation. In: *WACV 2022*. arXiv:2103.10504

[47] Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH (2021) nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nat Methods* 18:203–211. https://doi.org/10.1038/s41592-020-01008-z

---

*Journal Target: Neural Computing and Applications — Springer Nature*
