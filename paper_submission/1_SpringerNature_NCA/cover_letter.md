> **DRAFT — send only after every [TBD] in manuscript.tex is filled from experiments.**

The Editor-in-Chief
*Neural Computing and Applications*, Springer Nature

Date: [TBD]

Dear Editor,

We submit the manuscript **"BrainHybridNet: A Lightweight Hybrid CNN–Transformer with Self-Supervised Pre-training and Uncertainty Estimation for Joint Brain Tumor Segmentation, Detection and Grading from Multi-Modal MRI"** for consideration as an Original Article in *Neural Computing and Applications*.

**What the paper does.** Automated brain tumor analysis usually needs separate models for segmentation, detection and grading, and most of them give no measure of confidence. BrainHybridNet is a single hybrid CNN–transformer that segments whole tumor, tumor core and enhancing tumor in 3D, detects tumor presence and predicts low- versus high-grade glioma in one forward pass. The encoder is pre-trained with 3D masked image modeling, Monte Carlo dropout gives voxel-wise uncertainty, and attention maps give spatial explanations.

**Why it fits the journal.** The work is an application of neural computing — a hybrid convolutional/attention architecture, self-supervised pre-training and approximate Bayesian inference — to a clinical imaging problem, with attention to model size and inference cost.

**Main results.** [TBD: Dice WT/TC/ET with 95% CI, parameters and latency, compared with the strongest re-trained baseline; grading AUC; calibration result. Copy from Table 1–3 only.]

**Rigor.** All five baselines (3D U-Net, SegResNet, UNETR, SwinUNETR, nnU-Net v2) were re-trained by us on identical subject-level splits of BraTS 2021; we report bootstrap confidence intervals and Holm-corrected paired tests, external validation on UCSF-PDGM, and follow the CLAIM checklist. Code, splits and weights are public.

**Manuscript details.** Original Article; [TBD] words main text; [TBD] figures; 4 tables; 47 references.

The manuscript is original, has not been published and is not under consideration elsewhere. All authors approved the submission. The authors declare no competing interests. The study uses only public, de-identified datasets.

Suggested reviewers: [TBD: 3–4 names, institutions, e-mails; none from the authors' institution and no recent co-authors].

Sincerely,

Debobrato Biswas (corresponding author)
Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh
[TBD: institutional e-mail] · ORCID: [TBD]
