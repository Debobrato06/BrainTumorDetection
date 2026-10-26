> **DRAFT — J-BHI asks for the cover letter in the submission text box. Send only after every [TBD] in manuscript.tex is filled from experiments.**

Dear Editor-in-Chief,

We submit the regular paper **"BrainHybridNet: A Lightweight Hybrid CNN–Transformer for Joint Brain Tumor Segmentation, Detection and Grading with Self-Supervised Pre-training and Uncertainty Estimation"** to the *IEEE Journal of Biomedical and Health Informatics*.

**Innovation and significance.** Brain tumor work-ups usually need separate models for segmentation, detection and grading, and most give no confidence estimate. We present one compact model that performs all three on multi-modal MRI, pre-trains its hybrid CNN–transformer encoder by 3D masked image modeling, and returns voxel-wise uncertainty and attention maps. [TBD: key measured result — Dice WT/TC/ET with CI versus the strongest baseline, parameters, grading AUC, calibration.] This matters to health informatics because the model is small enough for standard clinical hardware and points readers to regions it is unsure about.

**Originality.** Five baselines (3D U-Net, SegResNet, UNETR, SwinUNETR, nnU-Net v2) were re-trained on identical patient-level splits of BraTS 2021; results carry bootstrap confidence intervals and Holm-corrected paired tests, with external validation on UCSF-PDGM. The manuscript has not been published or submitted elsewhere and is not derived from a conference paper.

**Data.** Only public, de-identified datasets (BraTS 2021, UCSF-PDGM) were used, with citations; no human-subjects approval was required.

**Suggested reviewers** (from institutions other than ours): [TBD: 4 names, affiliations, e-mails].

Sincerely,
Debobrato Biswas, Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh — [TBD: institutional e-mail] — ORCID [TBD]
