# Model Card — Web App Detection Model (`best_model.pth`)

**Status: research prototype. Not validated for clinical use. Not a diagnosis.**

## Purpose
Classifies a single 2D brain MRI slice as *tumor* or *no tumor*.

## Architecture
`BrainTumorClassifier` (`src/model.py`): SE-attention residual 3D U-Net with a classification head.
A 2D slice is repeated over depth (64) and 4 channels to form a (4, 64, 64, 64) input.
Only the detection head (`cls_head`, 2 logits) was trained. The segmentation (ET/TC/WT),
grade and tumor-type heads exist in the network but are **untrained** and are not used by the app.

## Input
- Supported: one axial 2D MRI slice, JPG/PNG, at least 32×32 px. Converted to grayscale, resized
  to 64×64, scaled to [0, 1] (no z-scoring) — `dataset.image_to_volume`.
- Accepted but **not validated**: NIfTI (3D, or 4D with T1, T1ce, T2, FLAIR in the last axis).
  The model never saw volumetric input in training.
- Screenshots, photos, CT, or non-axial slices are out of scope; the app cannot verify modality.

## Output
`softmax(cls_logits)`: class 1 = tumor, class 0 = no tumor; threshold 0.5 (argmax).
The app caps the displayed confidence of 2D results at 65%; the checkpoint has no fitted
temperature, so the probabilities are not calibrated.
No location, size, outline, grade or tumor type is predicted.

## Training data (as far as recorded)
Folder `data/train/{tumor,no_tumor}`: Kaggle "Brain Tumor MRI Dataset" Training split
(glioma + meningioma + pituitary → tumor; notumor → no tumor) plus 21 mock images.
The checkpoint was written by an older `train.py` on a GPU machine; its exact data, split and
hyper-parameters are **not recorded** in the checkpoint.

## Evaluation
Kaggle Testing split (`dummy_data/Testing`), patient identity unavailable, so evaluation is
image-level. On the 1,095 test images not pixel-identical to a training image:
accuracy 0.995, sensitivity 0.996, specificity 0.993, AUC 1.000 (3 false negatives, 2 false
positives). Reproduce with `python src/evaluate_app_model.py`. See TEST_REPORT.md.

## Limitations
- One public dataset, curated slices; no external or clinical validation; near-duplicate images
  between Training and Testing cannot be ruled out, so real-world performance is likely lower.
- No patient-level split was possible.
- A negative result does not rule out a tumor; a positive result does not say where it is.
- Small lesions, other pathologies, post-operative scans and non-axial views are not covered.

## Intended use
Research and teaching demonstrations only. Not for screening, diagnosis or treatment decisions.

---

# Model Card — 2D Segmentation / Tumor-Type Model (`models/seg2d_best.pth`)

**Status: research prototype. Not validated for clinical use. Not a diagnosis.**

## Purpose
For one 2D MRI slice: outline the tumor, classify no tumor / glioma / meningioma / pituitary
tumor. The app derives size (mm), side, depth and a rule-based risk level from the outline
(`src/pipeline/analysis_2d.py`). For 2D uploads this model also makes the tumor / no-tumor decision.

## Architecture
`TumorSegNet2D` (`src/seg2d.py`): ImageNet-pretrained ResNet34 encoder + U-Net decoder
(1-channel mask) + classification head (4 classes). Input: grayscale, zero-padded to a square,
256×256, [0, 1]. Test-time horizontal-flip averaging.

## Training (`src/train_seg2d.py`, 2026-10-25, CPU, 11 epochs)
- figshare (Cheng et al.): 3,064 T1-CE slices, 233 patients, expert masks. Patient-level split
  80/10/10 (seed 42) → 2,439 / 330 / 295 slices (`runs/seg2d/split.json`).
- Kaggle Brain Tumor MRI Dataset, Training folder: class labels only (no-tumor images use an
  empty mask). 305 images that copy figshare val/test slices were removed; 10% held out.
- Loss: BCE + Dice (mask) + 0.5 × cross-entropy (class). AdamW, one-cycle LR 3e-4, batch 12,
  flips / rotation / scale / gamma / blur / noise. Best epoch by val Dice + val class accuracy:
  epoch 11 (val Dice 0.792, Kaggle val accuracy 0.994). Epoch 12 did not run (stopped for low memory).

## Evaluation (`python src/evaluate_seg2d.py`, runs/seg2d_eval/summary.json)
figshare test patients (24 patients, 295 slices, never used in training):

| Metric | Value |
| --- | --- |
| Dice mean / median | 0.750 / 0.863 |
| Slices with Dice ≥ 0.5 | 85.8% |
| Longest diameter error (known spacing), median | 3.6 mm (8%) ; mean 9.4 mm |
| Side agreement with expert mask (axial/coronal, non-midline) | 90.4% (94/104); 100% (90/90) when the outline overlaps the tumor |
| Tumor type accuracy (3 types) | 99.3% |

Kaggle Testing split (1,311 images; 621 are copies of a training image):

| Rule | Accuracy | Sensitivity | Specificity | AUC |
| --- | --- | --- | --- | --- |
| 3D detection model (`best_model.pth`) | 0.937 | 0.923 | 0.968 | 0.985 |
| **2D model (app rule)** | **0.998** | **0.999** | **0.995** | **1.000** |
| mean of both | 0.986 | 0.989 | 0.978 | 0.999 |

Not-copied subset (690): 2D model 690/690 correct (only 28 no-tumor images in it).
4-class type accuracy 0.989. Calibration error (ECE) of the 2D model's P(tumor): 0.023, so the app
shows this probability without the former 65% cap.

## Known limitations
- Errors concentrate on subtle tumors at the skull base / posterior fossa: the outline can be in
  the wrong place (Dice 0 on 39 of 618 val+test slices). Outlines with mean probability < 0.8 are
  flagged "outline uncertain" (catches 15 of those 39, 11 false flags of 579).
- The expert masks sometimes include tissue around the enhancing rim; the model outlines the
  enhancing lesion more tightly and then reports a smaller size.
- mm scale for JPG/PNG is estimated from the head width (150 mm assumed, ±15%); slice level and
  head size change it. The figshare README spacing (0.49 mm) would imply 180 mm wide heads, so
  the true scale of those images is uncertain too.
- One slice: volume is an ellipsoid estimate; risk level is rule-based (size, type, depth,
  midline, ring pattern) and not validated against outcomes.
- Only T1-CE (figshare) and Kaggle-style JPGs seen in training; other sequences, photos of film,
  screenshots and CT are out of scope. Both datasets are public and partly overlapping; no
  external validation.
