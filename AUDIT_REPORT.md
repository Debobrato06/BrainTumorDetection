# Audit Report — Brain Tumor Detection Web App

Date: 2026-10-24 · Branch: `fix/research-gap-issues` · Scope: the Flask web app (`app.py`) and the
model it serves (`best_model.pth`). The 3D research pipeline (`src/training_pipeline.py`,
BrainHybridNet) was only checked where the app depends on it.

## 1. Architecture and data flow

```
Upload (templates/index.html, static/js/main.js)
  -> POST /predict (app.py)
  -> 2D: dataset.image_to_volume  -> (1, 4, 64, 64, 64), grayscale [0, 1], slice repeated over depth and channels
     NIfTI: app.load_nifti_for_app -> (1, 4, 64, 64, 64), RAS-reoriented, z-scored, trilinear resize
  -> BrainTumorClassifier (src/model.py, SE 3D U-Net)
       cls_head  (2 logits: 0 = no tumor, 1 = tumor)     <- trained
       seg_head  (3 sigmoid channels: ET, TC, WT)        <- NOT trained
       grade_head (2), type_head (5)                     <- NOT trained
  -> softmax / temperature -> class + confidence
  -> pipeline_validator (gating) -> measurements / radiomics / risk (rule-based, need a mask)
  -> report_generator -> JSON -> main.js renders it
```

| Item | Finding |
| --- | --- |
| Backend / frontend | Flask 3.1 + Jinja template, vanilla JS, Chart.js |
| Language / deps | Python 3.12, torch 2.5.1 (CPU build on this machine), see `requirements.txt` |
| Checkpoint | `best_model.pth`, 204 tensors, loads with `strict=True`; keys `epoch=19, model_state_dict, optimizer_state_dict, img_size=(64,64,64)`; **no** `trained_heads`, `temperature` or `val_acc` (written by an older `train.py`) |
| Training script | `src/train.py`: cross-entropy on `cls_head` only, data `data/train/{tumor,no_tumor}` (2D JPG/PNG) |
| Task actually supported | **2D image classification (tumor / no tumor)**. No segmentation, grading or type model is trained. |
| Data on disk | `data/train`: 4,133 tumor + 1,600 no-tumor 2D images (Kaggle "Brain Tumor MRI Dataset" Training split plus 21 mock images). `dummy_data/Testing`: Kaggle Testing split, 1,311 images, 4 classes. No segmentation masks exist for any 2D data; BraTS is not present locally. |

## 2. Confirmed defects (with evidence)

**D1 — Critical: "tumor" overlay drawn from an untrained segmentation head, then fabricated.**
`app.py` used `seg_logits` whenever the model ran, without checking that the segmentation head was
trained (grade and type heads *were* gated by `MODEL_TRAINED_HEADS`). Observed on
`Te-gl_0100.jpg` (glioma) and `Te-no_0100.jpg`: the thresholded mask covered 77% and 90% of the
volume. The code then (a) retried at threshold 0.05, (b) discarded masks > 50% coverage,
(c) ran an intensity/asymmetry heuristic (`extract_tumor_mask_from_heuristic`), and (d) as a last
resort drew a disk around the brightest pixel and labelled concentric rings ET/TC/WT.
Result for the glioma image: a red/yellow/green "tumor" on the **skull at the top of the head**
(baseline overlay saved during the audit), with type "Meningioma", location "Frontal Lobe" and risk
"Moderate", all derived from that invented mask.

**D2 — Critical (in committed HEAD; already fixed in the working tree before this audit):
classifier fed the wrong preprocessing.** Training uses grayscale scaled to [0, 1]; HEAD's
app z-scored the image. Measured on 100 test images: z-scored input gives sensitivity **0.00**
(every image "no tumor"), [0, 1] input gives 1.00 / 1.00. HEAD also set
`is_tumor = model or heuristic`; the asymmetry heuristic calls **68%** of tumor-free test images
tumors (60 per class: glioma 0.85, meningioma 0.97, pituitary 0.88, notumor 0.68). So HEAD's
positives came essentially from the heuristic.

**D3 — High: derived outputs presented without a basis.** Tumor type, location, radiomics, risk
score and midline shift were computed from the D1 mask; the report text claimed findings the
model never assesses ("gray-white matter differentiation is preserved", "no hemorrhage",
"focal lesion in the N/A, measuring 0.0 mL").

**D4 — High (NIfTI path): wrong axis conventions.** NIfTI arrays are (X, Y, Z) but the overlay,
`save_slice_b64` and `measurement.py` treat axis 0 as the slice axis: the app showed a sagittal
slice labelled "AXIAL" and applied slice thickness to the left-right axis. Volumes were also
computed with the original voxel spacing on a mask resized to 64³ (wrong units). No reorientation
to a standard orientation was done.

**D5 — Medium: 2D grid mismatch.** On CPU, the image used for the mask/overlay
(`process_2d_image`, 32³) and the segmentation output (64³) were on different grids; the overlay
was also rendered from the 32×32 thumbnail.

**D6 — Medium: robustness.**
- CPU memory: the full forward pass at 64³ (decoder included) crashed the process with an
  oneDNN "could not create a primitive" error followed by an access violation when free RAM was low
  (0.9 GB free of 15.8 GB during the audit); the decoder output was unused anyway.
- One shared temp filename (`data/temp_upload.png`) for all requests in a threaded server.
- No validation: any extension was saved as `.png`; corrupt files produced a 500 with a traceback.
- With no model, a 2D image still got a determinate heuristic result.
- MC-Dropout ran 15 full forward passes on every positive although the segmentation path has no
  dropout (always returned "not computed").

## 3. Things checked and found correct

- Checkpoint ↔ architecture: all 204 keys match, `strict=True` load succeeds.
- `model.eval()` and `torch.no_grad()` are used; softmax over the 2-class `cls_head` + argmax
  matches the cross-entropy training.
- BraTS label mapping in `src/brats_dataset.py` (`{1,2,4|3}` → WT/TC/ET) is correct; note the
  research pipeline orders channels WT, TC, ET while the app's `seg_head` assumes ET, TC, WT —
  this matters only if a segmentation-trained `BrainTumorClassifier` is ever wired in.
- Detection accuracy of the classifier on held-out data (Section 4).

## 4. Detection performance of the existing model (measured, not quoted)

`python src/evaluate_app_model.py` on `dummy_data/Testing` with the app's preprocessing, threshold 0.5:

| Subset | n (tumor / no tumor) | Accuracy | Sensitivity | Specificity | AUC |
| --- | --- | --- | --- | --- | --- |
| All Testing images | 1,311 (906 / 405) | 0.9962 | 0.9967 | 0.9951 | 1.0000 |
| Not pixel-identical to any `data/train` image | 1,095 (795 / 300) | 0.9954 | 0.9962 | 0.9933 | 1.0000 |

216 Testing images are exact pixel duplicates of training images (known issue of this dataset)
and are reported separately. Errors: 3 meningiomas missed (`Te-me_0064/0110/0145`), 2 tumor-free
images flagged (`Te-no_0105/0262`). Caveats: the checkpoint's training data and split are not
recorded, near-duplicates (not pixel-identical) cannot be excluded, and this is one public
dataset — no external validation.

**Conclusion:** the earliest point where behaviour diverged from expectation was preprocessing
(D2, already corrected in the working tree); the remaining "wrong detection" users saw was the
localization and everything derived from it (D1, D3), which the model cannot provide.

## 5. Missing components

- Segmentation ground truth for the 2D data → no pixel-level model can be trained on it.
- A segmentation-trained checkpoint for the app. The BraTS pipeline (`src/training_pipeline.py`)
  exists but has not been run (no BraTS data, no GPU on this machine).
- Grade / tumor-type labels; temperature calibration for the existing checkpoint.
- Validation of the detection model on 3D NIfTI input (it was trained on 2D slices only).

## 6. Remediation (ordered by severity) — status in FIXES_AND_CHANGES.md

1. Gate segmentation on `'seg' in trained_heads`; remove every fabricated-mask fallback (D1).
2. Keep the training-identical preprocessing in one shared function (D2).
3. Compute type / location / radiomics / risk only from a real mask; make report text match model outputs (D3).
4. Reorient NIfTI to RAS, use (axial, A→P, R→L) axes and grid spacing for measurements (D4, D5).
5. Encoder-only forward pass when segmentation is not used; per-request temp files; input validation; INDETERMINATE without a model (D6).
6. Next steps needing data/compute: train segmentation on BraTS; calibrate confidence; validate NIfTI.
