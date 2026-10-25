# Fixes and Changes (2026-10-24)

Constraint from the project owner: **the UI design must not change.** No file under
`templates/` or `static/` was edited; only backend responses and text produced by the backend changed.
The model weights (`best_model.pth`) were not modified or retrained.

| # | Defect (AUDIT_REPORT.md) | Change | Files |
| --- | --- | --- | --- |
| 1 | D1 untrained segmentation shown as tumor | Segmentation output is used only if the checkpoint lists `'seg'` in `trained_heads` and the model predicts a tumor. Removed the threshold-0.05 retry, the heuristic mask, and the "brightest pixel disk" rescue. Without a trained head the overlay is empty and the UI shows its existing placeholder. | `app.py` |
| 2 | D2 preprocessing drift | New `dataset.image_to_volume()` is the single 2D preprocessing used by training (`Image2DDataset`), the app (`model_input_2d`) and evaluation. Verified bit-identical to the previous training tensors. | `src/dataset.py`, `app.py` |
| 3 | D3 unsupported derived outputs | Radiomics, risk, rule-based type, location, midline shift and MC-Dropout run only when a real mask exists. Grade shows "Not available (grading model not trained)". Report impression now states only what the model predicted (positive: "does not predict tumor location, size, type or grade"; negative: "not proof that no tumor is present"). Report stage `segmentation: not_available`; `method`/`ai_system` name the real model. | `app.py`, `src/pipeline/report_generator.py` |
| 4 | D4 NIfTI axes / units | New `load_nifti_for_app()`: reorients to RAS (`nib.as_closest_canonical`), arranges (D = axial I→S, H = anterior→posterior, W = patient right→left), z-scores per channel, resizes to the model grid and returns the mm size of one grid voxel; measurements and sub-region volumes use it. Rejects 4D files without 4 sequences in the last axis, empty or non-finite volumes. | `app.py` |
| 5 | D5 grid mismatch | 2D model input, mask and pipeline volume are now the same 64³ tensor. | `app.py` |
| 6 | D6 memory crash | `BrainTumorClassifier.forward_heads()` computes cls/grade/type logits without the decoder (verified equal to `forward()`); used when segmentation is not trained. Per request: 3.3 s → 0.3 s on CPU, no crash. | `src/model.py`, `app.py` |
| 7 | D6 uploads | Extension whitelist (.jpg/.jpeg/.png/.nii/.nii.gz), `PIL.verify()`, minimum 32 px, unreadable NIfTI → HTTP 400 with a message (no server paths). Unique temp file per request (threaded server). | `app.py`, `.gitignore` |
| 8 | D6 no model | 2D without a working model → INDETERMINATE (the heuristic is not used for a decision); `model_available` now also reflects a failed forward pass. | `app.py` |
| 9 | — | Added `model_status.segmentation_available` and `model_status.input_validated` to the JSON (additive; existing fields unchanged). | `app.py` |
| 10 | — | `src/evaluate_app_model.py` (reproducible test-set evaluation with train/test duplicate check) and `tests/test_app.py` (19 tests). | new files |

## Behaviour change visible to users

- 2D image: label and confidence as before (from the model); no coloured overlay, volume, location,
  type, radiomics or risk — these show "—"/"N/A" because the model does not predict them.
- The confidence remains capped at 65% for single 2D images (existing rule; the checkpoint has no
  calibrated temperature, so the raw softmax is not shown).
- NIfTI: detection runs but is flagged as not validated (model trained on 2D slices).

## Requires retraining / new data (not done)

- Real segmentation: train a model with segmentation labels (e.g. `src/training_pipeline.py` on
  BraTS 2021, GPU needed) and save a checkpoint with `trained_heads` containing `'seg'`; the app
  enables the overlay automatically for a `BrainTumorClassifier` checkpoint. Check channel order
  (app expects ET, TC, WT).
- Calibration: retrain with the current `src/train.py` (it stores `trained_heads`, a validation
  split and a fitted temperature).

## Not changed (noted)

- Dashboard cards in `templates/index.html` show hard-coded figures ("96.4% accuracy",
  "2,842 scans", "84 ms"). They are not computed from this model; left as-is because the design
  must not change — replace the text with measured values (Section 4 of AUDIT_REPORT.md) if wanted.
- The printed-report template in `main.js` names "Hybrid CNN-Tx v2.0"; the served model is the SE 3D U-Net.
- Unused functions `extract_tumor_mask_from_heuristic` and `src/inference.load_volume` remain in place.

## Safe training (added later on 2026-10-24)

Incident: at 14:26 a one-epoch training run overwrote `best_model.pth` (the 20-epoch model,
test accuracy 0.995) with an unfinished model (epoch 0, test accuracy 0.831, sensitivity 0.786).
No copy of the old file existed.

Change (`src/train.py`, `app.py`): if the target model file already exists, `train()` now writes to
`<name>.candidate.pth`. After training *and* calibration finish, `promote_if_better()` scores the
candidate and the deployed model on the same validation split and replaces the deployed file only
if the candidate is strictly more accurate (or the deployed file cannot be loaded). The old file is
first copied to `<name>.backup-YYYYMMDD-HHMMSS.pth`. An interrupted run leaves only the candidate.
The app reloads the model only when a new one was promoted. Applies to both the app's Training page
and `python src/train.py`. Tests: `tests/test_safe_training.py`.

Note: the deployed model may have been trained on the validation images, which biases the
comparison toward keeping it (the safe direction). The Training page's Stop button still only
changes the status flag; it does not interrupt `train()`.

# Tumor localization, size, side and risk for 2D uploads (2026-10-25)

Problem reported: a brain MRI with an obvious tumor gave no size (mm), no side and no danger level.

Root causes
1. The served model (`best_model.pth`) only has a trained tumor / no-tumor head. Its segmentation
   head was never trained, so after the 2026-10-24 fixes (which removed fabricated masks) every
   size / side / risk field was correctly left empty. There was no model able to localize a tumor.
2. `best_model.pth` had been replaced by a 1-epoch model (16×64×64 input): on Kaggle Testing it
   misses 7.7% of tumors (sensitivity 0.923).
3. The 2D results were capped at 65% confidence regardless of the evidence.
4. No segmentation masks existed in the project's 2D data.

Changes
| Change | Files |
| --- | --- |
| Downloaded the figshare dataset (expert tumor masks) and converted it | `scripts/prepare_figshare.py`, `DATA.md` |
| New 2D model: ResNet34 U-Net with tumor-type head; patient-level split; Kaggle↔figshare duplicate removal | `src/seg2d.py`, `src/train_seg2d.py` |
| Outline → longest/perpendicular diameter (mm), area, ellipsoid volume, side (radiological convention), view check (sagittal: no side), depth, midline crossing, ring pattern; mm scale from head width or a supplied `pixel_spacing_mm`; rule-based 0–100 risk; overlay with the diameter drawn | `src/pipeline/analysis_2d.py` |
| App: runs the 2D model on JPG/PNG; tumor decision from it; fills measurements, location, type, risk, overlay, impression, scan pixel spacing; flags low-confidence outlines; honest alerts | `app.py` |
| Evaluation script | `src/evaluate_seg2d.py` |
| UI text only: caveat no longer says "confidence capped"; printed report names the real models | `static/js/main.js` |
| Tests | `tests/test_seg2d.py` (14), `tests/test_app.py` updated |

Seg head bug found during training: a ReLU without BatchNorm before the output layer died in the
first epoch, capping every tumor probability at 0.48 (Dice 0). Fixed with BatchNorm; training restarted.
