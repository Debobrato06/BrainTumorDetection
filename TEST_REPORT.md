# Test Report (2026-10-24)

Environment: Windows 11, Python 3.12.10, torch 2.5.1+cpu (no GPU), 15.8 GB RAM,
checkpoint `best_model.pth` (209 MB, epoch 19, img_size 64³, trained heads: cls).

## Unit / integration tests

```
python -m unittest discover -s tests -v
Ran 39 tests ... OK
```

| Requirement | Test | Status |
| --- | --- | --- |
| Unsupported / corrupt / tiny / bad NIfTI / wrong channel count rejected (400) | `TestInputValidation` (5) | PASS |
| No temp files left after requests | `test_no_temp_files_left` | PASS |
| 2D preprocessing: shape (4,64,64,64), float32, range [0,1]; color → grayscale | `TestPreprocessing` | PASS |
| App input identical to training input | `test_app_input_matches_training_input` | PASS |
| NIfTI orientation (RAS and LPS storage give the same axial/A-P/R-L layout) | `TestNiftiLoading` | PASS |
| Grid spacing after resize | `test_grid_spacing` | PASS |
| Checkpoint loads strictly; output shape (1, 2) | `test_checkpoint_loads_strictly` | PASS |
| Encoder-only path equals full forward | `test_forward_heads_matches_forward` | PASS |
| Masks are boolean and nested ET ⊆ TC ⊆ WT | `test_threshold_masks_are_nested_booleans` | PASS |
| No overlay / measurements / radiomics / risk without a trained segmentation head | `test_no_localization_without_trained_segmentation` | PASS |
| Model unavailable → INDETERMINATE | `test_model_unavailable_is_indeterminate` | PASS |
| With a trained segmentation head (stub), mask on model grid, volume in correct mm³ | `test_trained_segmentation_mask_is_aligned_and_measured` | PASS |
| Prediction regression on 5 held-out Kaggle images | `TestPredictionRegression` | PASS |
| Existing research tests (BraTS data, hybrid model, losses, metrics, Lightning) | `test_core.py` (20) | PASS |

## Detection evaluation

```
python src/evaluate_app_model.py --test_dir dummy_data/Testing --train_dir data/train
```

| Subset | n | Acc | Sens | Spec | Precision | AUC | TP/FN/TN/FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 1,311 | 0.9962 | 0.9967 | 0.9951 | 0.9978 | 1.0000 | 903/3/403/2 |
| Not in training (pixel-exact) | 1,095 | 0.9954 | 0.9962 | 0.9933 | 0.9975 | 1.0000 | 792/3/298/2 |

Per class, predicted-tumor rate: glioma 1.000, meningioma 0.990, pituitary 1.000, notumor 0.005.

Baseline comparison (same model, 100 test images, 25 per class):
z-scored input (HEAD app) acc 0.25 / sens 0.00; training preprocessing acc 1.00.
Heuristic used by HEAD (60 per class): flags 68% of tumor-free images as tumor.

## Segmentation evaluation — BLOCKED

No segmentation ground truth exists for the 2D data and no BraTS data is on this machine, and
the served model has no trained segmentation. Dice / IoU / HD95 cannot be measured.

## End-to-end

| Check | Status |
| --- | --- |
| `python app.py` starts; `GET /` and `/profile` return 200 | PASS |
| `POST /predict` over HTTP with glioma, pituitary, no-tumor images → correct labels, no overlay, honest impression | PASS |
| Corrupt PNG and `.txt` upload → 400 with message | PASS |
| UI (Chrome) renders result with existing design: alert, label, "—" for unavailable fields | PASS (visual check, one glioma image) |
| Real 3D NIfTI scan end-to-end | NOT TESTED (no real NIfTI on disk; synthetic NIfTI covered by tests) |
| Lint / static checks | NOT TESTED (no linter configured in the repo) |

## Update after the checkpoint was overwritten (later on 2026-10-24)

`python -m unittest discover -s tests -v` → Ran 45 tests, **1 FAIL**: `test_known_cases`
(`Te-gl_0100.jpg` predicted NEGATIVE). Cause: `best_model.pth` was replaced at 14:26 by a
one-epoch model; code unchanged. Same evaluation on the new checkpoint (not in train, n=1,095):
accuracy 0.8311, sensitivity 0.7862, specificity 0.9500, AUC 0.9306 (`runs/app_model_eval_new`).
The regression test is correct to fail until a good checkpoint is restored or retrained.

| Requirement | Status |
| --- | --- |
| Deployed model replaced only by a finished, strictly better candidate, with backup (`tests/test_safe_training.py`, 6 tests) | PASS |
| Full training run through the new candidate/promote path | NOT TESTED (CPU-only machine, low free RAM) |

## Known issues found during testing

- Full forward pass at 64³ on CPU can crash the process when free RAM is low (seen at 0.9 GB free).
  Avoided in the app by the encoder-only path; would return if a segmentation head is enabled on CPU.
  `evaluate_app_model.py` defaults to batch size 2 for the same reason.

## Update 2026-10-25 — 2D segmentation model

`python -m unittest discover -s tests` → Ran 59 tests, OK (includes `test_known_cases`, which
failed on 2026-10-24 and passes now that 2D decisions come from the 2D model).
Model metrics: MODEL_CARD.md (second card), `runs/seg2d_eval/summary.json`.

Browser test (Chrome, http://localhost:5000, Testing page, upload through the UI):

| Image | Source | Result | Expert mask |
| --- | --- | --- | --- |
| user upload (coronal, ring-enhancing) = Kaggle Te-gl_0030 / figshare 758 (training patient) | — | POSITIVE 99.5%, Glioma 97%, Left side, 31.5 × 23.5 mm, ~9.1 mL, High 61/100, overlay on the lesion | left side, 42 mm (mask includes surrounding tissue) |
| Te-me_0125 (unseen patient) | figshare 139 | POSITIVE 99.4%, Meningioma 98%, Right side, 32.5 × 23.3 mm, Moderate 45 | right side, 32.2 mm |
| Te-pi_0059 (unseen patient) | figshare 939 | POSITIVE 99.5%, Pituitary 98%, Midline (sellar region), 20.5 × 17.8 mm, Moderate 37 | midline, 19.9 mm |
| Te-gl_0024 (unseen patient, posterior fossa) | figshare 1975 | POSITIVE 97.7%, type uncertain (glioma 44%), outline in the WRONG place → shown as "outline uncertain" warning | left posterior fossa, 26 mm |
| Te-no_0100 | Kaggle no tumor | NEGATIVE 95.2%, no overlay, risk not applicable | — |

Expert sizes use the same head-width scale. Not tested: real DICOM-derived pixel spacing from the UI
(API field `pixel_spacing_mm` is unit-tested), NIfTI with the 2D model, print dialog.
