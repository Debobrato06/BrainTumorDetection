# BrainHybridNet — Brain Tumor Detection, Segmentation and Grading

Research code for a lightweight hybrid CNN-Transformer (BrainHybridNet) that segments
BraTS tumor sub-regions (WT/TC/ET), detects tumor presence and predicts LGG/HGG grade from
4-sequence MRI, plus a Flask demo app.

> **Status.** The code is fixed and unit-tested, but no experiment has been run with it yet.
> Every result in `Main_Manuscript_NCA_BrainHybridNet.md` is a placeholder until it is
> produced by the commands below. The web app is a research prototype, not a clinical tool.

## Layout

```
configs/               YAML for every experiment (proposed model, 4 baselines, ablations)
scripts/               data preparation: BraTS cache, patient-level splits, 2D duplicate check, nnU-Net splits
splits/                committed split files and duplicate reports
src/
  hybrid_model.py      BrainHybridNet (CNN stem + ViT + U-Net decoder, masked pre-training, MC Dropout)
  model.py             SE 3D U-Net used by the web app
  model_factory.py     builds the proposed model or a MONAI baseline from a config
  brats_dataset.py     BraTS loading, preprocessing, random crops
  losses.py            Tversky + multi-task loss
  lightning_module.py  training / validation logic
  training_pipeline.py config-driven training entry point (3D)
  evaluate.py          test-set evaluation -> per_case.csv, summary.json
  stats.py             mean ± SD, bootstrap CIs, Wilcoxon + Holm
  profile_model.py     parameters, FLOPs, latency, memory
  monai_training.py    2D 4-class ViT classifier (scope B)
  evaluate_2d.py       2D test-set evaluation
  train.py             trains the web app's detection model
  pipeline/            rule-based measurement / report modules used by the app (not validated)
tests/                 unit tests
app.py                 Flask demo app
```

## Setup

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Data are not stored in git; see [DATA.md](DATA.md).

## 3D experiments (BraTS 2021)

```bash
# 1. Cache and split (once; commit splits/brats2021_splits.csv)
python scripts/prepare_brats.py --raw_root <BraTS2021_Training_Data> --out data/brats2021_cache
python scripts/make_splits.py --cases_from data/brats2021_cache --out splits/brats2021_splits.csv

# 2. Proposed model, for each fold k in 0..4
python src/training_pipeline.py --config configs/hybrid_pretrain.yaml --fold k --mode pretrain
python src/training_pipeline.py --config configs/hybrid.yaml --fold k --pretrained runs/hybrid_pretrain/foldk/last.ckpt

# 3. Baselines and ablations, same folds
python src/training_pipeline.py --config configs/swinunetr.yaml --fold k      # unet3d, segresnet, unetr, swinunetr
python src/training_pipeline.py --config configs/abl_cnn_only.yaml --fold k   # abl_no_skips, abl_no_se, abl_no_multitask
python src/training_pipeline.py --config configs/hybrid.yaml --fold k         # "no SSL" ablation = no --pretrained
#    nnU-Net v2: python scripts/export_nnunet_splits.py ... then the nnU-Net CLI

# 4. Evaluate on the held-out test set and on each validation fold
python src/evaluate.py --ckpt runs/hybrid/foldk/best.ckpt --stage test --mc_samples 20
python src/evaluate.py --ckpt runs/hybrid/foldk/best.ckpt --stage val --fold k

# 5. Paper numbers
python src/stats.py folds   --pattern "runs/hybrid/fold*/eval_val/per_case.csv"
python src/stats.py summary --runs Ours=runs/hybrid/fold0/eval_test/per_case.csv SwinUNETR=runs/swinunetr/fold0/eval_test/per_case.csv
python src/stats.py compare --a runs/hybrid/fold0/eval_test/per_case.csv --b runs/swinunetr/fold0/eval_test/per_case.csv
python src/profile_model.py --config configs/hybrid.yaml
```

Notes
- BraTS 2021 has no tumor-free volumes and no grade labels. Detection is trained on random
  crops (tumor-free crops are negatives); grading needs a grade CSV passed to
  `make_splits.py --grades` (for example from UCSF-PDGM, also a good external test set).
- Self-supervised pre-training is SimMIM-style masked image modeling (masked input patches,
  learned mask token, loss on masked patches only), using training-fold cases only.

## 2D experiments (Kaggle 4-class dataset)

```bash
python scripts/dedupe_2d.py --root dummy_data --clean_split splits/kaggle4class_clean_split.csv
python src/monai_training.py
python src/evaluate_2d.py --ckpt runs/vit2d/best.ckpt
```

The original Testing folder shares 216 of its 1,311 images pixel-for-pixel with Training;
the clean split keeps each duplicate group on one side only.

## Web app

```bash
python src/train.py --data_dir data/train --epochs 20   # writes best_model.pth
python app.py                                            # http://localhost:5000 (FLASK_DEBUG=1 for debug mode)
```

What the app does (see [MODEL_CARD.md](MODEL_CARD.md)):

- **Input:** one 2D MRI slice (.jpg/.jpeg/.png; axial, coronal or sagittal). NIfTI (.nii/.nii.gz)
  is accepted but only gets tumor / no tumor (not validated on 3D).
- **2D output** (`models/seg2d_best.pth`, see [MODEL_CARD.md](MODEL_CARD.md)): tumor / no tumor,
  tumor outline (overlay), longest and perpendicular diameter in mm, estimated volume, side of the
  brain (radiological convention: image left = patient right), depth, tumor type
  (glioma / meningioma / pituitary) and a rule-based risk level. A JPG has no pixel spacing, so
  the mm scale is estimated from the head width (±15%); send `pixel_spacing_mm` with the upload
  for exact sizes. Outlines with low confidence are flagged.
- **Not predicted:** WHO grade (needs biopsy), edema, enhancement from other sequences.

Train / evaluate the 2D model:

```bash
python scripts/prepare_figshare.py --zips external/figshare_raw --out external/figshare_256.npz
python src/train_seg2d.py --epochs 12 --kaggle_per_epoch 1500 --out models/seg2d_best.pth
python src/evaluate_seg2d.py --out runs/seg2d_eval
```

Model weights: put `best_model.pth` in the project root and `seg2d_best.pth` in `models/`
(neither is in git). Without either model the app answers INDETERMINATE.

Tests: `python -m unittest discover -s tests -v` (app tests: `tests/test_app.py`).

Troubleshooting: on CPU with little free RAM, a 64³ forward pass can crash; the app avoids the
decoder when segmentation is untrained, and `evaluate_app_model.py --batch_size 1` reduces memory.

Audit and fixes: [AUDIT_REPORT.md](AUDIT_REPORT.md), [FIXES_AND_CHANGES.md](FIXES_AND_CHANGES.md),
[TEST_REPORT.md](TEST_REPORT.md).
