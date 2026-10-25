"""
Evaluate a trained checkpoint on the held-out test split (or a validation fold) and
write every number the paper reports to CSV/JSON.

    python src/evaluate.py --ckpt runs/hybrid/fold0/best.ckpt --stage test --mc_samples 20

Outputs in <ckpt dir>/eval_<stage>/:
    per_case.csv   Dice / HD95 / sensitivity / specificity per region, grade prob, uncertainty
    summary.json   means, SDs, grading AUC / accuracy, voxel ECE, uncertainty-error correlation
    reliability.csv  reliability-diagram bins (WT voxels)
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from brats_dataset import BraTSEvalDataset, cases_for, read_splits
from hybrid_model import enable_mc_dropout
from lightning_module import TumorResearchModule, sliding_window_seg
from metrics import REGIONS, expected_calibration_error, region_metrics, reliability_curve


def center_crop_around(image, center, size):
    starts = [int(min(max(0, c - s // 2), max(0, dim - s))) for c, s, dim in zip(center, size, image.shape[2:])]
    sl = (slice(None), slice(None)) + tuple(slice(st, st + s) for st, s in zip(starts, size))
    crop = image[sl]
    pad = []
    for s, have in zip(reversed(size), reversed(crop.shape[2:])):
        pad += [0, s - have]
    return torch.nn.functional.pad(crop, pad)


@torch.no_grad()
def main(args):
    module = TumorResearchModule.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    cfg = module.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.model.to(device).eval()
    roi = module.roi
    ev = cfg.get("eval", {})

    rows = read_splits(args.splits or cfg["data"]["splits_csv"])
    cases = cases_for(rows, args.stage, args.fold)
    ds = BraTSEvalDataset(cfg["data"]["cache_root"], cases)
    out_dir = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.stage}")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    per_case, voxel_probs, voxel_labels = [], [], []
    grade_true, grade_prob = [], []
    has_heads = model(torch.zeros(1, 4, *roi, device=device))[0] is not None

    for i in range(len(ds)):
        item = ds[i]
        image = item["image"].unsqueeze(0).to(device)
        ref = item["seg"].numpy() > 0.5

        if args.mc_samples > 1 and has_heads:
            enable_mc_dropout(model)
            samples = [torch.sigmoid(sliding_window_seg(model, image, roi, ev.get("sw_overlap", 0.5)))
                       for _ in range(args.mc_samples)]
            model.eval()
            stack = torch.stack(samples)
            prob, var = stack.mean(0)[0].cpu().numpy(), stack.var(0)[0].cpu().numpy()
        else:
            prob = torch.sigmoid(sliding_window_seg(model, image, roi, ev.get("sw_overlap", 0.5)))[0].cpu().numpy()
            var = None
        pred = prob > 0.5

        row = {"case_id": item["case_id"], **region_metrics(pred, ref)}

        if var is not None:
            err = pred[0] != ref[0]
            row["mean_uncertainty_wt"] = float(var[0][pred[0] | ref[0]].mean()) if (pred[0] | ref[0]).any() else 0.0
            sub = rng.choice(err.size, size=min(200_000, err.size), replace=False)
            row["voxel_unc_error_spearman"] = float(spearmanr(var[0].ravel()[sub], err.ravel()[sub]).correlation)

        # Detection / grading from one crop centered on the predicted (else image) tumor center
        if has_heads:
            fg = np.argwhere(pred[0])
            center = fg.mean(0) if len(fg) else np.array(image.shape[2:]) / 2
            crop = center_crop_around(image, center.astype(int), roi)
            cls_logits, _, grade_logits = model(crop)[:3]
            row["det_prob"] = float(torch.softmax(cls_logits, 1)[0, 1])
            g = int(item["grade"])
            row["grade_true"] = g
            row["grade_prob_hgg"] = float(torch.softmax(grade_logits, 1)[0, 1])
            if g >= 0:
                grade_true.append(g)
                grade_prob.append(row["grade_prob_hgg"])

        # voxel calibration sample (WT channel, inside the brain)
        brain = np.any(item["image"].numpy() != 0, axis=0).ravel()
        idx = np.flatnonzero(brain)
        idx = rng.choice(idx, size=min(20_000, idx.size), replace=False)
        voxel_probs.append(prob[0].ravel()[idx])
        voxel_labels.append(ref[0].ravel()[idx])

        per_case.append(row)
        print(f"[{i + 1}/{len(ds)}] {item['case_id']} " +
              " ".join(f"{r}={row[f'dice_{r}']:.3f}" for r in REGIONS))

    keys = list(per_case[0].keys())
    for r in per_case:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_case)

    summary = {"checkpoint": args.ckpt, "stage": args.stage, "n_cases": len(per_case)}
    for k in keys:
        vals = np.array([r[k] for r in per_case if isinstance(r.get(k), (int, float))], dtype=float)
        if k != "case_id" and vals.size:
            summary[f"{k}_mean"] = float(np.nanmean(vals))
            summary[f"{k}_sd"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0

    vp, vl = np.concatenate(voxel_probs), np.concatenate(voxel_labels)
    summary["voxel_ece_wt"] = expected_calibration_error(vp, vl)
    c, fr, n = reliability_curve(vp, vl)
    with open(os.path.join(out_dir, "reliability.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mean_predicted", "observed_frequency", "count"])
        w.writerows(zip(c, fr, n))

    if per_case and "mean_uncertainty_wt" in per_case[0]:
        u = [r["mean_uncertainty_wt"] for r in per_case]
        e = [1 - r["dice_wt"] for r in per_case]
        rho, p = spearmanr(u, e)
        summary["case_unc_vs_1minus_dice_wt_spearman"] = float(rho)
        summary["case_unc_vs_1minus_dice_wt_p"] = float(p)

    if len(set(grade_true)) == 2:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
        gp = np.array(grade_prob)
        summary["grade_auc"] = float(roc_auc_score(grade_true, gp))
        summary["grade_accuracy"] = float(accuracy_score(grade_true, gp > 0.5))
        summary["grade_balanced_accuracy"] = float(balanced_accuracy_score(grade_true, gp > 0.5))
        summary["grade_n"] = len(grade_true)
    else:
        summary["grade_note"] = "No cases with both LGG and HGG labels in this split; grading not evaluated."

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if "dice" in k and "mean" in k}, indent=2))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stage", choices=["test", "val"], default="test")
    p.add_argument("--fold", type=int, default=0, help="fold index when --stage val")
    p.add_argument("--splits", default=None, help="override splits CSV (e.g. external test set)")
    p.add_argument("--mc_samples", type=int, default=1, help=">1 enables MC Dropout uncertainty (hybrid only)")
    main(p.parse_args())
