"""Evaluate the 2D segmentation/type model (models/seg2d_best.pth) and the app's 2D decision rule.

1. figshare test patients (never seen in training): Dice / IoU of the tumor outline, error of
   the longest diameter, agreement of the side (left/right of the head midline) with the expert
   mask, tumor-type accuracy.
2. Kaggle Testing split: tumor / no-tumor detection for the detection model (best_model.pth),
   the 2D model (the app's rule for 2D images) and their mean; 4-class type accuracy; calibration (ECE).
   Reported for all images and for the images that are not copies of any training image.

Usage: python src/evaluate_seg2d.py --out runs/seg2d_eval
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline'))
from seg2d import CLASSES, load_seg2d, predict_slice  # noqa: E402
from train_seg2d import FIGSHARE_CLASS, find_duplicates, patient_split  # noqa: E402
from analysis_2d import analyze_slice, head_geometry, head_mask, tumor_mask_from_prob, feret_diameters  # noqa: E402

FIGSHARE_PX_MM = 0.98   # nominal mm per pixel of the 256 px cache (README: 0.49 mm at 512 px)


def detection_metrics(y, p, thr=0.5):
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    from sklearn.metrics import roc_auc_score
    return {'n': int(len(y)), 'accuracy': (tp + tn) / len(y), 'sensitivity': tp / max(tp + fn, 1),
            'specificity': tn / max(tn + fp, 1), 'auc': float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp}


def ece(y, p, bins=10):
    """Expected calibration error of the predicted class's confidence."""
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    idx = np.clip(np.digitize(conf, np.linspace(0.5, 1.0, bins + 1)) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        k = idx == b
        if k.any():
            e += k.mean() * abs(conf[k].mean() - correct[k].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/seg2d_best.pth')
    ap.add_argument('--figshare', default='external/figshare_256.npz')
    ap.add_argument('--kaggle_test', default='external/kaggle_test_256.npz')
    ap.add_argument('--kaggle_train', default='external/kaggle_train_256.npz')
    ap.add_argument('--detector_csv', default='runs/app_model_eval_candidate/per_image.csv',
                    help='per-image P(tumor) of best_model.pth from src/evaluate_app_model.py')
    ap.add_argument('--detector_temperature', type=float, default=None,
                    help='temperature the app applies (default: read from best_model.pth)')
    ap.add_argument('--out', default='runs/seg2d_eval')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    model, meta = load_seg2d(args.model)
    results = {'model': args.model, 'epoch': meta.get('epoch')}

    # ── 1. figshare test patients ──
    with np.load(args.figshare) as z:   # NpzFile re-reads an array on every access
        fs = {k: z[k] for k in ('images', 'masks', 'labels', 'pids', 'names')}
    test = patient_split(fs['pids'])['test']
    rows, dices, ious, diam_err, diam_rel, side_ok, side_n, type_ok, scale_ratio = [], [], [], [], [], 0, 0, 0, []
    side_ok_ov, side_n_ov = 0, 0   # slices whose predicted outline overlaps the expert mask
    for i in test:
        img, gt = fs['images'][i], fs['masks'][i] > 0
        prob, cls_p = predict_slice(model, img)
        a = analyze_slice(img, prob, cls_p, CLASSES, pixel_spacing_mm=FIGSHARE_PX_MM)
        a_est = analyze_slice(img, prob, cls_p, CLASSES)          # scale estimated from head width
        pred = a['mask']
        inter, union = (pred & gt).sum(), (pred | gt).sum()
        d = 2 * inter / max(pred.sum() + gt.sum(), 1)
        dices.append(d)
        ious.append(inter / max(union, 1))
        gt_L = feret_diameters(gt)[0] * FIGSHARE_PX_MM
        pr_L = a.get('longest_mm')
        if pr_L is not None:
            diam_err.append(abs(pr_L - gt_L))
            diam_rel.append(abs(pr_L - gt_L) / gt_L)
        if a_est.get('mm_per_px'):
            scale_ratio.append(a_est['mm_per_px'] / FIGSHARE_PX_MM)
        # side agreement on slices where the head looks symmetric (axial / coronal)
        geo = head_geometry(head_mask(img))
        if a['tumor_found'] and a['view'] == 'axial_or_coronal' and geo:
            gy, gx = np.argwhere(gt).mean(axis=0)
            g_off = (gx - geo['mid_x']) / (geo['width_px'] / 2)
            if abs(g_off) >= 0.08:     # ignore midline tumors
                same = int(np.sign(g_off) == np.sign(a['lateral_offset']))
                side_n += 1
                side_ok += same
                if inter > 0:
                    side_n_ov += 1
                    side_ok_ov += same
        true_cls = FIGSHARE_CLASS[int(fs['labels'][i])]
        type_ok += int(int(np.argmax(cls_p)) == true_cls)
        rows.append({'name': str(fs['names'][i]), 'pid': str(fs['pids'][i]), 'true_type': CLASSES[true_cls],
                     'pred_type': a['type_label'], 'dice': round(float(d), 4), 'gt_longest_mm': round(gt_L, 1),
                     'pred_longest_mm': pr_L, 'side': a.get('side'), 'view': a['view']})
    dices = np.array(dices)
    results['figshare_test'] = {
        'slices': int(len(test)), 'patients': int(len(set(fs['pids'][test].tolist()))),
        'dice_mean': float(dices.mean()), 'dice_median': float(np.median(dices)),
        'iou_mean': float(np.mean(ious)), 'slices_dice_ge_0.5': float((dices >= 0.5).mean()),
        'tumor_missed_slices': int(sum(1 for r in rows if r['pred_longest_mm'] is None)),
        'longest_diameter_abs_error_mm_median': float(np.median(diam_err)) if diam_err else None,
        'longest_diameter_abs_error_mm_mean': float(np.mean(diam_err)) if diam_err else None,
        'longest_diameter_rel_error_median': float(np.median(diam_rel)) if diam_rel else None,
        'side_agreement': side_ok / max(side_n, 1), 'side_n': side_n,
        'side_agreement_when_outline_overlaps_tumor': side_ok_ov / max(side_n_ov, 1), 'side_n_overlap': side_n_ov,
        'type_accuracy': type_ok / len(test),
        'estimated_scale_vs_nominal_ratio_percentiles_10_50_90':
            np.percentile(scale_ratio, [10, 50, 90]).round(3).tolist() if scale_ratio else None,
    }
    with open(os.path.join(args.out, 'figshare_test_per_slice.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ── 2. Kaggle Testing ──
    with np.load(args.kaggle_test) as z:
        kt = {k: z[k] for k in z.files}
    det = {}
    T = args.detector_temperature
    if T is None:
        ck = torch.load('best_model.pth', map_location='cpu', weights_only=False)
        T = float(ck.get('temperature', 1.0)) if isinstance(ck, dict) else 1.0
    with open(args.detector_csv) as f:
        for r in csv.DictReader(f):
            p = min(max(float(r['p_tumor']), 1e-7), 1 - 1e-7)
            det[r['path'].replace('\\', '/')] = 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))
    paths = [str(p) for p in kt['paths']]
    y_cls = kt['labels']
    y = (y_cls > 0).astype(int)
    p_det = np.array([det[p] for p in paths])
    cls_probs = []
    for img in kt['images']:
        cls_probs.append(predict_slice(model, img)[1])
    cls_probs = np.array(cls_probs)
    p_seg = 1 - cls_probs[:, 0]
    p_ens = (p_det + p_seg) / 2

    # Copies of training images (figshare train patients or Kaggle Training)
    fs_train = patient_split(fs['pids'])['train']
    dup_fs, _ = find_duplicates(kt['images'], fs['images'][fs_train])
    with np.load(args.kaggle_train) as z:
        dup_kg, _ = find_duplicates(kt['images'], z['images'])
    clean = ~(dup_fs | dup_kg)

    res = {}
    for name, k in [('all', np.ones(len(y), bool)), ('not_copy_of_training_image', clean)]:
        res[name] = {
            'detector_best_model': detection_metrics(y[k], p_det[k]),
            'seg2d_type_head (app rule)': detection_metrics(y[k], p_seg[k]),
            'ensemble_mean': detection_metrics(y[k], p_ens[k]),
            'seg2d_ece (app rule)': ece(y[k], p_seg[k]),
            'ensemble_ece': ece(y[k], p_ens[k]),
            'type_accuracy_4class': float((cls_probs[k].argmax(1) == y_cls[k]).mean()),
            'type_accuracy_on_tumors_3class': float(
                (cls_probs[k][y_cls[k] > 0][:, 1:].argmax(1) + 1 == y_cls[k][y_cls[k] > 0]).mean()),
        }
    res['n_copies_removed'] = int((~clean).sum())
    res['detector_temperature'] = T
    misses = [{'path': paths[i], 'true': CLASSES[y_cls[i]], 'p_det': round(float(p_det[i]), 4),
               'p_seg2d': round(float(p_seg[i]), 4)} for i in np.where((p_seg >= 0.5) != (y == 1))[0]]
    res['seg2d_errors'] = misses
    results['kaggle_test'] = res

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items()}, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
