"""All numbers and figures for the 2D segmentation paper (paper_submission/3_SpringerNature_2D).

Evaluates models/seg2d_best.pth on
  - figshare test patients (expert masks): Dice / IoU / precision / recall / HD95 per slice, per tumor
    type, patient-level bootstrap CIs, diameter agreement (Bland-Altman), side agreement, type confusion,
    TTA and threshold ablation, outline-confidence flag;
  - Kaggle Testing: 4-class confusion, per-class F1, tumor detection with CIs, calibration, clean subset,
    comparison with the 3D detection model;
  - CPU inference time.
Outputs: runs/seg2d_paper/metrics.json, per-image CSVs, figures/*.png|pdf.

Usage: python src/paper_metrics_seg2d.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'pipeline'))
from seg2d import CLASSES, load_seg2d  # noqa: E402
from train_seg2d import FIGSHARE_CLASS, find_duplicates, patient_split  # noqa: E402
from analysis_2d import (analyze_slice, feret_diameters, head_geometry, head_mask,  # noqa: E402
                         tumor_mask_from_prob)

OUT = os.path.join(ROOT, 'runs', 'seg2d_paper')
FIG = os.path.join(OUT, 'figures')
PX_MM = 0.98            # nominal mm per pixel of the 256 px figshare cache
B = 2000                # bootstrap resamples
rng = np.random.RandomState(0)


@torch.no_grad()
def forward_tta(model, img):
    """-> (prob without TTA, prob with flip-TTA, class probs with TTA) for one uint8 slice."""
    x = torch.from_numpy(img).float().div(255.0)[None, None]
    x = torch.cat([x, torch.flip(x, dims=[3])])
    seg, cls = model(x)
    p = torch.sigmoid(seg)[:, 0].numpy()
    c = torch.softmax(cls, 1).numpy()
    return p[0], (p[0] + p[1][:, ::-1]) / 2, c.mean(0)


def hd95(a, b, spacing):
    """95th-percentile symmetric Hausdorff distance (mm) between two masks; None if either is empty."""
    from scipy import ndimage
    if not a.any() or not b.any():
        return None
    ea = a & ~ndimage.binary_erosion(a)
    eb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~eb)[ea]
    db = ndimage.distance_transform_edt(~ea)[eb]
    return float(np.percentile(np.concatenate([da, db]), 95) * spacing)


def overlap(pred, gt):
    tp = (pred & gt).sum()
    ps, gs = pred.sum(), gt.sum()
    return {'dice': 2 * tp / max(ps + gs, 1), 'iou': tp / max((pred | gt).sum(), 1),
            'precision': tp / ps if ps else 0.0, 'recall': tp / max(gs, 1)}


def cluster_ci(values, groups):
    """Mean and 95% CI by resampling patients (groups) with replacement."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    means = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        means.append(values[np.concatenate([idx[g] for g in pick])].mean())
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci(fn, n):
    vals = [fn(rng.randint(0, n, n)) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def binary_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score
    pred = p >= thr
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    out = {'n': int(len(y)), 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
           'accuracy': (tp + tn) / len(y), 'sensitivity': tp / max(tp + fn, 1),
           'specificity': tn / max(tn + fp, 1), 'auc': float(roc_auc_score(y, p))}
    out['accuracy_ci'] = boot_ci(lambda i: ((p[i] >= thr) == (y[i] == 1)).mean(), len(y))
    return out


def ece(y, p, bins=10):
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == (y == 1)
    idx = np.clip(np.digitize(conf, np.linspace(0.5, 1, bins + 1)) - 1, 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        k = idx == b
        if k.any():
            e += k.mean() * abs(conf[k].mean() - correct[k].mean())
            rows.append((float(conf[k].mean()), float(correct[k].mean()), int(k.sum())))
    return float(e), rows


def main():
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(min(6, os.cpu_count() or 4))
    model, meta = load_seg2d(os.path.join(ROOT, 'models', 'seg2d_best.pth'))
    M = {'model': {'epoch': meta.get('epoch'), 'val_dice': meta.get('val_dice'),
                   'parameters': int(sum(p.numel() for p in model.parameters()))}}

    with np.load(os.path.join(ROOT, 'external', 'figshare_256.npz')) as z:
        fs = {k: z[k] for k in ('images', 'masks', 'labels', 'pids', 'names')}
    split = patient_split(fs['pids'])

    # ── figshare val + test: predictions ──
    rows, flag_rows = [], []
    for part in ('val', 'test'):
        for i in split[part]:
            img, gt = fs['images'][i], fs['masks'][i] > 0
            p_raw, p_tta, c = forward_tta(model, img)
            hm = head_mask(img)
            a = analyze_slice(img, p_tta, c, CLASSES, pixel_spacing_mm=PX_MM)
            pred = a['mask']
            flag_rows.append({'part': part, 'found': a['tumor_found'],
                              'dice': overlap(pred, gt)['dice'], 'mean_prob': a.get('mean_prob')})
            if part != 'test':
                continue
            r = {'name': str(fs['names'][i]), 'pid': str(fs['pids'][i]),
                 'type': CLASSES[FIGSHARE_CLASS[int(fs['labels'][i])]], 'pred_type': a['type_label'],
                 'type_probs': c.tolist(), 'found': a['tumor_found'], 'view': a['view']}
            r.update(overlap(pred, gt))
            r['hd95_mm'] = hd95(pred, gt, PX_MM)
            for tta, prob in (('tta', p_tta), ('raw', p_raw)):
                for thr in (0.3, 0.5, 0.7):
                    r[f'dice_{tta}_{thr}'] = overlap(tumor_mask_from_prob(prob, hm, thr), gt)['dice']
            gL, gP, _ = feret_diameters(gt)
            r['gt_longest_mm'], r['gt_perp_mm'] = gL * PX_MM, gP * PX_MM
            r['gt_area_mm2'] = gt.sum() * PX_MM ** 2
            r['pred_longest_mm'] = a.get('longest_mm')
            r['pred_perp_mm'] = a.get('perpendicular_mm')
            r['pred_area_mm2'] = a.get('area_mm2')
            geo = head_geometry(hm)
            r['gt_offset'] = float((np.argwhere(gt)[:, 1].mean() - geo['mid_x']) / (geo['width_px'] / 2)) if geo else None
            r['pred_offset'] = a.get('lateral_offset')
            a_est = analyze_slice(img, p_tta, c, CLASSES)
            r['scale_ratio'] = a_est['mm_per_px'] / PX_MM if a_est.get('mm_per_px') else None
            r['est_longest_mm'] = a_est.get('longest_mm')
            r['gt_longest_mm_est_scale'] = gL * a_est['mm_per_px'] if a_est.get('mm_per_px') else None
            rows.append(r)

    with open(os.path.join(OUT, 'figshare_test_per_slice.csv'), 'w', newline='') as f:
        keys = [k for k in rows[0] if k != 'type_probs']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    pid = np.array([r['pid'] for r in rows])
    seg = {'slices': len(rows), 'patients': int(len(set(pid))), 'missed_slices': int(sum(not r['found'] for r in rows))}
    for k in ('dice', 'iou', 'precision', 'recall'):
        v = np.array([r[k] for r in rows])
        m, lo, hi = cluster_ci(v, pid)
        seg[k] = {'mean': m, 'ci95': [lo, hi], 'median': float(np.median(v)), 'sd': float(v.std())}
    h = [(r['hd95_mm'], r['pid']) for r in rows if r['hd95_mm'] is not None]
    hv = np.array([x for x, _ in h])
    m, lo, hi = cluster_ci(hv, [p for _, p in h])
    seg['hd95_mm'] = {'mean': m, 'ci95': [lo, hi], 'median': float(np.median(hv)), 'n': len(hv)}
    seg['dice_ge_0.5'] = float(np.mean([r['dice'] >= 0.5 for r in rows]))
    seg['dice_lt_0.1'] = int(sum(r['dice'] < 0.1 for r in rows))
    per_type = {}
    for t in CLASSES[1:]:
        sel = [r for r in rows if r['type'] == t]
        v = np.array([r['dice'] for r in sel])
        mm, lo, hi = cluster_ci(v, [r['pid'] for r in sel])
        per_type[t] = {'slices': len(sel), 'patients': len({r['pid'] for r in sel}), 'dice_mean': mm, 'ci95': [lo, hi],
                       'dice_median': float(np.median(v)), 'hd95_median': float(np.median([r['hd95_mm'] for r in sel if r['hd95_mm'] is not None]))}
    seg['per_type'] = per_type
    ablation = {}
    for tta in ('raw', 'tta'):
        for thr in (0.3, 0.5, 0.7):
            v = np.array([r[f'dice_{tta}_{thr}'] for r in rows])
            ablation[f'{tta}_{thr}'] = {'dice_mean': float(v.mean()), 'dice_median': float(np.median(v))}
    seg['ablation_tta_threshold'] = ablation
    M['figshare_test_segmentation'] = seg

    # diameters (known spacing)
    dd = [r for r in rows if r['pred_longest_mm'] is not None]
    pL, gLv = np.array([r['pred_longest_mm'] for r in dd]), np.array([r['gt_longest_mm'] for r in dd])
    diff = pL - gLv
    ov = np.array([r['dice'] > 0.1 for r in dd])
    pA, gA = np.array([r['pred_area_mm2'] for r in dd]), np.array([r['gt_area_mm2'] for r in dd])
    M['figshare_test_measurement'] = {
        'n': len(dd), 'longest_mae_mm': float(np.abs(diff).mean()), 'longest_median_ae_mm': float(np.median(np.abs(diff))),
        'longest_bias_mm': float(diff.mean()), 'longest_loa_mm': [float(diff.mean() - 1.96 * diff.std()), float(diff.mean() + 1.96 * diff.std())],
        'longest_pearson_r': float(np.corrcoef(pL, gLv)[0, 1]),
        'longest_median_rel_error': float(np.median(np.abs(diff) / gLv)),
        'n_overlapping': int(ov.sum()),
        'longest_mae_mm_overlapping': float(np.abs(diff[ov]).mean()),
        'longest_median_ae_mm_overlapping': float(np.median(np.abs(diff[ov]))),
        'longest_bias_mm_overlapping': float(diff[ov].mean()),
        'longest_loa_mm_overlapping': [float(diff[ov].mean() - 1.96 * diff[ov].std()), float(diff[ov].mean() + 1.96 * diff[ov].std())],
        'longest_pearson_r_overlapping': float(np.corrcoef(pL[ov], gLv[ov])[0, 1]),
        'area_pearson_r': float(np.corrcoef(pA, gA)[0, 1]),
        'area_median_rel_error': float(np.median(np.abs(pA - gA) / gA)),
    }
    sr = np.array([r['scale_ratio'] for r in rows if r['scale_ratio']])
    M['scale_estimation'] = {'ratio_estimated_to_nominal_p10_p50_p90': np.percentile(sr, [10, 50, 90]).tolist(),
                             'median_abs_dev_from_median': float(np.median(np.abs(sr - np.median(sr)) / np.median(sr)))}
    side = [r for r in rows if r['found'] and r['view'] == 'axial_or_coronal' and r['gt_offset'] is not None and abs(r['gt_offset']) >= 0.08]
    side_ok = [np.sign(r['gt_offset']) == np.sign(r['pred_offset']) for r in side]
    side_ov = [ok for ok, r in zip(side_ok, side) if r['dice'] > 0]
    M['side'] = {'n': len(side), 'agreement': float(np.mean(side_ok)), 'n_overlapping': len(side_ov),
                 'agreement_overlapping': float(np.mean(side_ov))}
    cm_fs = np.zeros((4, 4), int)
    for r in rows:
        cm_fs[CLASSES.index(r['type']), CLASSES.index(r['pred_type'])] += 1
    M['figshare_test_type'] = {'confusion_rows_true_cols_pred': cm_fs.tolist(), 'classes': CLASSES,
                               'accuracy': float(np.trace(cm_fs) / cm_fs.sum())}

    # outline-confidence flag on val + test
    fr = [r for r in flag_rows if r['found']]
    bad = np.array([r['dice'] < 0.1 for r in fr])
    low = np.array([r['mean_prob'] < 0.8 for r in fr])
    M['outline_flag'] = {'threshold': 0.8, 'n': len(fr), 'failed_outlines': int(bad.sum()),
                         'flagged_failed': int((low & bad).sum()), 'flagged_correct': int((low & ~bad).sum())}

    # ── Kaggle Testing ──
    with np.load(os.path.join(ROOT, 'external', 'kaggle_test_256.npz')) as z:
        kt = {k: z[k] for k in z.files}
    probs, t0 = [], time.time()
    for img in kt['images']:
        probs.append(forward_tta(model, img)[2])
    probs = np.array(probs)
    y4 = kt['labels']
    y = (y4 > 0).astype(int)
    p_tumor = 1 - probs[:, 0]
    det = {}
    T = float(torch.load(os.path.join(ROOT, 'best_model.pth'), map_location='cpu', weights_only=False).get('temperature', 1.0))
    with open(os.path.join(ROOT, 'runs', 'app_model_eval_candidate', 'per_image.csv')) as f:
        for r in csv.DictReader(f):
            q = min(max(float(r['p_tumor']), 1e-7), 1 - 1e-7)
            det[r['path'].replace('\\', '/')] = 1 / (1 + np.exp(-np.log(q / (1 - q)) / T))
    paths = [str(p) for p in kt['paths']]
    p_det = np.array([det[p] for p in paths])
    fs_train = split['train']
    dup_fs, _ = find_duplicates(kt['images'], fs['images'][fs_train])
    with np.load(os.path.join(ROOT, 'external', 'kaggle_train_256.npz')) as z:
        dup_kg, _ = find_duplicates(kt['images'], z['images'])
    clean = ~(dup_fs | dup_kg)

    cm = np.zeros((4, 4), int)
    for t, pr in zip(y4, probs.argmax(1)):
        cm[t, pr] += 1
    f1 = {}
    for k, c in enumerate(CLASSES):
        prec = cm[k, k] / max(cm[:, k].sum(), 1)
        rec = cm[k, k] / max(cm[k].sum(), 1)
        f1[c] = {'precision': float(prec), 'recall': float(rec), 'f1': float(2 * prec * rec / max(prec + rec, 1e-9)), 'n': int(cm[k].sum())}
    e, rel = ece(y, p_tumor)
    acc4 = (probs.argmax(1) == y4)
    K = {'n': int(len(y)), 'n_clean': int(clean.sum()), 'n_clean_per_class': {c: int((clean & (y4 == k)).sum()) for k, c in enumerate(CLASSES)},
         'confusion_rows_true_cols_pred': cm.tolist(), 'per_class': f1,
         'accuracy_4class': float(acc4.mean()), 'accuracy_4class_ci': boot_ci(lambda i: acc4[i].mean(), len(y)),
         'accuracy_4class_clean': float(acc4[clean].mean()),
         'macro_f1': float(np.mean([v['f1'] for v in f1.values()])),
         'detection_seg2d': binary_metrics(y, p_tumor), 'detection_seg2d_clean': binary_metrics(y[clean], p_tumor[clean]),
         'detection_3d_detector': binary_metrics(y, p_det), 'detection_3d_detector_clean': binary_metrics(y[clean], p_det[clean]),
         'detection_mean_of_both': binary_metrics(y, (p_tumor + p_det) / 2),
         'ece_seg2d': e, 'ece_3d_detector': ece(y, p_det)[0], 'reliability_seg2d': rel,
         'detector_temperature': T}
    M['kaggle_test'] = K
    with open(os.path.join(OUT, 'kaggle_test_per_image.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path', 'true', 'pred', 'p_tumor_seg2d', 'p_tumor_3d_detector', 'copy_of_training_image'] + [f'p_{c}' for c in CLASSES])
        for i in range(len(y)):
            w.writerow([paths[i], CLASSES[y4[i]], CLASSES[probs[i].argmax()], round(float(p_tumor[i]), 5),
                        round(float(p_det[i]), 5), int(not clean[i])] + [round(float(v), 5) for v in probs[i]])

    # ── CPU timing: full analysis of one image (model + measurements) ──
    from PIL import Image
    from seg2d import prepare_slice, predict_slice
    from analysis_2d import risk_estimate_2d
    imgs = [os.path.join(ROOT, p) for p in paths[:40]]
    times = []
    for p in imgs:
        t = time.time()
        with Image.open(p) as im:
            sl, _ = prepare_slice(im)
        pr, c = predict_slice(model, sl)
        a = analyze_slice(sl, pr, c, CLASSES)
        if a['tumor_found']:
            risk_estimate_2d(a)
        times.append(time.time() - t)
    M['timing_cpu'] = {'threads': torch.get_num_threads(), 'median_s': float(np.median(times)), 'p90_s': float(np.percentile(times, 90))}

    with open(os.path.join(OUT, 'metrics.json'), 'w') as f:
        json.dump(M, f, indent=1)
    make_figures(rows, M, fs, split, model)
    print(json.dumps({k: v for k, v in M.items() if k != 'kaggle_test'}, indent=1))


# ─── Figures ──────────────────────────────────────────────────────────────────

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=300, bbox_inches='tight')


def make_figures(rows, M, fs, split, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})

    # Qualitative: best, median and worst slices per tumor type (expert red, model green)
    names = list(fs['names'])
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.0))
    for col, t in enumerate(CLASSES[1:]):
        sel = sorted([r for r in rows if r['type'] == t], key=lambda r: r['dice'])
        picks = [sel[-1], sel[len(sel) // 2], sel[0]]
        for row, r in enumerate(picks):
            i = names.index(r['name'])
            img = fs['images'][i]
            _, p, c = forward_tta(model, img)
            a = analyze_slice(img, p, c, CLASSES, pixel_spacing_mm=PX_MM)
            rgb = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_GRAY2RGB)
            for m, colr in ((fs['masks'][i], (255, 40, 40)), (a['mask'].astype(np.uint8), (40, 255, 40))):
                cs, _ = cv2.findContours(cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cs, -1, colr, 2)
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{t} — {['best', 'median', 'worst'][row]}\nDice {r['dice']:.2f}, pred: {a['type_label']}", fontsize=7)
    fig.tight_layout()
    save(fig, 'fig3_qualitative')
    plt.close(fig)

    # Diameter agreement: scatter + Bland-Altman
    dd = [r for r in rows if r['pred_longest_mm'] is not None]
    g = np.array([r['gt_longest_mm'] for r in dd])
    p = np.array([r['pred_longest_mm'] for r in dd])
    ov = np.array([r['dice'] > 0.1 for r in dd])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    a1.scatter(g[ov], p[ov], s=6, c='#1f77b4', label='outline overlaps tumor')
    a1.scatter(g[~ov], p[~ov], s=10, c='#d62728', marker='x', label='outline misses tumor')
    lim = [0, max(g.max(), p.max()) * 1.05]
    a1.plot(lim, lim, 'k--', lw=0.7)
    a1.set_xlim(lim), a1.set_ylim(lim)
    a1.set_xlabel('Expert longest diameter (mm)'), a1.set_ylabel('Predicted longest diameter (mm)')
    a1.legend(fontsize=6, loc='upper left')
    mean, diff = (g[ov] + p[ov]) / 2, p[ov] - g[ov]
    a2.scatter(mean, diff, s=6, c='#1f77b4')
    bias, sd = diff.mean(), diff.std()
    for v, ls in ((bias, '-'), (bias - 1.96 * sd, '--'), (bias + 1.96 * sd, '--')):
        a2.axhline(v, color='k', ls=ls, lw=0.7)
    a2.set_xlabel('Mean of expert and predicted (mm)'), a2.set_ylabel('Predicted − expert (mm)')
    a2.set_title(f'Bias {bias:.1f} mm, limits of agreement {bias - 1.96 * sd:.1f} to {bias + 1.96 * sd:.1f} mm', fontsize=7)
    fig.tight_layout()
    save(fig, 'fig4_diameter_agreement')
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    short = ['No tumor', 'Glioma', 'Mening.', 'Pituitary']
    for ax, cmx, title in ((axes[0], np.array(M['kaggle_test']['confusion_rows_true_cols_pred']), 'Kaggle Testing (n = 1,311)'),
                           (axes[1], np.array(M['figshare_test_type']['confusion_rows_true_cols_pred'])[1:, :], 'figshare test patients (n = 295)')):
        ax.imshow(cmx / np.maximum(cmx.sum(1, keepdims=True), 1), cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cmx):
            ax.text(j, i, str(v), ha='center', va='center', color='white' if v > cmx.sum(1)[i] * 0.6 else 'black', fontsize=7)
        ax.set_xticks(range(4)), ax.set_xticklabels(short, rotation=30)
        rows_lbl = short if cmx.shape[0] == 4 else short[1:]
        ax.set_yticks(range(len(rows_lbl))), ax.set_yticklabels(rows_lbl)
        ax.set_xlabel('Predicted'), ax.set_ylabel('True'), ax.set_title(title, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig5_confusion')
    plt.close(fig)

    # Training curves + reliability diagram
    hist = json.load(open(os.path.join(ROOT, 'runs', 'seg2d', 'history.json')))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ep = [h['epoch'] for h in hist]
    a1.plot(ep, [h['val_dice_figshare'] for h in hist], 'o-', ms=3, label='Val Dice (figshare)')
    a1.plot(ep, [h['val_cls_acc_kaggle'] for h in hist], 's-', ms=3, label='Val accuracy (Kaggle)')
    a1.plot(ep, [h['val_cls_acc_figshare'] for h in hist], '^-', ms=3, label='Val type accuracy (figshare)')
    a1.set_xlabel('Epoch'), a1.set_ylim(0.4, 1.0), a1.legend(fontsize=6), a1.set_title('Validation during training', fontsize=8)
    rel = M['kaggle_test']['reliability_seg2d']
    a2.plot([0.5, 1], [0.5, 1], 'k--', lw=0.7)
    a2.plot([r[0] for r in rel], [r[1] for r in rel], 'o-', ms=3)
    for r in rel:
        a2.annotate(str(r[2]), (r[0], r[1]), fontsize=5, textcoords='offset points', xytext=(2, -8))
    a2.set_xlabel('Confidence'), a2.set_ylabel('Accuracy')
    a2.set_title(f"Reliability, tumor vs no tumor (ECE {M['kaggle_test']['ece_seg2d']:.3f})", fontsize=8)
    fig.tight_layout()
    save(fig, 'fig6_training_calibration')
    plt.close(fig)


if __name__ == '__main__':
    main()
