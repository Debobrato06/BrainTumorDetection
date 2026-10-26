"""Figures for the Journal of Imaging Informatics in Medicine submission (5_JIIM_Submission/figures).

Springer artwork rules applied: Arial lettering of at least 8 pt, 174 mm (double-column) width,
600 dpi RGB TIFF for combination art, files named Fig1.tif ... Fig6.tif, panels labelled (a), (b),
no figure titles inside the artwork. Low-resolution PNG copies for the Word preview go to
_build/jiim_png. Every number is read from runs/seg2d_paper.
Usage: python paper_submission/make_figures_jiim.py [1-6 ...]
"""
import csv
import io
import json
import os
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, '5_JIIM_Submission', 'figures')
PREVIEW = os.path.join(HERE, '_build', 'jiim_png')
PAPER = os.path.join(ROOT, 'runs', 'seg2d_paper')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'pipeline'))

W_IN = 174 / 25.4            # double-column width
FS = 8                       # minimum lettering size (pt)
RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS,
                     'ytick.labelsize': FS, 'legend.fontsize': FS, 'axes.linewidth': 0.6,
      'pdf.fonttype': 42, 'ps.fonttype': 42}
plt.rcParams.update(RC)


def save(fig, n):
    """Fig<n>.tif at 600 dpi (RGB, LZW) plus a 300 dpi PNG for the Word files, both exactly W_IN wide."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, facecolor='white')
    plt.close(fig)
    im = Image.open(buf).convert('RGB')
    im.save(os.path.join(OUT, f'Fig{n}.tif'), compression='tiff_lzw', dpi=(600, 600))
    im.resize((round(im.width / 2), round(im.height / 2)), Image.LANCZOS).save(os.path.join(PREVIEW, f'Fig{n}.png'), dpi=(300, 300))
    print(f'Fig{n}.tif {im.width} x {im.height} px = {im.width / 600 * 25.4:.0f} x {im.height / 600 * 25.4:.0f} mm')


def label(ax, s, x=-0.02, y=1.02, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', **{'ha': 'right', 'va': 'bottom', **kw})


def panels(fig, xs=(0.0, 0.5)):
    """(a), (b) at the top-left corner of the left and right halves of a two-panel figure."""
    for x, s in zip(xs, '(a) (b)'.split()):
        fig.text(x + 0.005, 0.995, s, fontsize=9, fontweight='bold', ha='left', va='top')


def metrics():
    return json.load(open(os.path.join(PAPER, 'metrics.json')))


def test_rows():
    rows = list(csv.DictReader(open(os.path.join(PAPER, 'figshare_test_per_slice.csv'))))
    for r in rows:
        r['dice'] = float(r['dice'])
    return rows


# ─── Fig. 1: system overview ────────────────────────────────────────────────
def fig1():
    from make_figures_2d import arrow, box
    plt.rcParams.update(RC)   # make_figures_2d sets a serif font on import
    fig = plt.figure(figsize=(W_IN, 3.28))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14.4), ax.set_ylim(0.05, 6.78), ax.axis('off')
    box(ax, 0.05, 4.35, 2.15, 2.0, 'Input\n2D MRI slice\n(JPG / PNG)', '#e8eef7', FS)
    box(ax, 2.55, 4.35, 2.45, 2.0, 'Preprocessing\ngrayscale, pad to\nsquare, 256 × 256,\nscale to [0, 1]', '#e8eef7', FS)
    box(ax, 5.35, 4.35, 2.95, 2.0, 'ResNet34 encoder\n(ImageNet weights)\n+ U-Net decoder\n24.4 M parameters', '#cfe3f5', FS)
    box(ax, 8.7, 5.5, 2.55, 1.2, 'Classification head\n4 classes', '#efd9f0', FS)
    box(ax, 8.7, 4.0, 2.55, 1.2, 'Segmentation head\ntumor probability', '#d8efd3', FS)
    box(ax, 11.6, 4.35, 2.75, 2.35, 'Decision\nP(tumor) =\n1 − P(no tumor);\ntype = arg max;\nflip averaging', '#efd9f0', FS)
    for a, b in [((2.2, 5.35), (2.55, 5.35)), ((5.0, 5.35), (5.35, 5.35)), ((8.3, 5.7), (8.7, 6.0)),
                 ((8.3, 5.0), (8.7, 4.6)), ((11.25, 6.1), (11.6, 5.8))]:
        arrow(ax, a, b)
    box(ax, 0.05, 0.1, 3.3, 3.2, 'Head mask\n(Otsu, largest\ncomponent);\nview check by\nmirror correlation', '#f6efd9', FS)
    box(ax, 3.7, 0.1, 3.3, 3.2, 'Tumor mask\nthreshold 0.5,\ninside head, small\ncomponents removed;\nflag if mean\nprobability < 0.8', '#d8efd3', FS)
    box(ax, 7.35, 0.1, 3.3, 3.2, 'Measurements\nlongest and\nperpendicular\ndiameter, area;\nscale from DICOM\nor head width', '#f6efd9', FS)
    box(ax, 11.0, 0.1, 3.35, 3.2, 'Report\nside (radiological\nconvention), depth,\nmidline crossing,\noverlay image', '#f3f3f3', FS)
    arrow(ax, (9.95, 4.0), (5.35, 3.3))
    arrow(ax, (3.35, 1.7), (3.7, 1.7)), arrow(ax, (7.0, 1.7), (7.35, 1.7)), arrow(ax, (10.65, 1.7), (11.0, 1.7))
    arrow(ax, (3.7, 4.35), (1.7, 3.3))
    save(fig, 1)


# ─── Fig. 2: best, median and worst test slice per tumor type ───────────────
def read_rows(member, wanted):
    """Streams one array of the compressed figshare cache and returns {index: row} for the wanted
    indices, so the 200 MB arrays are never held in memory."""
    with zipfile.ZipFile(os.path.join(ROOT, 'external', 'figshare_256.npz')) as z, z.open(member) as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, _, dtype = read_header(f)
        row = int(np.prod(shape[1:])) * dtype.itemsize
        out = {}
        for i in range(max(wanted) + 1):
            b = f.read(row)
            if i in wanted:
                out[i] = np.frombuffer(b, dtype).reshape(shape[1:]).copy()
        return out


def fig2():
    import cv2
    import torch
    from seg2d import CLASSES, load_seg2d
    from analysis_2d import analyze_slice
    from paper_metrics_seg2d import PX_MM, forward_tta
    torch.set_num_threads(4)
    rows = test_rows()
    with zipfile.ZipFile(os.path.join(ROOT, 'external', 'figshare_256.npz')) as z:
        names = list(np.load(z.open('names.npy')))
    picks = []
    for t in CLASSES[1:]:
        sel = sorted([r for r in rows if r['type'] == t], key=lambda r: r['dice'])
        picks.append([sel[-1], sel[len(sel) // 2], sel[0]])
    idx = {r['name']: names.index(r['name']) for col in picks for r in col}
    imgs, masks = read_rows('images.npy', set(idx.values())), read_rows('masks.npy', set(idx.values()))
    model, _ = load_seg2d(os.path.join(ROOT, 'models', 'seg2d_best.pth'))
    fig, axes = plt.subplots(3, 3, figsize=(W_IN, W_IN * 1.04))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.925, bottom=0.005, wspace=0.04, hspace=0.2)
    for col, (t, sel) in enumerate(zip(CLASSES[1:], picks)):
        for row, r in enumerate(sel):
            i = idx[r['name']]
            img = imgs[i]
            _, p, c = forward_tta(model, img)
            a = analyze_slice(img, p, c, CLASSES, pixel_spacing_mm=PX_MM)
            rgb = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_GRAY2RGB)
            # expert outline red and solid, model outline green and dashed-looking (thinner), so the two
            # remain distinguishable without colour
            for m, colr, th in ((masks[i], (255, 40, 40), 3), (a['mask'].astype(np.uint8), (40, 255, 40), 2)):
                cs, _ = cv2.findContours(cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cs, -1, colr, th)
            ax = axes[row, col]
            ax.imshow(rgb), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"Dice {r['dice']:.2f}; predicted {a['type_label'].lower()}", pad=3)
            if row == 0:
                ax.text(0.5, 1.11, t, transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
            if col == 0:
                ax.set_ylabel(['Best', 'Median', 'Worst'][row], fontweight='bold')
            label(ax, '(' + 'abcdefghi'[row * 3 + col] + ')', x=0.02, y=0.97, ha='left', va='top', color='white')
    save(fig, 2)


# ─── Fig. 3: diameter agreement ─────────────────────────────────────────────
def fig3():
    rows = [r for r in test_rows() if r['pred_longest_mm'] not in ('', 'None')]
    g = np.array([float(r['gt_longest_mm']) for r in rows])
    p = np.array([float(r['pred_longest_mm']) for r in rows])
    ov = np.array([r['dice'] > 0.1 for r in rows])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_IN, 3.1))
    a1.scatter(g[ov], p[ov], s=8, c='#1f77b4', label='Outline overlaps tumor')
    a1.scatter(g[~ov], p[~ov], s=14, c='#d62728', marker='x', label='Outline misses tumor')
    lim = [0, max(g.max(), p.max()) * 1.05]
    a1.plot(lim, lim, 'k--', lw=0.7)
    a1.set_xlim(lim), a1.set_ylim(lim)
    a1.set_xlabel('Expert longest diameter (mm)'), a1.set_ylabel('Predicted longest diameter (mm)')
    a1.legend(loc='upper left', frameon=False, handletextpad=0.2)
    mean, diff = (g[ov] + p[ov]) / 2, p[ov] - g[ov]
    a2.scatter(mean, diff, s=8, c='#1f77b4')
    bias, sd = diff.mean(), diff.std()
    for v, ls, txt in ((bias, '-', f'Bias {bias:.1f}'), (bias - 1.96 * sd, '--', f'{bias - 1.96 * sd:.1f}'), (bias + 1.96 * sd, '--', f'{bias + 1.96 * sd:.1f}')):
        a2.axhline(v, color='k', ls=ls, lw=0.7)
        a2.text(1.01, v, f'{txt} mm'.replace('-', '−'), transform=a2.get_yaxis_transform(), va='center', ha='left')
    a2.set_xlabel('Mean of expert and predicted (mm)'), a2.set_ylabel('Predicted − expert (mm)')
    fig.tight_layout(pad=0.4, w_pad=1.5, rect=(0, 0, 0.9, 0.95))
    panels(fig, (0.0, 0.46))
    save(fig, 3)


# ─── Fig. 4: confusion matrices ─────────────────────────────────────────────
def fig4():
    M = metrics()
    short = ['No tumor', 'Glioma', 'Meningioma', 'Pituitary']
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 3.2))
    for ax, cmx in ((axes[0], np.array(M['kaggle_test']['confusion_rows_true_cols_pred'])),
                    (axes[1], np.array(M['figshare_test_type']['confusion_rows_true_cols_pred'])[1:, :])):
        # vector cells instead of imshow: no 600 dpi raster has to be resampled
        ax.pcolormesh(np.arange(5) - 0.5, np.arange(cmx.shape[0] + 1) - 0.5, cmx / np.maximum(cmx.sum(1, keepdims=True), 1),
                      cmap='Blues', vmin=0, vmax=1)
        ax.set_aspect('equal'), ax.invert_yaxis()
        for (i, j), v in np.ndenumerate(cmx):
            ax.text(j, i, str(v), ha='center', va='center', color='white' if v > cmx.sum(1)[i] * 0.6 else 'black')
        ax.set_xticks(range(4)), ax.set_xticklabels(short, rotation=30, ha='right')
        rows = short if cmx.shape[0] == 4 else short[1:]
        ax.set_yticks(range(len(rows))), ax.set_yticklabels(rows)
        ax.set_xlabel('Predicted'), ax.set_ylabel('True')
    fig.tight_layout(pad=0.4, w_pad=2.5, rect=(0, 0, 1, 0.95))
    panels(fig)
    save(fig, 4)


# ─── Fig. 5: training curves and reliability diagram ────────────────────────
def fig5():
    hist = json.load(open(os.path.join(ROOT, 'runs', 'seg2d', 'history.json')))
    rows = list(csv.DictReader(open(os.path.join(PAPER, 'kaggle_test_per_image.csv'))))
    p = np.array([float(r['p_tumor_seg2d']) for r in rows])
    y = np.array([r['true'] != 'No tumor' for r in rows])
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    edges = np.linspace(0.5, 1.0, 6)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 4)
    fig = plt.figure(figsize=(W_IN, 3.0))
    a1 = fig.add_axes([0.08, 0.15, 0.38, 0.77])
    ep = [h['epoch'] for h in hist]
    a1.plot(ep, [h['val_dice_figshare'] for h in hist], 'o-', ms=3, label='Dice (figshare)')
    a1.plot(ep, [h['val_cls_acc_kaggle'] for h in hist], 's--', ms=3, label='Accuracy (Kaggle)')
    a1.plot(ep, [h['val_cls_acc_figshare'] for h in hist], '^:', ms=3, label='Type accuracy (figshare)')
    a1.set_xlabel('Epoch'), a1.set_ylabel('Validation metric'), a1.set_ylim(0.4, 1.0)
    a1.legend(loc='lower right', frameon=False)
    a2 = fig.add_axes([0.6, 0.42, 0.38, 0.5])
    a3 = fig.add_axes([0.6, 0.15, 0.38, 0.2], sharex=a2)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = [correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(5)]
    cnt = [int((idx == b).sum()) for b in range(5)]
    a2.plot([0.5, 1], [0.5, 1], 'k--', lw=0.7, label='Perfect calibration')
    a2.bar(centers, acc, width=0.09, color='#4c78a8', label='Observed accuracy')
    a2.set_ylim(0, 1.4), a2.set_yticks([0, 0.5, 1.0]), a2.set_ylabel('Accuracy')
    a2.legend(loc='upper left', frameon=False, ncol=2, columnspacing=1.0, handlelength=1.5)
    a2.tick_params(labelbottom=False)
    a3.bar(centers, cnt, width=0.09, color='#999999')
    a3.set_yscale('log'), a3.set_ylabel('Images'), a3.set_xlabel('Confidence'), a3.set_ylim(0.8, 60000)
    for c, n in zip(centers, cnt):
        a3.text(c, n * 1.3, str(n), ha='center')
    panels(fig)
    save(fig, 5)


# ─── Fig. 6: system output for two unseen test patients ─────────────────────
def fig6():
    import base64
    import torch
    from seg2d import CLASSES, INPUT_SIZE, load_seg2d, predict_slice, prepare_slice
    from analysis_2d import analyze_slice, render_overlay
    torch.set_num_threads(4)
    model, _ = load_seg2d(os.path.join(ROOT, 'models', 'seg2d_best.pth'))
    cases = [('dummy_data/Testing/meningioma/Te-me_0125.jpg', '(a)'), ('dummy_data/Testing/pituitary/Te-pi_0059.jpg', '(b)')]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 3.9))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.005, wspace=0.03)
    for ax, (p, tag) in zip(axes, cases):
        with Image.open(os.path.join(ROOT, p)) as im:
            im.load()
            sl, info = prepare_slice(im)
            prob, c = predict_slice(model, sl)
            a = analyze_slice(sl, prob, c, CLASSES)
            ov = Image.open(io.BytesIO(base64.b64decode(render_overlay(im, a, info, INPUT_SIZE))))
        ax.imshow(ov), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{tag} {a['type_label']} ({a['type_confidence']:.0%}); {a['side']}\n"
                     f"{a['longest_mm']:.1f} × {a['perpendicular_mm']:.1f} mm, ~{a['volume_est_ml']:.1f} mL (estimated scale)", pad=4)
    save(fig, 6)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True), os.makedirs(PREVIEW, exist_ok=True)
    todo = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
    for n in todo:
        globals()[f'fig{n}']()
