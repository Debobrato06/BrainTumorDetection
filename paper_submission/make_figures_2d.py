"""Figures for the 2D paper (3_SpringerNature_BMC_MedImaging/figures):
fig1_pipeline (system overview), fig7_example_output (system output on unseen-patient images), and
copies of the evaluation figures made by src/paper_metrics_seg2d.py.
Usage: python paper_submission/make_figures_2d.py"""
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, '3_SpringerNature_BMC_MedImaging', 'figures')
sys.path.insert(0, os.path.join(ROOT, 'src'))
plt.rcParams.update({'font.family': 'serif'})


def box(ax, x, y, w, h, text, fc, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec='#333', lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8, lw=0.8, color='#333'))


def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 14.4), ax.set_ylim(0, 5.8), ax.axis('off')
    box(ax, 0.1, 3.4, 2.0, 1.6, 'Input\n2D MRI slice\n(JPG / PNG)', '#e8eef7')
    box(ax, 2.5, 3.4, 2.3, 1.6, 'Preprocessing\ngrayscale, pad to\nsquare, 256 x 256,\n[0, 1]', '#e8eef7')
    box(ax, 5.2, 3.4, 3.0, 1.6, 'ResNet34 encoder\n(ImageNet weights)\n+ U-Net decoder\n24.4 M parameters', '#cfe3f5')
    box(ax, 8.7, 4.3, 2.3, 0.9, 'Classification head\nno tumor / glioma /\nmeningioma / pituitary', '#efd9f0', 6.0)
    box(ax, 8.7, 3.2, 2.3, 0.9, 'Segmentation head\ntumor probability', '#d8efd3')
    for a, b in [((2.1, 4.2), (2.5, 4.2)), ((4.8, 4.2), (5.2, 4.2)), ((8.2, 4.5), (8.7, 4.7)), ((8.2, 3.9), (8.7, 3.65))]:
        arrow(ax, a, b)
    box(ax, 11.4, 3.4, 2.9, 1.8, 'Decision\nP(tumor) = 1 − P(no tumor)\ntype = arg max\n(flip test-time\naveraging)', '#efd9f0', 6.2)
    arrow(ax, (11.0, 4.75), (11.4, 4.5))
    # Post-processing row
    box(ax, 0.1, 0.4, 3.1, 2.2, 'Head mask (Otsu, largest\ncomponent) and view check\n(mirror correlation:\naxial/coronal vs sagittal)', '#f6efd9', 6.2)
    box(ax, 3.6, 0.4, 3.1, 2.2, 'Tumor mask\nthreshold 0.5, inside head,\nsmall components removed;\nlow-confidence flag if mean\nprobability < 0.8', '#d8efd3', 6.2)
    box(ax, 7.1, 0.4, 3.4, 2.2, 'Measurements\nlongest / perpendicular\ndiameter, area, ellipsoid volume;\nmm scale from DICOM spacing or\nhead width (150 mm, ±15%)', '#f6efd9', 6.2)
    box(ax, 10.9, 0.4, 3.4, 2.2, 'Report\nside (radiological convention),\ndepth, midline crossing,\nrule-based concern score,\noverlay image', '#f3f3f3', 6.2)
    arrow(ax, (9.85, 3.2), (5.2, 2.6))
    arrow(ax, (3.2, 1.5), (3.6, 1.5)), arrow(ax, (6.7, 1.5), (7.1, 1.5)), arrow(ax, (10.5, 1.5), (10.9, 1.5))
    arrow(ax, (3.5, 3.4), (1.6, 2.6))
    fig.tight_layout(pad=0.1)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig1_pipeline.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def example_output():
    """System output (overlay with the longest diameter, side, size, type) on two unseen-patient images."""
    import base64
    import io
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, 'src', 'pipeline'))
    from seg2d import CLASSES, INPUT_SIZE, load_seg2d, predict_slice, prepare_slice
    from analysis_2d import analyze_slice, render_overlay
    model, _ = load_seg2d(os.path.join(ROOT, 'models', 'seg2d_best.pth'))
    cases = [('dummy_data/Testing/meningioma/Te-me_0125.jpg', '(a)'), ('dummy_data/Testing/pituitary/Te-pi_0059.jpg', '(b)')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.7))
    for ax, (p, tag) in zip(axes, cases):
        with Image.open(os.path.join(ROOT, p)) as im:
            im.load()
            sl, info = prepare_slice(im)
            prob, c = predict_slice(model, sl)
            a = analyze_slice(sl, prob, c, CLASSES)
            ov = Image.open(io.BytesIO(base64.b64decode(render_overlay(im, a, info, INPUT_SIZE))))
        ax.imshow(ov), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{tag} {a['type_label']} ({a['type_confidence']:.0%}); {a['side']}\n"
                     f"{a['longest_mm']:.1f} x {a['perpendicular_mm']:.1f} mm, ~{a['volume_est_ml']:.1f} mL (estimated scale)",
                     fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig7_example_output.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def confusion():
    import json
    import numpy as np
    M = json.load(open(os.path.join(ROOT, 'runs', 'seg2d_paper', 'metrics.json')))
    short = ['No tumor', 'Glioma', 'Meningioma', 'Pituitary']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, cmx, title in ((axes[0], np.array(M['kaggle_test']['confusion_rows_true_cols_pred']), f"Kaggle Testing (n = {M['kaggle_test']['n']})"),
                           (axes[1], np.array(M['figshare_test_type']['confusion_rows_true_cols_pred'])[1:, :], f"figshare test patients (n = {int(np.array(M['figshare_test_type']['confusion_rows_true_cols_pred']).sum())})")):
        ax.imshow(cmx / np.maximum(cmx.sum(1, keepdims=True), 1), cmap='Blues', vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cmx):
            ax.text(j, i, str(v), ha='center', va='center', color='white' if v > cmx.sum(1)[i] * 0.6 else 'black', fontsize=7)
        ax.set_xticks(range(4)), ax.set_xticklabels(short, rotation=30, fontsize=7)
        rows = short if cmx.shape[0] == 4 else short[1:]
        ax.set_yticks(range(len(rows))), ax.set_yticklabels(rows, fontsize=7)
        ax.set_xlabel('Predicted', fontsize=8), ax.set_ylabel('True', fontsize=8), ax.set_title(title, fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig5_confusion.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def training_calibration():
    """Validation curves and a reliability diagram of P(tumor) with five confidence bins; the
    histogram underneath shows how many test images fall in each bin."""
    import csv
    import json
    import numpy as np
    hist = json.load(open(os.path.join(ROOT, 'runs', 'seg2d', 'history.json')))
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'runs', 'seg2d_paper', 'kaggle_test_per_image.csv'))))
    p = np.array([float(r['p_tumor_seg2d']) for r in rows])
    y = np.array([r['true'] != 'No tumor' for r in rows])
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5) == y
    edges = np.linspace(0.5, 1.0, 6)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 4)
    fig = plt.figure(figsize=(7.0, 2.9))
    a1 = fig.add_axes([0.07, 0.17, 0.4, 0.73])
    ep = [h['epoch'] for h in hist]
    a1.plot(ep, [h['val_dice_figshare'] for h in hist], 'o-', ms=3, label='Val Dice (figshare)')
    a1.plot(ep, [h['val_cls_acc_kaggle'] for h in hist], 's-', ms=3, label='Val accuracy (Kaggle)')
    a1.plot(ep, [h['val_cls_acc_figshare'] for h in hist], '^-', ms=3, label='Val type accuracy (figshare)')
    a1.set_xlabel('Epoch', fontsize=8), a1.set_ylim(0.4, 1.0), a1.legend(fontsize=6), a1.tick_params(labelsize=7)
    a1.set_title('Validation during training', fontsize=8)
    a2 = fig.add_axes([0.58, 0.40, 0.38, 0.50])
    a3 = fig.add_axes([0.58, 0.17, 0.38, 0.17], sharex=a2)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = [correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(5)]
    cnt = [int((idx == b).sum()) for b in range(5)]
    a2.plot([0.5, 1], [0.5, 1], 'k--', lw=0.7, label='perfect calibration')
    a2.bar(centers, acc, width=0.09, color='#4c78a8', alpha=0.85, label='observed accuracy')
    a2.set_ylim(0, 1.05), a2.set_ylabel('Accuracy', fontsize=8), a2.legend(fontsize=6, loc='upper left')
    a2.tick_params(labelbottom=False, labelsize=7)
    ece = sum(c / len(p) * abs(conf[idx == b].mean() - correct[idx == b].mean()) for b, c in enumerate(cnt) if c)
    a2.set_title(f'Reliability of P(tumor), Kaggle Testing (ECE {ece:.3f}, 5 bins)', fontsize=8)
    a3.bar(centers, cnt, width=0.09, color='#999999')
    a3.set_yscale('log'), a3.set_ylabel('Images', fontsize=7), a3.set_xlabel('Confidence', fontsize=8), a3.tick_params(labelsize=7)
    for c, n in zip(centers, cnt):
        a3.text(c, n * 1.2, str(n), ha='center', fontsize=6)
    a3.set_ylim(0.8, 5000)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig6_training_calibration.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    pipeline()
    example_output()
    src = os.path.join(ROOT, 'runs', 'seg2d_paper', 'figures')
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), OUT)
    training_calibration()   # replaces the copied version
    confusion()              # replaces the copied version (no thousands separator, BMC style)
    print(sorted(os.listdir(OUT)))
