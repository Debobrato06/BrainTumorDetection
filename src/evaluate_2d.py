"""
Evaluate the 2D 4-class classifier (src/monai_training.py) on the held-out test split.
Replaces the old research_viz.py / research_dashboard.py, which drew random HD95
values, synthetic loss curves and "ground-truth masks" derived from saliency.

    python src/evaluate_2d.py --ckpt runs/vit2d/best.ckpt

Outputs in runs/vit2d/eval_test/:
    predictions.csv      path, true class, predicted class, class probabilities
    summary.json         accuracy, balanced accuracy, macro F1, macro AUC (+ 95% bootstrap CIs), top-label ECE
    confusion_matrix.png, reliability.png, loss_curves.png (from the CSVLogger file, if present)
    saliency_examples.png  gradient saliency on a few test images (qualitative only: no masks exist,
                           so no Dice/HD95 can be computed for this dataset)
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)

from metrics import expected_calibration_error  # noqa: E402
from monai_training import CLASSES, DATA_ROOT, SPLIT_CSV, HybridTumorClassifier, make_transforms, read_split  # noqa: E402


def bootstrap(fn, y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            vals.append(fn(y[idx], p[idx]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def saliency(model, x):
    x = x.clone().requires_grad_(True)
    logits = model(x)
    logits = logits[0] if isinstance(logits, (list, tuple)) else logits
    logits[0, logits.argmax(1).item()].backward()
    s = x.grad.abs().amax(dim=1)[0].cpu().numpy()
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = HybridTumorClassifier.load_from_checkpoint(args.ckpt, map_location=device)
    model = module.model.to(device).eval()
    out_dir = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    paths, labels = read_split(args.split_csv, args.data_root, args.split)
    tf = make_transforms(train=False)
    from monai.transforms import LoadImage
    loader = LoadImage(image_only=True)

    probs = []
    with torch.no_grad():
        for p in paths:
            x = tf(loader(p)).unsqueeze(0).to(device)
            logits = model(x)
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            probs.append(torch.softmax(logits.float(), 1)[0].cpu().numpy())
    probs = np.stack(probs)
    y = np.array(labels)
    pred = probs.argmax(1)

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred"] + [f"p_{c}" for c in CLASSES])
        for pth, t, pr, pb in zip(paths, y, pred, probs):
            w.writerow([pth, CLASSES[t], CLASSES[pr]] + [f"{v:.5f}" for v in pb])

    conf = probs.max(1)
    correct = (pred == y).astype(float)
    summary = {
        "checkpoint": args.ckpt, "split": args.split, "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "accuracy_ci95": bootstrap(lambda a, b: accuracy_score(a, b.argmax(1)), y, probs),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "macro_f1_ci95": bootstrap(lambda a, b: f1_score(a, b.argmax(1), average="macro"), y, probs),
        "macro_auc_ovr": roc_auc_score(y, probs, multi_class="ovr"),
        "top_label_ece": expected_calibration_error(conf, correct),
        "per_class": classification_report(y, pred, target_names=CLASSES, output_dict=True),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2, default=float))

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (n={len(y)})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # Reliability diagram (top-label confidence)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
    xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [correct[idx == b].mean() for b in range(10) if (idx == b).any()]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability (ECE = {summary['top_label_ece']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=300)
    plt.close(fig)

    # Loss curves from the real training log, if present
    logs = sorted(glob.glob(os.path.join(os.path.dirname(args.ckpt), "logs", "*", "metrics.csv")))
    if logs:
        import pandas as pd
        df = pd.read_csv(logs[-1])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for col in ("train/loss", "val/loss"):
            if col in df:
                d = df[["epoch", col]].dropna().groupby("epoch").mean()
                ax.plot(d.index + 1, d[col], label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=300)
        plt.close(fig)

    # Qualitative saliency, one test image per class
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(3 * len(CLASSES), 6))
    for c in range(len(CLASSES)):
        sel = np.flatnonzero(y == c)
        if not len(sel):
            continue
        i = sel[0]
        x = tf(loader(paths[i])).unsqueeze(0).to(device)
        with torch.enable_grad():
            sal = saliency(model, x)
        img = x[0, 0].detach().cpu().numpy()
        axes[0, c].imshow(img.T, cmap="gray")
        axes[0, c].set_title(f"true {CLASSES[c]}\npred {CLASSES[pred[i]]}", fontsize=9)
        axes[1, c].imshow(img.T, cmap="gray")
        axes[1, c].imshow(sal.T, cmap="jet", alpha=0.45)
        axes[1, c].set_title("gradient saliency", fontsize=9)
        for a in axes[:, c]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "saliency_examples.png"), dpi=300)
    plt.close(fig)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--split_csv", default=str(SPLIT_CSV))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    main(p.parse_args())
