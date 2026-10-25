"""
Statistics for the paper tables, computed from evaluate.py per_case.csv files.

Summarize one or more methods (mean, SD, 95% bootstrap CI per metric):
    python src/stats.py summary --runs Ours=runs/hybrid/fold0/eval_test/per_case.csv \
                                       SwinUNETR=runs/swinunetr/fold0/eval_test/per_case.csv

Paired comparison (Wilcoxon signed-rank on matched cases, Holm-corrected over metrics):
    python src/stats.py compare --a runs/hybrid/.../per_case.csv --b runs/swinunetr/.../per_case.csv

Cross-validation folds (mean +- SD of fold means):
    python src/stats.py folds --pattern "runs/hybrid/fold*/eval_val/per_case.csv"
"""
import argparse
import csv
import glob

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["dice_wt", "dice_tc", "dice_et", "hd95_wt", "hd95_tc", "hd95_et"]


def read_per_case(path):
    with open(path, newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def column(rows, metric, ids=None):
    ids = ids if ids is not None else list(rows)
    return np.array([float(rows[i][metric]) for i in ids])


def bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def cmd_summary(args):
    print("method," + ",".join(f"{m}_mean,{m}_sd,{m}_ci_lo,{m}_ci_hi" for m in METRICS))
    for spec in args.runs:
        name, path = spec.split("=", 1)
        rows = read_per_case(path)
        cells = []
        for m in METRICS:
            v = column(rows, m)
            lo, hi = bootstrap_ci(v)
            cells += [f"{v.mean():.4f}", f"{v.std(ddof=1):.4f}", f"{lo:.4f}", f"{hi:.4f}"]
        print(name + "," + ",".join(cells))


def cmd_compare(args):
    a, b = read_per_case(args.a), read_per_case(args.b)
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        print(f"[warn] comparing {len(ids)} shared cases (a={len(a)}, b={len(b)})")
    res = []
    for m in METRICS:
        va, vb = column(a, m, ids), column(b, m, ids)
        diff = va - vb
        p = wilcoxon(va, vb).pvalue if np.any(diff != 0) else 1.0
        res.append((m, va.mean(), vb.mean(), diff.mean(), p))
    adj = holm(np.array([r[4] for r in res]))
    print("metric,mean_a,mean_b,mean_diff,p_wilcoxon,p_holm,significant_0.05")
    for (m, ma, mb, d, p), pa in zip(res, adj):
        print(f"{m},{ma:.4f},{mb:.4f},{d:+.4f},{p:.4g},{pa:.4g},{pa < 0.05}")


def cmd_folds(args):
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files match {args.pattern}")
    fold_means = np.array([[column(read_per_case(p), m).mean() for m in METRICS] for p in paths])
    print(f"{len(paths)} folds")
    for j, m in enumerate(METRICS):
        print(f"{m}: {fold_means[:, j].mean():.4f} +- {fold_means[:, j].std(ddof=1) if len(paths) > 1 else 0:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--runs", nargs="+", required=True, help="Name=path/to/per_case.csv")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    f = sub.add_parser("folds")
    f.add_argument("--pattern", required=True)
    args = p.parse_args()
    {"summary": cmd_summary, "compare": cmd_compare, "folds": cmd_folds}[args.cmd](args)
