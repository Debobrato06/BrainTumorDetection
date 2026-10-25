"""
Create a fixed, patient-level split: a held-out test set plus K cross-validation folds.

    python scripts/make_splits.py --cases_from data/brats2021_cache --out splits/brats2021_splits.csv
    # optional grade labels (CSV with columns case_id,grade where grade is LGG or HGG):
    python scripts/make_splits.py --cases_from data/brats2021_cache --grades my_grades.csv --out ...

Each BraTS case is one patient, so a case-level split is patient-level. Splits are
stratified by grade when grades are given. Commit the output CSV and never change it
after the first experiment; the test rows are only used for the final evaluation.
"""
import argparse
import csv
import os
import re

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def main(args):
    if os.path.isdir(args.cases_from):
        ids = sorted({re.sub(r"_(img|seg)\.npy$", "", f) for f in os.listdir(args.cases_from) if f.endswith("_img.npy")})
        if not ids:
            ids = sorted(d for d in os.listdir(args.cases_from) if d.startswith("BraTS"))
    else:
        with open(args.cases_from) as f:
            ids = sorted(line.strip() for line in f if line.strip())
    if not ids:
        raise SystemExit("No cases found")

    grades = {}
    if args.grades:
        with open(args.grades, newline="") as f:
            grades = {r["case_id"]: r["grade"].strip().upper() for r in csv.DictReader(f)}
    strat = np.array([grades.get(i, "NA") for i in ids])

    dev, test = train_test_split(ids, test_size=args.test_frac, random_state=args.seed,
                                 stratify=strat if args.grades else None)
    dev = np.array(dev)
    dev_strat = np.array([grades.get(i, "NA") for i in dev])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_of = {}
    for k, (_, va) in enumerate(skf.split(dev, dev_strat)):
        for i in dev[va]:
            fold_of[i] = str(k)
    for i in test:
        fold_of[i] = "test"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "grade"])
        for i in ids:
            w.writerow([i, fold_of[i], grades.get(i, "")])
    counts = {s: sum(v == s for v in fold_of.values()) for s in sorted(set(fold_of.values()))}
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases_from", required=True, help="cache dir, raw dir, or a text file with one case id per line")
    p.add_argument("--grades", default=None)
    p.add_argument("--out", default="splits/brats2021_splits.csv")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
