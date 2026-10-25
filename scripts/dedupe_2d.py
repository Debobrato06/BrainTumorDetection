"""
Find exact and near-duplicate images in a 2D dataset (and leaks between Training/Testing).

    python scripts/dedupe_2d.py --root dummy_data --out splits/2d_duplicates.csv

Exact duplicates: identical MD5 of pixel data after grayscale conversion.
Near duplicates: 64-bit difference hash (dHash) within --max_distance bits.
The report lists each duplicate group with its files, classes and splits; any group
that spans Training and Testing is a leak and must be removed from one side, and any
group that spans classes has a label conflict.

    # also write a leak-free split (train / val / test) for the 2D experiments
    python scripts/dedupe_2d.py --root dummy_data --clean_split splits/kaggle4class_clean_split.csv
"""
import argparse
import csv
import hashlib
import os
from collections import defaultdict

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".bmp")


def dhash(img, size=8):
    small = np.asarray(img.resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def main(args):
    files = []
    for dirpath, _, names in os.walk(args.root):
        for n in names:
            if n.lower().endswith(EXTS):
                files.append(os.path.join(dirpath, n))
    print(f"{len(files)} images under {args.root}")

    recs = []
    for path in files:
        rel = os.path.relpath(path, args.root).replace("\\", "/")
        parts = rel.split("/")
        split = parts[0] if len(parts) > 2 else ""
        cls = parts[-2] if len(parts) > 1 else ""
        img = Image.open(path).convert("L")
        md5 = hashlib.md5(np.asarray(img).tobytes()).hexdigest()
        recs.append({"path": rel, "split": split, "class": cls, "md5": md5, "dhash": dhash(img)})

    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_md5 = defaultdict(list)
    for i, r in enumerate(recs):
        by_md5[r["md5"]].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            union(idx[0], j)

    if args.max_distance > 0:
        # Leader clustering (not transitive, so similar-looking scans do not chain into one
        # giant group): an image joins the first leader within max_distance bits, else it
        # becomes a leader. Candidates come from 4 x 16-bit bands; with max_distance <= 3
        # any match shares at least one band exactly.
        bands = defaultdict(list)
        for i, r in enumerate(recs):
            if find(i) != i and recs[find(i)]["md5"] == r["md5"]:
                continue  # exact duplicate already grouped
            h = r["dhash"]
            keys = [(b, (h >> (16 * b)) & 0xFFFF) for b in range(4)]
            leader = next((j for k in keys for j in bands[k]
                           if bin(h ^ recs[j]["dhash"]).count("1") <= args.max_distance), None)
            if leader is not None:
                union(i, leader)
            else:
                for k in keys:
                    bands[k].append(i)

    groups = defaultdict(list)
    for i in range(len(recs)):
        groups[find(i)].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]

    leaks = conflicts = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "path", "split", "class", "exact_md5", "cross_split_leak", "label_conflict"])
        for gi, g in enumerate(dup_groups):
            splits = {recs[i]["split"] for i in g}
            classes = {recs[i]["class"] for i in g}
            leak, conflict = len(splits) > 1, len(classes) > 1
            leaks += leak
            conflicts += conflict
            for i in g:
                w.writerow([gi, recs[i]["path"], recs[i]["split"], recs[i]["class"], recs[i]["md5"], leak, conflict])

    n_dup_files = sum(len(g) - 1 for g in dup_groups)
    print(f"{len(dup_groups)} duplicate groups; {n_dup_files} redundant files; "
          f"{leaks} groups leak across splits; {conflicts} groups have conflicting labels")
    print(f"report: {args.out}")

    if args.clean_split:
        write_clean_split(recs, groups, args.clean_split, args.val_frac, args.seed)


def write_clean_split(recs, groups, out_path, val_frac, seed):
    """
    One image per duplicate group, kept in a single split: a group touching Testing stays
    in test (its Training copies are dropped); groups with conflicting labels are dropped.
    A stratified validation set is carved out of the remaining Training images.
    """
    rng = np.random.default_rng(seed)
    keep = []
    dropped_conflict = 0
    for g in groups.values():
        if len({recs[i]["class"] for i in g}) > 1:
            dropped_conflict += 1
            continue
        test = [i for i in g if recs[i]["split"].lower().startswith("test")]
        keep.append((test or g)[0])
    rows = []
    by_class = defaultdict(list)
    for i in keep:
        r = recs[i]
        if r["split"].lower().startswith("test"):
            rows.append((r["path"], r["class"], "test"))
        else:
            by_class[r["class"]].append(r["path"])
    for cls, paths in by_class.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = int(round(len(paths) * val_frac))
        rows += [(pth, cls, "val") for pth in paths[:n_val]] + [(pth, cls, "train") for pth in paths[n_val:]]
    rows.sort()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "split"])
        w.writerows(rows)
    counts = defaultdict(int)
    for _, _, sp in rows:
        counts[sp] += 1
    print(f"clean split: {dict(counts)} ({dropped_conflict} conflicting groups dropped) -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="splits/2d_duplicates.csv")
    p.add_argument("--max_distance", type=int, default=0, help="dHash bits for near-duplicate CANDIDATES (review by eye; MRI slices look alike, so >0 gives false matches); 0 = exact only")
    p.add_argument("--clean_split", default=None, help="also write a deduplicated path,class,split CSV here")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
