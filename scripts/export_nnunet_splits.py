"""
Write nnU-Net v2 `splits_final.json` from our split CSV so the nnU-Net baseline uses
exactly the same 5 folds; the held-out test cases are excluded from nnU-Net training
and predicted afterwards with `nnUNetv2_predict`.

    python scripts/export_nnunet_splits.py --splits splits/brats2021_splits.csv \
        --out $nnUNet_preprocessed/Dataset501_BraTS2021/splits_final.json

Case ids must match the nnU-Net dataset case names (keep the BraTS2021_xxxxx ids).
"""
import argparse
import csv
import json


def main(args):
    with open(args.splits, newline="") as f:
        rows = list(csv.DictReader(f))
    folds = sorted({r["split"] for r in rows if r["split"] != "test"}, key=int)
    dev = [r for r in rows if r["split"] != "test"]
    splits = [{"train": [r["case_id"] for r in dev if r["split"] != k],
               "val": [r["case_id"] for r in dev if r["split"] == k]} for k in folds]
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    test = [r["case_id"] for r in rows if r["split"] == "test"]
    print(f"wrote {len(splits)} folds to {args.out}; keep these {len(test)} test cases out of nnU-Net's imagesTr")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
