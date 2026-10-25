"""
Convert raw BraTS 2021 cases into the .npy cache read by src/brats_dataset.py.

    python scripts/prepare_brats.py --raw_root D:/data/BraTS2021_Training_Data --out data/brats2021_cache

Per case: z-score over brain voxels per sequence, crop to the brain bounding box,
save image as float16 (4, X, Y, Z) and WT/TC/ET masks as uint8 (3, X, Y, Z).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from brats_dataset import list_raw_cases, load_raw_case  # noqa: E402


def main(args):
    os.makedirs(args.out, exist_ok=True)
    cases = list_raw_cases(args.raw_root)
    if not cases:
        raise SystemExit(f"No BraTS* case folders in {args.raw_root}")
    for i, case_id in enumerate(cases, 1):
        img_path = os.path.join(args.out, f"{case_id}_img.npy")
        if os.path.exists(img_path) and not args.overwrite:
            continue
        image, regions = load_raw_case(os.path.join(args.raw_root, case_id))
        if regions is None:
            print(f"[skip] {case_id}: no segmentation")
            continue
        np.save(img_path, image.astype(np.float16))
        np.save(os.path.join(args.out, f"{case_id}_seg.npy"), regions)
        if i % 50 == 0 or i == len(cases):
            print(f"[{i}/{len(cases)}] {case_id} shape={image.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True)
    p.add_argument("--out", default="data/brats2021_cache")
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
