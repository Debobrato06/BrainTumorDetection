"""
BraTS 2021 data loading for the 3D experiments.

Expected raw layout (as downloaded from the BraTS 2021 Task 1 release):

    <raw_root>/BraTS2021_00000/BraTS2021_00000_{flair,t1,t1ce,t2,seg}.nii.gz

BraTS volumes are already skull-stripped, co-registered to the SRI24 atlas and
resampled to 1 mm isotropic by the organizers, so no registration is done here.

`scripts/prepare_brats.py` converts each case once into
<cache_root>/<case_id>_img.npy (float16, 4 x X x Y x Z, z-scored, cropped to the
brain bounding box) and <case_id>_seg.npy (uint8, 3 x X x Y x Z = WT/TC/ET).
The datasets below read that cache.
"""
import csv
import os
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")


# ----------------------------------------------------------------- preprocessing
def zscore_nonzero(volume):
    """Z-score using brain (non-zero) voxels only; background stays 0."""
    mask = volume != 0
    if not mask.any():
        return volume.astype(np.float32)
    v = volume[mask]
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def brats_label_to_regions(seg):
    """BraTS labels {0,1,2,4} (or 3 for ET in BraTS 2023) -> binary (3, ...) WT, TC, ET."""
    et = (seg == 4) | (seg == 3)
    tc = (seg == 1) | et
    wt = (seg == 2) | tc
    return np.stack([wt, tc, et]).astype(np.uint8)


def foreground_bbox(image, margin=2):
    """Bounding box of voxels that are non-zero in any channel. image: (C, X, Y, Z)."""
    nz = np.argwhere(np.any(image != 0, axis=0))
    if nz.size == 0:
        return tuple(slice(0, s) for s in image.shape[1:])
    lo = np.maximum(nz.min(0) - margin, 0)
    hi = np.minimum(nz.max(0) + margin + 1, image.shape[1:])
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def load_raw_case(case_dir):
    """Returns (image (4, X, Y, Z) float32 z-scored and cropped, regions (3, X, Y, Z) uint8 or None)."""
    case_id = os.path.basename(os.path.normpath(case_dir))
    chans = []
    for mod in MODALITIES:
        path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        chans.append(zscore_nonzero(nib.load(path).get_fdata(dtype=np.float32)))
    image = np.stack(chans)
    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
    regions = None
    if os.path.exists(seg_path):
        regions = brats_label_to_regions(np.asarray(nib.load(seg_path).dataobj))
    box = foreground_bbox(image)
    image = image[(slice(None),) + box]
    if regions is not None:
        regions = regions[(slice(None),) + box]
    return image, regions


def list_raw_cases(raw_root):
    return sorted(os.path.basename(p) for p in glob(os.path.join(raw_root, "BraTS*")) if os.path.isdir(p))


# ------------------------------------------------------------------------ splits
def read_splits(csv_path):
    """CSV columns: case_id, split ('test' or fold index 0..K-1), grade (optional: LGG/HGG/blank)."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def grade_to_int(grade):
    g = (grade or "").strip().upper()
    return {"LGG": 0, "HGG": 1}.get(g, -1)


def cases_for(rows, stage, fold):
    """stage: 'train' (all folds except `fold`), 'val' (fold `fold`), 'test' (held-out)."""
    if stage == "test":
        sel = [r for r in rows if r["split"] == "test"]
    elif stage == "val":
        sel = [r for r in rows if r["split"] == str(fold)]
    elif stage == "train":
        sel = [r for r in rows if r["split"] not in ("test", str(fold))]
    else:
        raise ValueError(stage)
    return [(r["case_id"], grade_to_int(r.get("grade"))) for r in sel]


# ---------------------------------------------------------------------- datasets
def _pad_to(arr, min_shape):
    pads = [(0, 0)] + [(0, max(0, m - s)) for s, m in zip(arr.shape[1:], min_shape)]
    return np.pad(arr, pads) if any(p[1] for p in pads) else arr


class BraTSTrainDataset(Dataset):
    """
    Random crops for training. With probability `fg_prob` the crop is centered on a
    tumor voxel; otherwise it is uniform, which also yields tumor-free crops so the
    detection head sees negatives. Detection label = WT voxels in crop >= min_tumor_voxels.
    """

    def __init__(self, cache_root, cases, crop_size=(128, 128, 128), fg_prob=0.66,
                 min_tumor_voxels=100, augment=True, seed=None):
        self.cache_root = cache_root
        self.cases = cases
        self.crop_size = tuple(crop_size)
        self.fg_prob = fg_prob
        self.min_tumor_voxels = min_tumor_voxels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.cases)

    def _load(self, case_id):
        img = np.load(os.path.join(self.cache_root, f"{case_id}_img.npy"), mmap_mode="r")
        seg = np.load(os.path.join(self.cache_root, f"{case_id}_seg.npy"), mmap_mode="r")
        return img, seg

    def _crop_start(self, seg, shape):
        if self.rng.random() < self.fg_prob and seg[0].any():
            fg = np.argwhere(seg[0])
            center = fg[self.rng.integers(len(fg))]
            start = [int(c) - cs // 2 for c, cs in zip(center, self.crop_size)]
        else:
            start = [int(self.rng.integers(0, max(1, s - cs + 1))) for s, cs in zip(shape, self.crop_size)]
        return [min(max(0, st), max(0, s - cs)) for st, s, cs in zip(start, shape, self.crop_size)]

    def __getitem__(self, idx):
        case_id, grade = self.cases[idx]
        img, seg = self._load(case_id)
        img = _pad_to(np.asarray(img, dtype=np.float32), self.crop_size)
        seg = _pad_to(np.asarray(seg), self.crop_size)

        st = self._crop_start(seg, img.shape[1:])
        sl = (slice(None),) + tuple(slice(s, s + c) for s, c in zip(st, self.crop_size))
        x, y = img[sl].copy(), seg[sl].astype(np.float32)

        if self.augment:
            for axis in (1, 2, 3):
                if self.rng.random() < 0.5:
                    x, y = np.flip(x, axis), np.flip(y, axis)
            scale = self.rng.uniform(0.9, 1.1, size=(x.shape[0], 1, 1, 1))
            shift = self.rng.uniform(-0.1, 0.1, size=(x.shape[0], 1, 1, 1))
            x = x * scale + shift
            if self.rng.random() < 0.2:
                x = x + self.rng.normal(0, 0.01, size=x.shape)
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)

        label = int(y[0].sum() >= self.min_tumor_voxels)
        return {
            "image": torch.from_numpy(x.astype(np.float32)),
            "seg": torch.from_numpy(y),
            "label": torch.tensor(label, dtype=torch.long),
            "grade": torch.tensor(grade, dtype=torch.long),
            "case_id": case_id,
        }


class BraTSEvalDataset(Dataset):
    """Whole (brain-cropped) volumes for sliding-window validation / testing; batch size 1."""

    def __init__(self, cache_root, cases):
        self.cache_root = cache_root
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id, grade = self.cases[idx]
        img = np.load(os.path.join(self.cache_root, f"{case_id}_img.npy")).astype(np.float32)
        seg = np.load(os.path.join(self.cache_root, f"{case_id}_seg.npy")).astype(np.float32)
        return {
            "image": torch.from_numpy(img),
            "seg": torch.from_numpy(seg),
            "grade": torch.tensor(grade, dtype=torch.long),
            "case_id": case_id,
        }
