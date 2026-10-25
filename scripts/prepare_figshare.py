"""Convert the figshare brain tumor dataset (Cheng et al., 3064 T1-CE slices with expert tumor
masks, https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) into one cache file
for 2D segmentation training.

Each .mat file (MATLAB v7.3 / HDF5) holds cjdata.{image, tumorMask, label, PID}.
Images are min-max scaled to uint8 exactly like the dataset's README (and like the Kaggle JPGs
built from it), then resized to SIZE x SIZE. Pixel spacing is 0.49 mm at 512 px (README), so the
mm size of one cached pixel is 0.49 * original_width / SIZE.

Usage: python scripts/prepare_figshare.py --zips external/figshare_raw --out external/figshare_256.npz
"""
import argparse
import glob
import io
import os
import zipfile

import cv2
import h5py
import numpy as np

SIZE = 256
SPACING_AT_512 = 0.49  # mm per pixel, from the dataset README


def read_mat(raw_bytes):
    with h5py.File(io.BytesIO(raw_bytes), 'r') as f:
        cj = f['cjdata']
        image = np.array(cj['image']).T.astype(np.float64)   # HDF5 stores MATLAB arrays transposed
        mask = np.array(cj['tumorMask']).T.astype(np.uint8)
        label = int(np.array(cj['label']).item())
        pid = ''.join(chr(int(c)) for c in np.array(cj['PID']).ravel())
    return image, mask, label, pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zips', default='external/figshare_raw')
    ap.add_argument('--out', default='external/figshare_256.npz')
    args = ap.parse_args()

    images, masks, labels, pids, names, spacings = [], [], [], [], [], []
    for zpath in sorted(glob.glob(os.path.join(args.zips, '*'))):
        if not zipfile.is_zipfile(zpath):
            continue
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.endswith('.mat'):
                    continue
                image, mask, label, pid = read_mat(zf.read(name))
                lo, hi = image.min(), image.max()
                img8 = np.uint8(255.0 * (image - lo) / max(hi - lo, 1e-8))
                h, w = img8.shape
                images.append(cv2.resize(img8, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
                masks.append(cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
                labels.append(label)
                pids.append(pid)
                names.append(os.path.basename(name))
                spacings.append(SPACING_AT_512 * 512.0 / w)
        print(f'{zpath}: total {len(images)} slices')

    np.savez_compressed(
        args.out, images=np.stack(images), masks=np.stack(masks).astype(np.uint8),
        labels=np.array(labels, dtype=np.int64), pids=np.array(pids), names=np.array(names),
        spacing_mm=np.array(spacings, dtype=np.float32))
    print(f'Saved {len(images)} slices from {len(set(pids))} patients to {args.out}')


if __name__ == '__main__':
    main()
