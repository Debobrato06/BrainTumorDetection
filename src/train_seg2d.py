"""Train TumorSegNet2D (src/seg2d.py): tumor mask + tumor type from one 2D MRI slice.

Data
  figshare (scripts/prepare_figshare.py): 3064 T1-CE slices, 233 patients, expert tumor masks,
      labels meningioma / glioma / pituitary. Split by patient: 80% train, 10% val, 10% test.
  Kaggle Brain Tumor MRI Dataset (dummy_data/Training): 4 classes, no masks. "notumor" images
      are used with an empty mask; tumor images only train the class head. 10% held out for
      validation. The Kaggle set re-uses figshare slices, so every Kaggle image that matches a
      figshare val/test slice is removed from training (thumbnail match, see find_duplicates).

Usage: python src/train_seg2d.py --epochs 14 --out models/seg2d_best.pth
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg2d import CLASSES, INPUT_SIZE, TumorSegNet2D, prepare_slice  # noqa: E402

KAGGLE_CLASS = {'notumor': 0, 'glioma': 1, 'meningioma': 2, 'pituitary': 3}
FIGSHARE_CLASS = {1: 2, 2: 1, 3: 3}   # figshare label -> CLASSES index


def build_folder_cache(root, out_path):
    """Kaggle-style folders (glioma/meningioma/notumor/pituitary) -> npz of prepared slices."""
    if os.path.exists(out_path):
        return dict(np.load(out_path))
    images, labels, paths = [], [], []
    for cname, cidx in KAGGLE_CLASS.items():
        for p in sorted(glob.glob(os.path.join(root, cname, '*'))):
            if not p.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            with Image.open(p) as im:
                images.append(prepare_slice(im)[0])
            labels.append(cidx)
            paths.append(p.replace('\\', '/'))
    data = {'images': np.stack(images), 'labels': np.array(labels, dtype=np.int64), 'paths': np.array(paths)}
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, **data)
    return data


def thumbnails(images, size=32):
    t = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA) for im in images]).astype(np.float32)
    t = t.reshape(len(t), -1)
    t -= t.mean(axis=1, keepdims=True)
    t /= t.std(axis=1, keepdims=True) + 1e-6
    return t


def find_duplicates(query_images, ref_images, max_dist=0.08):
    """Boolean per query image: True if a reference image has nearly the same 32x32 thumbnail
    (mean absolute difference of z-scored thumbnails below max_dist; copies measure < 0.05,
    different slices > 0.1). Also returns the distances."""
    q, r = thumbnails(query_images), thumbnails(ref_images)
    best = np.full(len(q), np.inf, dtype=np.float32)
    step = max(1, 25_000_000 // (len(r) * q.shape[1]))   # ~100 MB per block
    for i in range(0, len(q), step):
        d = np.abs(q[i:i + step, None, :] - r[None, :, :]).mean(axis=2)
        best[i:i + step] = d.min(axis=1)
    return best < max_dist, best


def patient_split(pids, seed=42, frac=(0.8, 0.1, 0.1)):
    uniq = np.array(sorted(set(pids.tolist())))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_tr, n_va = int(round(frac[0] * len(uniq))), int(round(frac[1] * len(uniq)))
    groups = {'train': set(uniq[:n_tr]), 'val': set(uniq[n_tr:n_tr + n_va]), 'test': set(uniq[n_tr + n_va:])}
    return {k: np.where(np.isin(pids, list(v)))[0] for k, v in groups.items()}


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment(img, mask, rng):
    s = img.shape[0]
    if rng.rand() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.85, 1.15)
    m = cv2.getRotationMatrix2D((s / 2, s / 2), angle, scale)
    m[:, 2] += rng.uniform(-0.08, 0.08, size=2) * s
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (s, s), flags=cv2.INTER_NEAREST, borderValue=0)
    x = img.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.5)
    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    if rng.rand() < 0.2:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.2))
    if rng.rand() < 0.3:
        x = x + rng.normal(0, 0.02, x.shape).astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32), mask


def make_batch(items, rng=None):
    """items: list of (img uint8, mask uint8 or None, label). -> tensors (x, y_mask, has_mask, y_cls)"""
    xs, ms, hm, ys = [], [], [], []
    for img, mask, label in items:
        has = mask is not None
        m = mask if has else np.zeros_like(img)
        if rng is not None:
            x, m = augment(img, m, rng)
        else:
            x = img.astype(np.float32) / 255.0
        xs.append(x)
        ms.append(m.astype(np.float32))
        hm.append(has)
        ys.append(label)
    return (torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ms))[:, None],
            torch.tensor(hm), torch.tensor(ys, dtype=torch.long))


def seg_loss(logits, target, has_mask):
    if not has_mask.any():
        return logits.sum() * 0.0
    lg, tg = logits[has_mask], target[has_mask]
    bce = F.binary_cross_entropy_with_logits(lg, tg)
    p = torch.sigmoid(lg)
    pos = tg.flatten(1).sum(1) > 0       # Dice only where a tumor is present
    if pos.any():
        inter = (p * tg).flatten(1).sum(1)[pos]
        den = (p.flatten(1).sum(1) + tg.flatten(1).sum(1))[pos]
        dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    else:
        dice = 0.0
    return bce + dice


@torch.no_grad()
def evaluate(model, images, masks, labels, batch=16):
    """Dice over slices with a mask (mean), class accuracy over all."""
    model.eval()
    dices, correct = [], 0
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch].astype(np.float32) / 255.0)[:, None]
        seg, cls = model(x)
        correct += int((cls.argmax(1).numpy() == labels[i:i + batch]).sum())
        if masks is not None:
            pred = (torch.sigmoid(seg)[:, 0] > 0.5).numpy()
            gt = masks[i:i + batch] > 0
            for p, g in zip(pred, gt):
                if g.any():
                    dices.append(2 * (p & g).sum() / (p.sum() + g.sum()))
    return (float(np.mean(dices)) if dices else None), correct / max(len(images), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figshare', default='external/figshare_256.npz')
    ap.add_argument('--kaggle_train', default='dummy_data/Training')
    ap.add_argument('--cache_dir', default='external')
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--kaggle_per_epoch', type=int, default=2500,
                    help='Kaggle images sampled per epoch (all figshare train slices are used every epoch)')
    ap.add_argument('--out', default='models/seg2d_best.pth')
    ap.add_argument('--run_dir', default='runs/seg2d')
    ap.add_argument('--max_minutes', type=float, default=0, help='stop after this many minutes (0 = no limit)')
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    torch.set_num_threads(os.cpu_count() or 4)
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    fs = dict(np.load(args.figshare))
    fs_cls = np.array([FIGSHARE_CLASS[int(l)] for l in fs['labels']])
    split = patient_split(fs['pids'])
    kg = build_folder_cache(args.kaggle_train, os.path.join(args.cache_dir, 'kaggle_train_256.npz'))

    # Remove Kaggle training images that are copies of figshare val/test slices
    held = np.concatenate([split['val'], split['test']])
    dup, _ = find_duplicates(kg['images'], fs['images'][held])
    kg_idx = np.where(~dup)[0]
    k_rng = np.random.RandomState(1)
    k_rng.shuffle(kg_idx)
    n_kval = len(kg_idx) // 10
    kg_val, kg_tr = kg_idx[:n_kval], kg_idx[n_kval:]

    info = {
        'figshare_patients': {k: sorted(set(fs['pids'][v].tolist())) for k, v in split.items()},
        'figshare_slices': {k: int(len(v)) for k, v in split.items()},
        'kaggle_removed_as_figshare_val_test_duplicates': int(dup.sum()),
        'kaggle_train': int(len(kg_tr)), 'kaggle_val': int(len(kg_val)),
        'kaggle_val_paths': kg['paths'][kg_val].tolist(),
    }
    with open(os.path.join(args.run_dir, 'split.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print(f"figshare slices {info['figshare_slices']}, kaggle train {len(kg_tr)}, val {len(kg_val)}, "
          f"removed duplicates {int(dup.sum())}", flush=True)

    model = TumorSegNet2D(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    fs_tr = split['train']
    steps_per_epoch = (len(fs_tr) + args.kaggle_per_epoch) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    cls_weight = torch.tensor([1.0, 1.0, 1.2, 1.0])

    best_score, history, t0 = -1.0, [], time.time()
    for epoch in range(args.epochs):
        model.train()
        items = [('fs', i) for i in fs_tr] + [('kg', i) for i in rng.choice(kg_tr, args.kaggle_per_epoch, replace=False)]
        rng.shuffle(items)
        run_loss, n = 0.0, 0
        for step in range(steps_per_epoch):
            chunk = items[step * args.batch:(step + 1) * args.batch]
            batch = []
            for src, i in chunk:
                if src == 'fs':
                    batch.append((fs['images'][i], fs['masks'][i], fs_cls[i]))
                else:
                    label = int(kg['labels'][i])
                    batch.append((kg['images'][i], np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8) if label == 0 else None, label))
            x, m, has, y = make_batch(batch, rng)
            seg, cls = model(x)
            loss = seg_loss(seg, m, has) + 0.5 * F.cross_entropy(cls, y, weight=cls_weight, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
            n += 1
            if step % 50 == 0:
                pmax = torch.sigmoid(seg.detach()).max().item()
                print(f'epoch {epoch + 1}/{args.epochs} step {step}/{steps_per_epoch} loss {run_loss / n:.4f} '
                      f'max tumor prob {pmax:.3f} ({(time.time() - t0) / 60:.1f} min)', flush=True)

        v = split['val']
        dice, fs_acc = evaluate(model, fs['images'][v], fs['masks'][v], fs_cls[v])
        _, kg_acc = evaluate(model, kg['images'][kg_val], None, kg['labels'][kg_val])
        score = dice + kg_acc
        history.append({'epoch': epoch + 1, 'loss': run_loss / max(n, 1), 'val_dice_figshare': dice,
                        'val_cls_acc_figshare': fs_acc, 'val_cls_acc_kaggle': kg_acc})
        print(f'== epoch {epoch + 1}: loss {run_loss / max(n, 1):.4f} | val Dice {dice:.4f} | '
              f'val acc figshare {fs_acc:.4f} kaggle {kg_acc:.4f}', flush=True)
        if score > best_score:
            best_score = score
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch + 1, 'input_size': INPUT_SIZE,
                        'classes': CLASSES, 'val_dice': dice, 'val_cls_acc_kaggle': kg_acc,
                        'val_cls_acc_figshare': fs_acc, 'split_file': os.path.join(args.run_dir, 'split.json')},
                       args.out)
            print(f'   saved {args.out}', flush=True)
        with open(os.path.join(args.run_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print('Time limit reached.', flush=True)
            break


if __name__ == '__main__':
    main()
