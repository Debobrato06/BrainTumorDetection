"""Baseline for the 2D paper: a classic U-Net (Ronneberger et al. 2015) trained from scratch for
tumor segmentation only, on the same figshare patient split, input size, augmentation, loss and
number of epochs as TumorSegNet2D (src/train_seg2d.py), and evaluated with the same post-processing.

Usage: python src/train_unet_baseline.py --epochs 12 --out models/unet_baseline.pth
       python src/train_unet_baseline.py --eval_only --out models/unet_baseline.pth
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'pipeline'))
from train_seg2d import make_batch, patient_split, seg_loss  # noqa: E402
from analysis_2d import head_mask, tumor_mask_from_prob  # noqa: E402


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Four-level U-Net, base width 32, BatchNorm, bilinear upsampling."""
    def __init__(self, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([block(1, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList([block(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for d in self.dec:
            s = skips.pop()
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False), s], 1))
        return self.out(x)


@torch.no_grad()
def evaluate(model, images, masks):
    model.eval()
    dice = []
    for img, gt in zip(images, masks):
        p = torch.sigmoid(model(torch.from_numpy(img).float().div(255)[None, None]))[0, 0].numpy()
        pred = tumor_mask_from_prob(p, head_mask(img), 0.5)
        g = gt > 0
        dice.append(2 * (pred & g).sum() / max(pred.sum() + g.sum(), 1))
    return np.array(dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='models/unet_baseline.pth')
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    torch.set_num_threads(min(10, os.cpu_count() or 4))
    with np.load(os.path.join(ROOT, 'external', 'figshare_256.npz')) as z:
        images, masks, pids = z['images'], z['masks'], z['pids']
    split = patient_split(pids)
    model = UNet()
    out = os.path.join(ROOT, args.out)
    if not args.eval_only:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        tr = split['train']
        steps = len(tr) // args.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
        best, t0, hist = -1.0, time.time(), []
        for ep in range(args.epochs):
            model.train()
            order = rng.permutation(tr)
            tot = 0.0
            for s in range(steps):
                idx = order[s * args.batch:(s + 1) * args.batch]
                x, m, has, _ = make_batch([(images[i], masks[i], 1) for i in idx], rng)
                loss = seg_loss(model(x), m, has)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
                if s % 50 == 0:
                    print(f'epoch {ep + 1} step {s}/{steps} loss {tot / (s + 1):.4f} ({(time.time() - t0) / 60:.1f} min)', flush=True)
            vd = float(evaluate(model, images[split['val']], masks[split['val']]).mean())
            hist.append({'epoch': ep + 1, 'loss': tot / steps, 'val_dice': vd})
            print(f'== epoch {ep + 1}: loss {tot / steps:.4f} | val Dice {vd:.4f}', flush=True)
            if vd > best:
                best = vd
                torch.save({'model_state_dict': model.state_dict(), 'epoch': ep + 1, 'val_dice': vd}, out)
        json.dump(hist, open(os.path.join(ROOT, 'runs', 'seg2d_paper', 'unet_baseline_history.json'), 'w'), indent=1)
    ck = torch.load(out, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    te = split['test']
    d = evaluate(model, images[te], masks[te])
    res = {'epoch': ck['epoch'], 'val_dice': ck['val_dice'], 'parameters': int(sum(p.numel() for p in model.parameters())),
           'test_dice_mean': float(d.mean()), 'test_dice_median': float(np.median(d)),
           'test_dice_ge_0.5': float((d >= 0.5).mean()), 'test_dice_per_slice': d.round(4).tolist()}
    json.dump(res, open(os.path.join(ROOT, 'runs', 'seg2d_paper', 'unet_baseline_test.json'), 'w'), indent=1)
    print({k: v for k, v in res.items() if k != 'test_dice_per_slice'})


if __name__ == '__main__':
    main()
