"""
Evaluates the web app's detection model (best_model.pth) on a folder of labelled 2D images,
using the app's own preprocessing (dataset.image_to_volume).

    python src/evaluate_app_model.py --test_dir dummy_data/Testing --train_dir data/train

Folder layout: <test_dir>/<class>/*.jpg; the class `notumor` (or `no_tumor`) is the negative
class, every other class is "tumor". Test images that are pixel-identical to an image in
--train_dir are flagged and metrics are also reported without them.
Writes <out_dir>/per_image.csv and <out_dir>/summary.json.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image

from dataset import image_to_volume
from model import BrainTumorClassifier

NEGATIVE = {'notumor', 'no_tumor'}


def md5(path):
    with Image.open(path) as img:
        return hashlib.md5(np.asarray(img.convert('L')).tobytes()).hexdigest()


def binary_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    d = (p >= thr).astype(int)
    tp, fn = int(((d == 1) & (y == 1)).sum()), int(((d == 0) & (y == 1)).sum())
    tn, fp = int(((d == 0) & (y == 0)).sum()), int(((d == 1) & (y == 0)).sum())
    out = dict(n=len(y), n_tumor=int(y.sum()), n_no_tumor=int((1 - y).sum()), tp=tp, fn=fn, tn=tn, fp=fp,
               accuracy=(tp + tn) / max(len(y), 1),
               sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
               precision=tp / max(tp + fp, 1))
    out['auc'] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None
    return out


def main(args):
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    img_size = tuple(ckpt.get('img_size', (64, 64, 64)))
    temperature = float(ckpt.get('temperature', 1.0))
    model = BrainTumorClassifier(in_channels=4, num_classes=2)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=True)
    model.eval()

    train_hashes = set()
    if args.train_dir:
        train_hashes = {md5(p) for p in glob.glob(os.path.join(args.train_dir, '*', '*'))
                        if p.lower().endswith(('.jpg', '.jpeg', '.png'))}

    rows = []
    for cls in sorted(os.listdir(args.test_dir)):
        for p in sorted(glob.glob(os.path.join(args.test_dir, cls, '*'))):
            if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append(dict(path=p.replace(os.sep, '/'), cls=cls, label=int(cls not in NEGATIVE)))

    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            x = []
            for r in batch:
                with Image.open(r['path']) as img:
                    x.append(image_to_volume(img, img_size))
            probs = torch.softmax(model.forward_heads(torch.stack(x))[0] / temperature, 1)[:, 1]
            for r, q in zip(batch, probs.tolist()):
                r['p_tumor'] = q
                r['in_train'] = md5(r['path']) in train_hashes

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'per_image.csv'), 'w') as f:
        f.write('path,class,label,p_tumor,pred,in_train\n')
        for r in rows:
            f.write(f"{r['path']},{r['cls']},{r['label']},{r['p_tumor']:.6f},{int(r['p_tumor'] >= 0.5)},{int(r['in_train'])}\n")

    clean = [r for r in rows if not r['in_train']]
    summary = {
        'checkpoint': args.ckpt, 'img_size': img_size, 'temperature': temperature,
        'test_dir': args.test_dir, 'train_dir': args.train_dir, 'threshold': 0.5,
        'all': binary_metrics([r['label'] for r in rows], [r['p_tumor'] for r in rows]),
        'not_in_train': binary_metrics([r['label'] for r in clean], [r['p_tumor'] for r in clean]),
        'per_class_predicted_tumor_rate': {
            c: float(np.mean([r['p_tumor'] >= 0.5 for r in rows if r['cls'] == c]))
            for c in sorted({r['cls'] for r in rows})},
        'misclassified': [r['path'] for r in rows if (r['p_tumor'] >= 0.5) != bool(r['label'])],
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ('all', 'not_in_train', 'per_class_predicted_tumor_rate',
                                              'misclassified')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='best_model.pth')
    parser.add_argument('--test_dir', default='dummy_data/Testing')
    parser.add_argument('--train_dir', default='data/train', help="flag test images also present here ('' to skip)")
    parser.add_argument('--out_dir', default='runs/app_model_eval')
    parser.add_argument('--batch_size', type=int, default=2, help='keep small on CPU: one 64^3 sample needs ~0.5 GB')
    main(parser.parse_args())
