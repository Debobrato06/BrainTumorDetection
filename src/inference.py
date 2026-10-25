"""
Command-line detection with the app model (best_model.pth from src/train.py).

    python src/inference.py --input_path scan_4ch.nii.gz        # 4D NIfTI, 4 sequences (T1, T1ce, T2, FLAIR)
    python src/inference.py --input_dir BraTS2021_00000         # folder with *_t1/_t1ce/_t2/_flair.nii.gz

Only heads listed in the checkpoint's `trained_heads` are reported.
"""
import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
import torch

from brats_dataset import MODALITIES, zscore_nonzero
from model import BrainTumorClassifier


def load_volume(path=None, input_dir=None, img_size=(64, 64, 64), allow_single_sequence=False):
    """Returns a (1, 4, *img_size) z-scored tensor. A 3D (single-sequence) file is only
    accepted with allow_single_sequence=True, in which case it is copied into all 4 channels;
    the model was not trained that way, so such predictions are unreliable."""
    if input_dir:
        chans = []
        for mod in MODALITIES:
            matches = [m for m in glob(os.path.join(input_dir, f"*_{mod}.nii*"))]
            if not matches:
                raise FileNotFoundError(f"No *_{mod}.nii(.gz) in {input_dir}")
            chans.append(nib.load(matches[0]).get_fdata(dtype=np.float32))
        data = np.stack(chans)
    else:
        data = nib.load(path).get_fdata(dtype=np.float32)
        if data.ndim == 4 and data.shape[-1] == 4:
            data = data.transpose(3, 0, 1, 2)
        elif data.ndim == 4 and data.shape[0] == 4:
            pass
        elif data.ndim == 3 and allow_single_sequence:
            print("[warn] single-sequence volume copied into 4 channels; results are unreliable")
            data = np.stack([data] * 4)
        else:
            raise ValueError(f"Expected 4 sequences (4D NIfTI), got shape {data.shape}. "
                             "Use --input_dir with separate T1/T1ce/T2/FLAIR files.")
    data = np.stack([zscore_nonzero(c) for c in data])
    tensor = torch.from_numpy(data).unsqueeze(0)
    return torch.nn.functional.interpolate(tensor, size=img_size, mode='trilinear', align_corners=False)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    img_size = tuple(ckpt.get('img_size', (64, 64, 64)))
    heads = set(ckpt.get('trained_heads', ['cls']))
    temperature = float(ckpt.get('temperature', 1.0))

    model = BrainTumorClassifier(input_channels=4, num_classes=2, volume_size=img_size)
    model.load_state_dict(state, strict=True)  # fail loudly instead of running random weights
    model.to(device).eval()

    x = load_volume(args.input_path, args.input_dir, img_size).to(device)
    with torch.no_grad():
        cls_logits, _, grade_logits = model(x)[:3]
        probs = torch.softmax(cls_logits / temperature, dim=1)[0]
    pred = int(probs.argmax())
    print("-" * 30)
    print(f"PREDICTION: {'LESION DETECTED' if pred == 1 else 'NO LESION DETECTED'}")
    print(f"P(lesion): {probs[1].item():.4f}  (temperature {temperature:.2f})")
    if 'grade' in heads:
        g = int(torch.softmax(grade_logits, 1)[0].argmax())
        print(f"ESTIMATED GRADE: {'High-Grade (HGG)' if g == 1 else 'Low-Grade (LGG)'}")
    else:
        print("GRADE: not available (grading head was not trained)")
    print("Research prototype; not for clinical use.")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_path", type=str, help="4D NIfTI with 4 sequences")
    src.add_argument("--input_dir", type=str, help="folder with *_t1, *_t1ce, *_t2, *_flair NIfTI files")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    predict(parser.parse_args())
