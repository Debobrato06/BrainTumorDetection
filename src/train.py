"""
Trains the web app's detection model (BrainTumorClassifier) on the 2D tumor / no_tumor
images. Only the detection head gets a loss; the checkpoint records this in
`trained_heads`, so the app does not display the untrained segmentation / grade / type
outputs. For the paper's 3D experiments use training_pipeline.py instead.

    python src/train.py --data_dir data/train --epochs 20
    python src/train.py --use_synthetic_data --epochs 1        # smoke test, writes smoke_test_model.pth
"""
import argparse
import hashlib
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import Image2DDataset, RandomGeneratorDataset
from model import BrainTumorClassifier


def dedup_and_split(data_dir, val_frac=0.15, seed=42):
    """Exact duplicates (same pixels) collapse to one file before a stratified train/val split."""
    files = Image2DDataset(data_dir).files
    seen, unique = set(), []
    for path, label in files:
        h = hashlib.md5(np.asarray(Image.open(path).convert('L')).tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append((path, label))
    print(f"{len(files)} images, {len(unique)} after removing exact duplicates")
    labels = [lab for _, lab in unique]
    tr, va = train_test_split(unique, test_size=val_frac, random_state=seed, stratify=labels)
    return tr, va


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, targets = [], []
    for data, target in loader:
        logits.append(model.forward_heads(data.to(device))[0].float().cpu())
        targets.append(target)
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits, targets):
    """Temperature scaling (Guo et al., 2017) fitted on validation logits by NLL."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def val_accuracy(ckpt_path, val_files, device, batch_size=8):
    """Accuracy of a saved checkpoint on the validation images, each checkpoint scored with its
    own input size (img_size), or None if it cannot be loaded."""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BrainTumorClassifier(input_channels=4, num_classes=2).to(device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=True)
    except Exception as e:
        print(f"Could not load {ckpt_path} for comparison: {e}")
        return None
    d, h, w = tuple(ckpt.get('img_size', (64, 64, 64)))
    val_loader = DataLoader(Image2DDataset(None, img_size=(h, w), files=val_files, depth=d),
                            batch_size=batch_size if d <= 16 else 1)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in val_loader:
            correct += (model.forward_heads(data.to(device))[0].argmax(1).cpu() == target).sum().item()
            total += target.numel()
    return correct / max(total, 1)


def promote_if_better(candidate_path, deploy_path, val_files, device, report):
    """Replace the deployed model with the candidate only if the candidate finished training
    and is more accurate on the same validation images. The old file is kept as a backup.
    The deployed model may have seen these images in training, which favours keeping it."""
    cand = torch.load(candidate_path, map_location='cpu', weights_only=False)
    if 'temperature' not in cand:
        report(f"Candidate {candidate_path} did not finish training; {deploy_path} left unchanged.")
        return False
    report(f"Comparing {candidate_path} with deployed {deploy_path} on the validation split...")
    acc_new = val_accuracy(candidate_path, val_files, device)
    acc_old = val_accuracy(deploy_path, val_files, device)
    report(f"Validation accuracy: new {acc_new:.4f} vs deployed "
           + ("unreadable" if acc_old is None else f"{acc_old:.4f}"))
    if acc_old is not None and acc_new <= acc_old:
        report(f"New model is not better; {deploy_path} left unchanged (candidate kept as {candidate_path}).")
        return False
    backup = f"{os.path.splitext(deploy_path)[0]}.backup-{time.strftime('%Y%m%d-%H%M%S')}.pth"
    shutil.copy2(deploy_path, backup)
    os.replace(candidate_path, deploy_path)
    report(f"New model promoted to {deploy_path}; previous model backed up as {backup}.")
    return True


def train(args, progress_callback=None):
    def report(msg, **kw):
        print(msg)
        if progress_callback:
            progress_callback(msg, **kw)

    seed, val_frac = getattr(args, "seed", 42), getattr(args, "val_frac", 0.15)  # the app's TrainArgs omit these
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report(f"Using device: {device}")
    output_path = getattr(args, "output_path", None) or (
        "smoke_test_model.pth" if args.use_synthetic_data else "best_model.pth")
    # Never train straight into an existing real model: an interrupted or weaker run would
    # replace it. Train into a candidate file and promote it at the end only if it is better.
    deploy_path = None
    if not args.use_synthetic_data and os.path.exists(output_path):
        deploy_path = output_path
        output_path = f"{os.path.splitext(output_path)[0]}.candidate.pth"
        report(f"{deploy_path} exists; training into {output_path} (promoted only if better).")

    # The 2D slice is copied along depth, so depth carries no information; 16 (the minimum for
    # the 4 poolings) is ~4x cheaper than 64 and the default on CPU. The checkpoint records
    # img_size and the app builds its input to match.
    depth = getattr(args, "depth", None) or (64 if device.type == "cuda" else 16)
    img_size = (depth, 64, 64)
    batch_size = args.batch_size
    accumulation_steps = 1

    if args.use_synthetic_data:
        train_ds = RandomGeneratorDataset(length=20, img_size=img_size)
        val_ds = RandomGeneratorDataset(length=5, img_size=img_size)
        va_files = []
    elif os.path.exists(os.path.join(args.data_dir, 'tumor')):
        tr_files, va_files = dedup_and_split(args.data_dir, val_frac, seed)
        # Image2DDataset builds a (4, depth, 64, 64) pseudo-volume from one 2D slice
        train_ds = Image2DDataset(args.data_dir, img_size=(64, 64), files=tr_files, depth=depth)
        val_ds = Image2DDataset(args.data_dir, img_size=(64, 64), files=va_files, depth=depth)
        report(f"Train {len(train_ds)} / Val {len(val_ds)} images")
    else:
        raise FileNotFoundError(
            f"{args.data_dir} has no tumor/ and no_tumor/ folders. For 3D BraTS training use "
            "src/training_pipeline.py.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BrainTumorClassifier(input_channels=4, volume_size=img_size, num_classes=2).to(device)
    init_from = getattr(args, "init_from", None)
    if init_from:
        init = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(init.get('model_state_dict', init), strict=True)
        report(f"Fine-tuning from {init_from}")
    report(f"Input size {img_size}, batch {batch_size}, {args.epochs} epochs, lr {args.lr}")

    labels = torch.tensor([int(lab) for _, lab in getattr(train_ds, 'files', [])] or [0, 1])
    counts = torch.bincount(labels, minlength=2).float().clamp(min=1)
    criterion = nn.CrossEntropyLoss(weight=(counts.sum() / (2 * counts)).to(device))  # class imbalance
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), torch.as_tensor(target).to(device)
            cls_logits = model.forward_heads(data)[0]  # only the detection head is trained; skip the decoder
            loss = criterion(cls_logits, target) / accumulation_steps
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * accumulation_steps
            correct += cls_logits.argmax(1).eq(target).sum().item()
            total += target.size(0)
            if batch_idx % 5 == 0:
                report(f"Epoch {epoch + 1}/{args.epochs} [Batch {batch_idx}] Loss: {loss.item() * accumulation_steps:.4f}",
                       progress=(epoch + batch_idx / len(train_loader)) / args.epochs * 100,
                       accuracy=correct / total)
        scheduler.step()

        val_logits, val_targets = collect_logits(model, val_loader, device)
        val_acc = (val_logits.argmax(1) == val_targets).float().mean().item()
        report(f"Epoch {epoch + 1}: train loss {train_loss / len(train_loader):.4f}, "
               f"train acc {correct / total:.4f}, val acc {val_acc:.4f}", accuracy=val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'img_size': img_size,
                'val_acc': val_acc,
                'trained_heads': ['cls'],
                'synthetic_data': bool(args.use_synthetic_data),
            }, output_path)
            report(f"New best val acc {val_acc:.4f}; saved {output_path}")

    # Calibrate the best checkpoint on the validation set
    ckpt = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    val_logits, val_targets = collect_logits(model, val_loader, device)
    ckpt['temperature'] = fit_temperature(val_logits, val_targets)
    torch.save(ckpt, output_path)
    report(f"Training finished. Best val acc {best_val_acc:.4f}, temperature {ckpt['temperature']:.3f}. "
           f"Report test results only from a held-out test set, not this validation split.")

    promoted = None
    if deploy_path is not None:
        promoted = promote_if_better(output_path, deploy_path, va_files, device, report)
    return {'output_path': deploy_path if promoted else output_path, 'promoted': promoted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/train", help="folder with tumor/ and no_tumor/")
    parser.add_argument("--use_synthetic_data", action="store_true", help="random tensors; smoke test only")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--depth", type=int, default=None, help="pseudo-volume depth (default 64 on GPU, 16 on CPU)")
    parser.add_argument("--init_from", type=str, default=None, help="checkpoint to fine-tune from")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default=None)
    train(parser.parse_args())
