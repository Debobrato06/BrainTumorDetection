import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from dataset import MRIDataset, RandomGeneratorDataset, Image2DDataset
from model import BrainTumorClassifier
from tqdm import tqdm

# --- Disable MKLDNN to avoid CPU memory spikes on some Windows builds ---
torch.backends.mkldnn.enabled = False


def train(args, progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if progress_callback:
        progress_callback(f"Using device: {device}")

    # ── OOM FIX: Smaller size for CPU ─────────────────────────────────────
    if device.type == 'cpu':
        img_size = (32, 32, 32)
        batch_size = 1
        train_len = 20
        accumulation_steps = 4  # Simulate larger batch without memory spike
        print(f"[OOM FIX] CPU detected — using img_size={img_size}, batch={batch_size}, accum={accumulation_steps}")
    else:
        img_size = (64, 64, 64)
        batch_size = args.batch_size
        train_len = 100
        accumulation_steps = 1
        print(f"[GPU MODE] img_size={img_size}, batch={batch_size}")

    # ── Data Loader Selection ──────────────────────────────────────────────
    if args.use_synthetic_data:
        train_dataset = RandomGeneratorDataset(length=train_len, img_size=img_size)
        val_dataset = RandomGeneratorDataset(length=max(5, train_len // 5), img_size=img_size)
    elif args.data_dir and os.path.exists(os.path.join(args.data_dir, 'tumor')):
        if progress_callback:
            progress_callback(f"Detected 2D Image Dataset at {args.data_dir}")
        train_dataset = Image2DDataset(args.data_dir, img_size=(64, 64))
        val_dataset = train_dataset  # Simple split for demo
    else:
        data_root = args.data_dir
        train_dataset = MRIDataset(data_root, split='train', img_size=img_size)
        val_dataset = MRIDataset(data_root, split='val', img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ── OOM FIX: Model with correct input size ─────────────────────────────
    model = BrainTumorClassifier(
        input_channels=4,
        volume_size=img_size,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()  # Zero grad at start for accumulation

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            cls_logits, seg_logits, grade_logits = model(data)
            loss = criterion(cls_logits, target)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Update weights only after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Stats (use unscaled loss for logging)
            train_loss += loss.item() * accumulation_steps
            _, predicted = cls_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Progress callback
            if progress_callback and batch_idx % 5 == 0:
                acc = 100. * correct / total
                msg = f"Epoch {epoch + 1}/{args.epochs} [Batch {batch_idx}] Loss: {loss.item() * accumulation_steps:.4f} Acc: {acc:.2f}%"
                progress_callback(
                    msg,
                    progress=(epoch + (batch_idx / len(train_loader))) / args.epochs * 100,
                    accuracy=acc
                )

        scheduler.step()

        # Save checkpoint
        if progress_callback:
            progress_callback(f"Epoch {epoch + 1} Complete. Saving checkpoint...")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'img_size': img_size,
        }, "best_model.pth")

    if progress_callback:
        progress_callback("Training finished. best_model.pth saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--use_synthetic_data", action="store_true", help="Use random numbers instead of real files")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)

    args = parser.parse_args()

    # Check if data exists, otherwise default to synthetic
    if not os.path.exists(args.data_dir) and not args.use_synthetic_data:
        print("Data directory not found, defaulting to synthetic mode.")
        args.use_synthetic_data = True

    train(args)