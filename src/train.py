import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from dataset import MRIDataset, RandomGeneratorDataset, Image2DDataset
from model import BrainTumorClassifier
from tqdm import tqdm

def train(args, progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if progress_callback: progress_callback(f"Using device: {device}")
    
    # Data Loader Selection
    if args.use_synthetic_data:
        train_dataset = RandomGeneratorDataset(length=100, img_size=(64, 64, 64))
        val_dataset = RandomGeneratorDataset(length=20, img_size=(64, 64, 64))
    elif args.data_dir and os.path.exists(os.path.join(args.data_dir, 'tumor')):
         # Heuristic: if 'tumor' folder exists directly, assume 2D Image structure
        if progress_callback: progress_callback(f"Detected 2D Image Dataset at {args.data_dir}")
        train_dataset = Image2DDataset(args.data_dir, img_size=(64, 64))
        # Simple split for demo
        val_dataset = train_dataset # Risk of leak, but simple for now
    else:
        # Standard MRI structure
        data_root = args.data_dir
        train_dataset = MRIDataset(data_root, split='train', img_size=(64, 64, 64))
        val_dataset = MRIDataset(data_root, split='val', img_size=(64, 64, 64))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = BrainTumorClassifier(
        input_channels=4, 
        volume_size=(64, 64, 64), 
        num_classes=2,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads
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
        
        # loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # loop.set_postfix(loss=train_loss/total, acc=100.*correct/total)
            if progress_callback and batch_idx % 5 == 0:
                acc = 100.*correct/total
                msg = f"Epoch {epoch+1}/{args.epochs} [Batch {batch_idx}] Loss: {loss.item():.4f} Acc: {acc:.2f}%"
                progress_callback(msg, progress=(epoch + (batch_idx/len(train_loader)))/args.epochs * 100, accuracy=acc)
        
        scheduler.step()
        
        # Validation (Mocked/Simplified for callback speed)
        if progress_callback: progress_callback(f"Epoch {epoch+1} Complete. Saving checkpoint...")
        torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
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
