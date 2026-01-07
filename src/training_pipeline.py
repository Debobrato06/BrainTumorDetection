import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from dataset_research import ResearchMockDataset
from lightning_module import TumorResearchModule
import torch

def main(args):
    # Reproducibility
    pl.seed_everything(42)
    
    # Data Setup
    print("Initializing Research Dataset...")
    # For real use, substitute with MRIDataset logic
    dataset = ResearchMockDataset(length=100) 
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model Setup
    print(f"Initializing Hybrid Model (Mode: {args.mode})...")
    model = TumorResearchModule(
        lr=args.lr,
        max_epochs=args.epochs,
        mode=args.mode,
        mask_ratio=args.mask_ratio
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tumor-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='train/loss' # Monitoring train loss for mock
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer
    # DDP Strategy and Mixed Precision as requested
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    strategy = 'ddp' if devices > 1 else 'auto'
    
    print(f"Starting Trainer with accelerator={accelerator}, devices={devices}, strategy={strategy}...")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=16 if accelerator == 'gpu' else 32, # Mixed Precision
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=5
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("Training Complete.")
    
    # Optional: Demonstrate/Test Uncertainty or Transfer mechanism
    if args.mode == 'finetune':
        print("Running Uncertainty Estimation on a sample batch...")
        batch = next(iter(val_loader))
        x, _ = batch
        model.eval()
        x = x.to(model.device)
        mean_seg, uncertainty = model.model.get_uncertainty_map(x)
        print(f"Uncertainty Map Stat: Mean Variance {uncertainty.mean():.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='finetune', choices=['pretrain', 'finetune'], help="SSL Pretrain or Finetune")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio for SSL")
    
    args = parser.parse_args()
    main(args)
