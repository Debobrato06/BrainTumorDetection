
import os
import glob
import pathlib
import torch
import pytorch_lightning as pl
import monai
from monai.data import DataLoader, ImageDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    RandRotate,
    RandFlip,
    RandZoom,
    ToTensor,
    EnsureType
)
from monai.networks.nets import ViT
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss

# Configuration
DATA_ROOT = pathlib.Path(r"D:\DebobratoResearch\BrainTumorDetection\dummy_data\Training")
IMG_SIZE = (224, 224)
BATCH_SIZE = 4 # Adjust based on GPU memory
MAX_EPOCHS = 50
NUM_CLASSES = 4
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

class BrainTumorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.batch_size = batch_size
        self.train_files = []
        self.val_files = []
        self.train_labels = []
        self.val_labels = []

    def prepare_data(self):
        # Scan files
        all_images = []
        all_labels = []
        
        # Verify directory exists
        if not self.data_dir.exists():
            print(f"Warning: Directory {self.data_dir} does not exist.")
            return

        for class_idx, class_name in enumerate(CLASSES):
            class_path = self.data_dir / class_name
            if not class_path.exists():
                print(f"Warning: Class folder {class_path} not found.")
                continue
            
            # Support common image formats
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
                images.extend(glob.glob(str(class_path / ext)))
            
            print(f"Found {len(images)} images for class {class_name}")
            all_images.extend(images)
            all_labels.extend([class_idx] * len(images))

        if len(all_images) == 0:
            print("No images found! Please ensure data is in D:\\DebobratoResearch\\BrainTumorDetection\\dummy_data\\Training")
            return

        # Split
        self.train_files, self.val_files, self.train_labels, self.val_labels = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        # Define Transforms
        self.train_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'), 
            # Force 1 channel if input is RGB
            monai.transforms.Lambda(func=lambda x: x[:1, ...] if x.shape[0] > 1 else x),
            ScaleIntensity(),
            Resize(spatial_size=IMG_SIZE),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ])
        
        self.val_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.Lambda(func=lambda x: x[:1, ...] if x.shape[0] > 1 else x),
            ScaleIntensity(),
            Resize(spatial_size=IMG_SIZE),
            EnsureType(),
        ])

        # MONAI ImageDataset automatically handles loading from list of paths
        self.train_ds = ImageDataset(
            image_files=self.train_files,
            labels=self.train_labels,
            transform=self.train_transforms
        )
        
        self.val_ds = ImageDataset(
            image_files=self.val_files,
            labels=self.val_labels,
            transform=self.val_transforms
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

class HybridTumorClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Using ViT (Vision Transformer) from MONAI
        # A robust Transformer Classification model
        self.model = ViT(
            in_channels=1,
            img_size=IMG_SIZE,
            patch_size=(16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=True,
            num_classes=num_classes,
            spatial_dims=2
        )
        
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        # x: (B, C, H, W)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if len(logits.shape) > 2:
            logits = logits.view(logits.size(0), -1)
        
        # y is typically Tensor(B,), convert if needed
        loss = self.loss_fn(logits, y.long())
        
        self.log('train/loss', loss, prog_bar=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train/acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if len(logits.shape) > 2:
            logits = logits.view(logits.size(0), -1)
            
        loss = self.loss_fn(logits, y.long())
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

def main():
    pl.seed_everything(42)
    
    print("Setting up Data Module...")
    dm = BrainTumorDataModule(data_dir=DATA_ROOT, batch_size=BATCH_SIZE)
    dm.prepare_data()
    # dm.setup() # Let trainer handle this to avoid 'stage' argument error
    
    # If using prepare_data to check files, we need to ensure it populates them 
    # Or just check count of all_images inside prepare_data.
    # For now, let's just run trainer.
    
    print("Initializing MONAI ViT Classifier...")
    model = HybridTumorClassifier(num_classes=NUM_CLASSES)
    
    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints_monai',
        filename='vit_tumor-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
        monitor='val/acc'
    )
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32, # Mixed precision
        callbacks=[checkpoint_callback],
        log_every_n_steps=5
    )
    
    print("Starting Training...")
    trainer.fit(model, dm)
    print("Training Finished.")

if __name__ == "__main__":
    main()
