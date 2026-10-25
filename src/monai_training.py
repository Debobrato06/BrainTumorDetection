
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
from pytorch_lightning.loggers import CSVLogger
from torch.nn import CrossEntropyLoss

# Configuration
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "dummy_data"   # holds the real Kaggle 4-class dataset (Training/, Testing/)
SPLIT_CSV = PROJECT_ROOT / "splits" / "kaggle4class_clean_split.csv"  # scripts/dedupe_2d.py --clean_split
IMG_SIZE = (224, 224)
BATCH_SIZE = 4 # Adjust based on GPU memory
MAX_EPOCHS = 50
NUM_CLASSES = 4
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def read_split(split_csv, data_root, split):
    """(paths, labels) for one split of the deduplicated split CSV (path,class,split)."""
    import csv
    paths, labels = [], []
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                paths.append(str(pathlib.Path(data_root) / r["path"]))
                labels.append(CLASSES.index(r["class"]))
    return paths, labels


def make_transforms(train):
    base = [
        EnsureChannelFirst(channel_dim='no_channel'),
        # Force 1 channel if input is RGB
        monai.transforms.Lambda(func=lambda x: x[:1, ...] if x.shape[0] > 1 else x),
        ScaleIntensity(),
        Resize(spatial_size=IMG_SIZE),
    ]
    if train:
        base += [
            RandRotate(range_x=0.26, prob=0.5, keep_size=True),  # radians (~15 degrees)
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    return Compose(base + [EnsureType()])


class BrainTumorDataModule(pl.LightningDataModule):
    """Train/val come from the leak-free split CSV; the test split is only used by evaluate_2d.py."""

    def __init__(self, data_dir=DATA_ROOT, split_csv=SPLIT_CSV, batch_size=16):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.split_csv = pathlib.Path(split_csv)
        self.batch_size = batch_size

    def setup(self, stage=None):
        if not self.split_csv.exists():
            raise FileNotFoundError(f"{self.split_csv} not found; run scripts/dedupe_2d.py --clean_split first")
        train_files, train_labels = read_split(self.split_csv, self.data_dir, "train")
        val_files, val_labels = read_split(self.split_csv, self.data_dir, "val")
        print(f"train={len(train_files)} val={len(val_files)}")
        self.train_labels = train_labels
        self.train_ds = ImageDataset(image_files=train_files, labels=train_labels, transform=make_transforms(True))
        self.val_ds = ImageDataset(image_files=val_files, labels=val_labels, transform=make_transforms(False))

    def class_weights(self):
        counts = torch.bincount(torch.tensor(self.train_labels), minlength=NUM_CLASSES).float()
        return counts.sum() / (NUM_CLASSES * counts)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

class HybridTumorClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4, class_weights=None):
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
        
        self.loss_fn = CrossEntropyLoss(weight=class_weights)

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
    pl.seed_everything(42, workers=True)

    print("Setting up Data Module...")
    dm = BrainTumorDataModule(batch_size=BATCH_SIZE)
    dm.setup()

    print("Initializing MONAI ViT Classifier...")
    model = HybridTumorClassifier(num_classes=NUM_CLASSES, class_weights=dm.class_weights())

    out_dir = PROJECT_ROOT / "runs" / "vit2d"
    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir,
        filename='best',
        save_top_k=1,
        mode='max',
        monitor='val/acc'
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback],
        logger=CSVLogger(out_dir, name="logs"),
        log_every_n_steps=5
    )

    print("Starting Training...")
    trainer.fit(model, dm)
    print(f"Training Finished. Best checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
