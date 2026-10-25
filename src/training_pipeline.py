"""
Config-driven training for the BraTS experiments (proposed model, baselines, ablations).

    # 1) self-supervised pre-training of the hybrid encoder (train folds only)
    python src/training_pipeline.py --config configs/hybrid_pretrain.yaml --fold 0 --mode pretrain
    # 2) supervised fine-tuning, initialised from (1)
    python src/training_pipeline.py --config configs/hybrid.yaml --fold 0 \
        --pretrained runs/hybrid_pretrain/fold0/last.ckpt
    # baselines / ablations: same command with their config
    python src/training_pipeline.py --config configs/swinunetr.yaml --fold 0

Outputs go to runs/<experiment>/fold<k>/ : checkpoints, metrics.csv (CSVLogger), the resolved config.
"""
import argparse
import os
import shutil

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from brats_dataset import BraTSEvalDataset, BraTSTrainDataset, cases_for, read_splits
from lightning_module import TumorResearchModule


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)
    tr, dc = cfg["train"], cfg["data"]
    if args.epochs:
        tr["epochs"] = args.epochs
    seed = tr.get("seed", 42) + args.fold
    pl.seed_everything(seed, workers=True)

    exp = cfg.get("experiment", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = os.path.join(args.runs_dir, exp, f"fold{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    rows = read_splits(dc["splits_csv"])
    train_cases = cases_for(rows, "train", args.fold)
    val_cases = cases_for(rows, "val", args.fold)
    print(f"[data] fold {args.fold}: {len(train_cases)} train / {len(val_cases)} val cases")

    train_ds = BraTSTrainDataset(dc["cache_root"], train_cases, crop_size=dc["crop_size"],
                                 fg_prob=dc.get("fg_prob", 0.66), augment=args.mode == "finetune", seed=seed)
    if args.mode == "pretrain":
        val_ds = BraTSTrainDataset(dc["cache_root"], val_cases, crop_size=dc["crop_size"], augment=False, seed=seed)
        val_bs = tr["batch_size"]
    else:
        val_ds = BraTSEvalDataset(dc["cache_root"], val_cases)
        val_bs = 1
    nw = tr.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=min(nw, 2))

    module = TumorResearchModule(cfg, mode=args.mode)
    if args.pretrained:
        module.load_pretrained_encoder(args.pretrained)

    if args.mode == "pretrain":
        ckpt = ModelCheckpoint(dirpath=out_dir, filename="best", monitor="val/mim_loss", mode="min", save_last=True)
    else:
        ckpt = ModelCheckpoint(dirpath=out_dir, filename="best", monitor="val/dice_mean", mode="max", save_last=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = torch.cuda.device_count() if accelerator == "gpu" else 1
    trainer = pl.Trainer(
        default_root_dir=out_dir,
        max_epochs=tr["epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy="ddp" if devices > 1 else "auto",
        precision=tr.get("precision", "16-mixed") if accelerator == "gpu" else 32,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(out_dir, name="logs"),
        check_val_every_n_epoch=tr.get("val_every", 5),
        gradient_clip_val=tr.get("grad_clip", 1.0),
        deterministic=False,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"[done] best checkpoint: {ckpt.best_model_path} (score {ckpt.best_model_score})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--mode", choices=["pretrain", "finetune"], default="finetune")
    p.add_argument("--pretrained", default=None, help="checkpoint from --mode pretrain")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--runs_dir", default="runs")
    main(p.parse_args())
