import math

import numpy as np
import pytorch_lightning as pl
import torch

from losses import MultiTaskLoss
from metrics import dice
from model_factory import build_model


def sliding_window_seg(model, image, roi_size, overlap=0.5, sw_batch_size=2):
    """Sliding-window segmentation logits for a whole volume (B, C, X, Y, Z)."""
    from monai.inferers import sliding_window_inference

    def seg_only(x):
        return model(x)[1]

    return sliding_window_inference(image, roi_size, sw_batch_size, seg_only, overlap=overlap, mode="gaussian")


class TumorResearchModule(pl.LightningModule):
    """
    mode='pretrain': masked image modeling on the hybrid model (no labels used).
    mode='finetune': joint segmentation + detection + grading (baselines: segmentation only).
    """

    def __init__(self, cfg, mode="finetune"):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg, "mode": mode})
        self.cfg = cfg
        self.mode = mode
        self.roi = tuple(cfg["data"]["crop_size"])
        self.model = build_model(cfg["model"], self.roi)
        lw = cfg.get("loss", {})
        self.loss_fn = MultiTaskLoss(
            w_seg=lw.get("w_seg", 0.6), w_cls=lw.get("w_cls", 0.2), w_grade=lw.get("w_grade", 0.2),
            tversky_alpha=lw.get("tversky_alpha", 0.3), tversky_beta=lw.get("tversky_beta", 0.7),
        )
        if mode == "pretrain" and not hasattr(self.model, "forward_mim"):
            raise ValueError("Pre-training is only defined for the hybrid model")
        self._val_dice = []

    def load_pretrained_encoder(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.") and "mim_head" not in k}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[pretrained] loaded {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        if self.mode == "pretrain":
            loss = self.model.forward_mim(x, mask_ratio=self.cfg["train"].get("mask_ratio", 0.75))
            self.log("train/mim_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
            return loss

        outputs = self.model(x)
        loss, terms = self.loss_fn(
            outputs,
            seg=batch["seg"],
            label=batch["label"] if outputs[0] is not None else None,
            grade=batch["grade"] if outputs[2] is not None else None,
        )
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        for k, v in terms.items():
            self.log(f"train/{k}_loss", v, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "pretrain":
            loss = self.model.forward_mim(batch["image"], mask_ratio=self.cfg["train"].get("mask_ratio", 0.75))
            self.log("val/mim_loss", loss, prog_bar=True)
            return
        ev = self.cfg.get("eval", {})
        logits = sliding_window_seg(self.model, batch["image"], self.roi, ev.get("sw_overlap", 0.5), ev.get("sw_batch_size", 2))
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]
        ref = batch["seg"].cpu().numpy()[0] > 0.5
        self._val_dice.append([dice(pred[i], ref[i]) for i in range(3)])

    def on_validation_epoch_end(self):
        if self.mode == "pretrain" or not self._val_dice:
            return
        d = np.array(self._val_dice).mean(0)
        self._val_dice = []
        self.log_dict({"val/dice_wt": d[0], "val/dice_tc": d[1], "val/dice_et": d[2], "val/dice_mean": d.mean()},
                      prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        tr = self.cfg["train"]
        opt = torch.optim.AdamW(self.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 1e-4))
        epochs, warmup = tr["epochs"], tr.get("warmup_epochs", 5)

        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "epoch"}]
