import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from hybrid_model import HybridTumorModel
from losses import JointLoss

class TumorResearchModule(pl.LightningModule):
    def __init__(self, 
                 lr=1e-4, 
                 weight_decay=1e-4, 
                 max_epochs=100, 
                 mode='finetune', # 'pretrain' or 'finetune'
                 mask_ratio=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = HybridTumorModel(in_channels=4)
        
        # Losses
        self.joint_loss = JointLoss()
        self.recon_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, x, mask_ratio=0.0):
        return self.model(x, mask_ratio=mask_ratio)
    
    def training_step(self, batch, batch_idx):
        # Unwrap data (robust to different dataset formats)
        if len(batch) == 2:
            x, y = batch # Supervised
        else:
            x = batch # Unsupervised/SSL
            y = None
            
        if self.hparams.mode == 'pretrain':
            # Self-Supervised Learning (MAE-style)
            # Mask input, try to reconstruct original X from visible patches
            # Using the segmentation head as a pixel-wise reconstructor for simplicity
            cls_logits, recon = self.model(x, mask_ratio=self.hparams.mask_ratio)
            
            # Loss is MSE between reconstructed and original x
            loss = self.recon_loss(recon, x)
            self.log('train/ssl_loss', loss, prog_bar=True)
            return loss
            
        else:
            # Joint Classification + Segmentation
            cls_logits, seg_logits = self.model(x)
            
            # Assuming y is {label, mask} dict or tuple?
            # Adjust based on dataset. Let's assume y is tuple (class_label, seg_mask)
            # If dataset yields (x, y_class), we ignore seg?
            # Let's assume we have full supervision for this stage as requested.
            
            # Mock unpacking for robust coding
            if isinstance(y, (list, tuple)):
                target_cls, target_seg = y
            else:
                target_cls = y
                target_seg = torch.zeros_like(x) # Dummy mask if missing
                
            loss_cls = self.cls_loss(cls_logits, target_cls)
            loss_seg = self.joint_loss(seg_logits, target_seg)
            
            loss = loss_cls + loss_seg
            
            self.log('train/loss', loss)
            self.log('train/cls_loss', loss_cls)
            self.log('train/seg_loss', loss_seg)
            
            return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]
