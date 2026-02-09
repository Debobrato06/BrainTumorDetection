import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTBlock(nn.Module):
    """
    Standard Vision Transformer Block with Self-Attention and MLP.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attention=False):
        # x: (B, N, C)
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x

class HybridEncoder(nn.Module):
    def __init__(self, img_size=(64,64,64), in_channels=1, embed_dim=768, num_heads=12, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        
        # 1. CNN Tokenizer (Patch Embedding + Local Features)
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(4) # 4 layers for demo
        ])
        
    def forward(self, x, mask_ratio=0.0, return_attention=False):
        # x: (B, C, D, H, W)
        B = x.shape[0]
        x = self.patch_embed(x) # (B, Embed, D', H', W')
        x = x.flatten(2).transpose(1, 2) # (B, N, Embed)
        
        x = x + self.pos_embed
        
        # SSL: Random Masking for MAE
        if mask_ratio > 0:
            N = x.shape[1]
            num_keep = int(N * (1 - mask_ratio))
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :num_keep]
            x = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim))
            
        attns = []
        for blk in self.transformer_blocks:
            if return_attention:
                x, attn = blk(x, return_attention=True)
                attns.append(attn)
            else:
                x = blk(x)
            
        if return_attention:
            return x, attns
        return x

class HybridTumorModel(nn.Module):
    """
    Research-Grade Hybrid CNN-Transformer Model.
    Supports:
    - Joint Classification & Segmentation
    - SSL (MAE style masking)
    - Uncertainty (Dropout)
    """
    def __init__(self, in_channels=1, num_classes=2, embed_dim=128):
        super().__init__()
        
        # --- Encoder ---
        # Initial CNN features (High Res)
        self.cnn_stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), # Down 2
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        
        # Transformer Bridge (Global Context)
        self.transformer_encoder = HybridEncoder(
            img_size=(32,32,32), # Assuming input 64->32
            in_channels=64, 
            embed_dim=embed_dim,
            num_heads=4
        )
        
        # --- Decoder (High-Resolution U-Net Path) ---
        self.up_conv1 = nn.ConvTranspose3d(embed_dim, 64, kernel_size=2, stride=2)
        self.dec_conv1 = ResidualEncoderBlock3D(128, 64) # Concat skip
        
        self.up_conv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec_conv2 = ResidualEncoderBlock3D(64, 32)
        
        # --- Multi-Task Heads (Journal Grade) ---
        # 1. Segmentation Head (Fine Delineation)
        self.seg_head = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1) 
        )
        
        # 2. Classification Head (Detection)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # 3. WHO GRADING HEAD (Professional Innovation)
        # Estimates High-Grade Glioma (HGG) vs Low-Grade Glioma (LGG)
        self.grade_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [LGG, HGG]
        )
        
    def forward(self, x, mask_ratio=0.0, return_attention=False):
        # x: (B, C, 64, 64, 64)
        B, C, D, H, W = x.shape
        
        # 1. CNN Feature Extraction (Encoder)
        c1 = self.cnn_stem[0:3](x) # (B, 32, 64, 64, 64)
        c2 = self.cnn_stem[3:](c1) # (B, 64, 32, 32, 32)
        
        # 2. Transformer Feature Learning (Global Context)
        if return_attention:
            z, attns = self.transformer_encoder(c2, mask_ratio=mask_ratio, return_attention=True)
        else:
            z = self.transformer_encoder(c2, mask_ratio=mask_ratio) # (B, N, Embed)
        
        # 3. Bottleneck Analysis Logic
        z_mean = z.mean(dim=1) # Global latent vector
        cls_logits = self.cls_head(z_mean)
        grade_logits = self.grade_head(z_mean)
        
        # 4. Decoding (Spatial Reconstruction)
        # Reshape z (B, 512, 128) -> (B, 128, 8, 8, 8) assuming patch_size and pooling context
        # In this research prototype, we project and upsample for skip connections
        z_vol = z.transpose(1, 2).view(B, -1, 8, 8, 8) 
        
        # Upsampling path
        u1 = self.up_conv1(z_vol) # -> (B, 64, 16, 16, 16)
        # Note: c2 is (B, 64, 32, 32, 32). We upsample u1 again or downsample c2?
        # Standard U-Net: u1 matches c2 size. 
        # Here we project to match c2
        u1_res = F.interpolate(u1, size=c2.shape[2:], mode='trilinear')
        d1 = self.dec_conv1(torch.cat([u1_res, c2], dim=1))
        
        u2 = self.up_conv2(d1) # -> (B, 32, 32, 32, 32)
        u2_res = F.interpolate(u2, size=c1.shape[2:], mode='trilinear')
        d2 = self.dec_conv2(torch.cat([u2_res, c1], dim=1))
        
        seg_logits = self.seg_head(d2)
        
        if return_attention:
            return cls_logits, seg_logits, grade_logits, attns
        return cls_logits, seg_logits, grade_logits

    def get_uncertainty_map(self, x, num_samples=5):
        """
        Monte Carlo Dropout for Uncertainty Estimation.
        """
        self.train() # Enable dropout
        seg_preds = []
        for _ in range(num_samples):
            _, seg = self.forward(x)
            seg_preds.append(torch.sigmoid(seg))
            
        seg_preds = torch.stack(seg_preds)
        mean_seg = seg_preds.mean(dim=0)
        uncertainty = seg_preds.var(dim=0) # Variance as uncertainty
        return mean_seg, uncertainty

    def inflate_2d_weights(self, state_dict_2d):
        """
        Transfer Learning: Inflate 2D ImageNet weights to 3D.
        """
        own_state = self.state_dict()
        for name, param in state_dict_2d.items():
            if name in own_state:
                # If shapes differ only in depth dim
                if own_state[name].shape != param.shape:
                    # Generic inflation: repeat along depth
                    # Example: (64, 3, 3, 3) vs (64, 3, 3)
                    # We unsqueeze and repeat
                    pass # Implementation left as exercise / simpler placeholder
