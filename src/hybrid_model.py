"""
BrainHybridNet: lightweight hybrid CNN-Transformer for joint brain tumor
detection, BraTS sub-region segmentation (WT/TC/ET) and LGG/HGG grading.

Data flow (input X: B x C x D x H x W, D=H=W divisible by 2 * patch_size):

    X ──► SE channel attention ──► CNN stem ──► c1 (32, D)   ──────────────┐ skip
                                         └──► c2 (64, D/2) ──────────┐     │ skip
                                                 │ patch embed (p)   │     │
                                                 ▼                   │     │
                                      ViT encoder (L blocks)         │     │
                                                 │ z: B x N x E      │     │
                       mean(z) ──► cls / grade   ▼                   │     │
                                      reshape to (E, D/2p) ──► log2(p) up-blocks
                                                 ──► cat c2 ──► up ──► cat c1 ──► seg head

Self-supervised pre-training is SimMIM-style masked image modeling: masked
patches are zeroed in the input, their tokens are replaced by a learned mask
token, and a light linear head reconstructs the masked input patches. The loss
is computed on masked patches only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResidualSEBlock3D, SqueezeAndExcitation3D


class ViTBlock(nn.Module):
    """Pre-norm Transformer block: MHSA + MLP, both with residuals."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attn_weights
        return x


class UpBlock(nn.Module):
    """x2 transposed-conv upsampling, optional skip concat, residual SE refinement."""

    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.refine = ResidualSEBlock3D(out_ch + skip_ch, out_ch, reduction=8)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class HybridTumorModel(nn.Module):
    """
    Args:
        in_channels: MRI sequences (4 = T1, T1ce, T2, FLAIR).
        img_size: training crop size (D, H, W); needed for positional embeddings.
        patch_size: ViT patch size on the stride-2 CNN feature map.
        embed_dim, depth, num_heads: ViT width / layers / heads.
        dropout: dropout in ViT blocks and heads (also used by MC Dropout).
        decoder_dropout: Dropout3d on the bottleneck feature map (MC Dropout).
        num_seg_classes: 3 for BraTS WT/TC/ET (independent sigmoid channels).
        use_se: input sequence attention (ablation switch).
        use_skips: CNN skip connections into the decoder; False gives the
            "transformer path only" ablation. depth=0 gives the "CNN only" ablation.
    """

    def __init__(self, in_channels=4, num_classes=2, img_size=(128, 128, 128), patch_size=8,
                 embed_dim=128, depth=4, num_heads=4, dropout=0.1, decoder_dropout=0.1,
                 num_seg_classes=3, num_grades=2, use_se=True, use_skips=True):
        super().__init__()
        if any(s % (2 * patch_size) for s in img_size):
            raise ValueError(f"img_size {img_size} must be divisible by 2 * patch_size = {2 * patch_size}")
        if patch_size & (patch_size - 1):
            raise ValueError("patch_size must be a power of two")

        self.in_channels = in_channels
        self.img_size = tuple(img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid = tuple(s // (2 * patch_size) for s in img_size)
        self.num_patches = math.prod(self.grid)
        self.use_skips = use_skips

        self.input_se = SqueezeAndExcitation3D(in_channels, reduction=2) if use_se else None

        # CNN stem: c1 at full resolution, c2 at 1/2 resolution
        self.stem1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.stem2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # ViT encoder on c2
        self.patch_embed = nn.Conv3d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: log2(patch_size) up-blocks bring tokens back to c2 resolution,
        # the last of them fuses c2; one more fuses c1 at full resolution.
        n_up = int(math.log2(patch_size))
        self.bottleneck_dropout = nn.Dropout3d(decoder_dropout)
        chans = [embed_dim] + [max(64, embed_dim // (2 ** (i + 1))) for i in range(n_up)]
        chans[-1] = 64
        self.up_blocks = nn.ModuleList()
        for i in range(n_up):
            skip_ch = 64 if (i == n_up - 1 and use_skips) else 0
            self.up_blocks.append(UpBlock(chans[i], chans[i + 1], skip_ch=skip_ch))
        self.up_final = UpBlock(64, 32, skip_ch=32 if use_skips else 0)
        self.seg_head = nn.Conv3d(32, num_seg_classes, kernel_size=1)

        # Global heads on mean-pooled tokens
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.grade_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, num_grades)
        )

        # Masked-image-modeling head: token -> voxels of its input-space patch
        self.input_patch = 2 * patch_size
        self.mim_head = nn.Linear(embed_dim, in_channels * self.input_patch ** 3)

    # ------------------------------------------------------------------ encoder
    def _encode(self, x, token_mask=None, return_attention=False):
        seq_weights = None
        if self.input_se is not None:
            x, seq_weights = self.input_se(x)
        c1 = self.stem1(x)
        c2 = self.stem2(c1)

        tokens = self.patch_embed(c2).flatten(2).transpose(1, 2)  # (B, N, E)
        if tokens.shape[1] != self.num_patches:
            raise ValueError(
                f"Input spatial size {tuple(x.shape[2:])} does not match img_size {self.img_size}; "
                "use sliding-window inference for other sizes."
            )
        if token_mask is not None:
            m = token_mask.unsqueeze(-1).type_as(tokens)
            tokens = tokens * (1 - m) + self.mask_token * m
        z = tokens + self.pos_embed

        attns = []
        for blk in self.blocks:
            if return_attention:
                z, a = blk(z, return_attention=True)
                attns.append(a)
            else:
                z = blk(z)
        z = self.norm(z)
        return z, c1, c2, seq_weights, attns

    def _decode(self, z, c1, c2):
        B = z.shape[0]
        vol = z.transpose(1, 2).reshape(B, self.embed_dim, *self.grid)
        vol = self.bottleneck_dropout(vol)
        last = len(self.up_blocks) - 1
        for i, up in enumerate(self.up_blocks):
            vol = up(vol, c2 if (i == last and self.use_skips) else None)
        vol = self.up_final(vol, c1 if self.use_skips else None)
        return self.seg_head(vol)

    # ------------------------------------------------------------------ forward
    def forward(self, x, return_attention=False):
        """Returns (cls_logits, seg_logits, grade_logits); with return_attention also a dict
        {'attention': [per-block (B, N, N)], 'sequence_weights': (B, C)}."""
        z, c1, c2, seq_weights, attns = self._encode(x, return_attention=return_attention)
        pooled = z.mean(dim=1)
        cls_logits = self.cls_head(pooled)
        grade_logits = self.grade_head(pooled)
        seg_logits = self._decode(z, c1, c2)
        if return_attention:
            return cls_logits, seg_logits, grade_logits, {"attention": attns, "sequence_weights": seq_weights}
        return cls_logits, seg_logits, grade_logits

    # ------------------------------------------------------- self-supervision
    def random_token_mask(self, batch_size, mask_ratio, device):
        num_mask = int(round(self.num_patches * mask_ratio))
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids = noise.argsort(dim=1)[:, :num_mask]
        mask = torch.zeros(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids, 1.0)
        return mask  # 1 = masked

    def _token_mask_to_voxels(self, mask):
        B = mask.shape[0]
        m = mask.reshape(B, 1, *self.grid)
        return F.interpolate(m, scale_factor=self.input_patch, mode="nearest")

    def _patchify(self, x):
        """(B, C, D, H, W) -> (B, N, C * P^3) in the same token order as patch_embed."""
        B, C = x.shape[:2]
        P = self.input_patch
        gd, gh, gw = self.grid
        x = x.reshape(B, C, gd, P, gh, P, gw, P)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return x.reshape(B, gd * gh * gw, C * P ** 3)

    def forward_mim(self, x, mask_ratio=0.75):
        """Masked image modeling loss (MSE on masked patches only, per-patch normalized target)."""
        mask = self.random_token_mask(x.shape[0], mask_ratio, x.device)
        x_masked = x * (1 - self._token_mask_to_voxels(mask))
        z, *_ = self._encode(x_masked, token_mask=mask)
        pred = self.mim_head(z)  # (B, N, C * P^3)

        target = self._patchify(x)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((pred - target) ** 2).mean(dim=-1)  # per patch
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def enable_mc_dropout(model):
    """Eval mode everywhere (BatchNorm uses running stats) except dropout layers."""
    model.eval()
    for m in model.modules():
        # nn.MultiheadAttention applies its internal attention dropout only in training mode
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.MultiheadAttention)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, x, num_samples=20):
    """Returns (mean sigmoid seg prob, voxel-wise variance, mean softmax cls prob)."""
    was_training = model.training
    enable_mc_dropout(model)
    seg_samples, cls_samples = [], []
    for _ in range(num_samples):
        cls_logits, seg_logits, _ = model(x)[:3]
        seg_samples.append(torch.sigmoid(seg_logits))
        cls_samples.append(torch.softmax(cls_logits, dim=1))
    model.train(was_training)
    seg = torch.stack(seg_samples)
    return seg.mean(0), seg.var(0), torch.stack(cls_samples).mean(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
