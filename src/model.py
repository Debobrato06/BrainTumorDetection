import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcitation3D(nn.Module):
    """
    3D Squeeze-and-Excitation (SE) Channel Attention Block.
    Performs channel-wise feature recalibration across multi-modal MRI sequences 
    (T1, T1c, T2, FLAIR) and internal feature maps.
    
    Learns dynamic sequence weights:
    - High T1c weight -> Enhancing Tumor (ET) / Blood-Brain Barrier disruption
    - High FLAIR weight -> Whole Tumor (WT) / Vasogenic Edema
    - High T2/T1 weight -> Tumor Core (TC) / Necrosis & Anatomy
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_dim = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_dim, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, _, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling -> (B, C)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        weights = self.sigmoid(y)        # Channel attention weights -> (B, C)
        out = x * weights.view(b, c, 1, 1, 1)
        return out, weights


class ResidualSEBlock3D(nn.Module):
    """
    3D Residual Block integrated with Squeeze-and-Excitation (SE) Attention.
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SqueezeAndExcitation3D(out_channels, reduction=reduction)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out, _ = self.se(out)
        out += residual
        return self.relu(out)

# Backward compatibility alias
ResidualEncoderBlock3D = ResidualSEBlock3D


class BrainTumorClassifier(nn.Module):
    """
    State-of-the-Art SE-Attention 3D U-Net Architecture for Multi-modal Brain Tumor 
    Segmentation (BraTS ET/TC/WT), WHO CNS5 Grade & Type Classification, and Risk Scoring.
    """
    def __init__(self, input_channels=4, num_classes=2, volume_size=(64, 64, 64), in_channels=None):
        super().__init__()
        if in_channels is not None:
            input_channels = in_channels
        
        # 1. Multi-modal Sequence SE Channel Attention Recalibration
        # Directly recalibrates input MRI channels: [T1, T1c, T2, FLAIR]
        self.input_se = SqueezeAndExcitation3D(input_channels, reduction=2)

        # 2. Encoder Path with SE Attention
        self.enc1 = ResidualSEBlock3D(input_channels, 32, reduction=4)
        self.enc2 = ResidualSEBlock3D(32, 64, reduction=8)
        self.enc3 = ResidualSEBlock3D(64, 128, reduction=16)
        self.enc4 = ResidualSEBlock3D(128, 256, reduction=16)
        
        self.pool = nn.MaxPool3d(2)
        
        # 3. Bottleneck
        self.bottleneck = ResidualSEBlock3D(256, 512, reduction=16)
        
        # 4. Decoder Path (Segmentation: ET, TC, WT sub-regions)
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ResidualSEBlock3D(512, 256, reduction=16)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualSEBlock3D(256, 128, reduction=16)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualSEBlock3D(128, 64, reduction=8)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualSEBlock3D(64, 32, reduction=4)
        
        # 5. Multi-task Output Heads
        # a) BraTS Sub-region Segmentation Head: 3 channels (0: ET, 1: TC, 2: WT)
        self.seg_head = nn.Conv3d(32, 3, kernel_size=1)

        # b) Global Pooling for Classification & Grading
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # c) SOL Binary Detection Classifier Head
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # d) WHO CNS5 Grade Classification Head (LGG vs HGG / Grade 1-4)
        self.grade_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # [0: LGG, 1: HGG]
        )

        # e) Tumor Type Classification Head (Glioma, Meningioma, Metastasis, Medulloblastoma, Healthy)
        self.type_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x, return_attention=False):
        # x: (B, 4, 64, 64, 64)
        
        # Recalibrate input MRI sequence channels (T1, T1c, T2, FLAIR)
        x_se, input_channel_weights = self.input_se(x)

        # Encoder Path
        e1 = self.enc1(x_se)       # (B, 32, 64, 64, 64)
        p1 = self.pool(e1)         # (B, 32, 32, 32, 32)
        
        e2 = self.enc2(p1)         # (B, 64, 32, 32, 32)
        p2 = self.pool(e2)         # (B, 64, 16, 16, 16)
        
        e3 = self.enc3(p2)         # (B, 128, 16, 16, 16)
        p3 = self.pool(e3)         # (B, 128, 8, 8, 8)
        
        e4 = self.enc4(p3)         # (B, 256, 8, 8, 8)
        p4 = self.pool(e4)         # (B, 256, 4, 4, 4)
        
        b = self.bottleneck(p4)    # (B, 512, 4, 4, 4)

        # Global Bottleneck Feature Vector
        feat = self.avg_pool(b).view(b.size(0), -1)

        # Classification Heads
        cls_logits = self.cls_head(feat)
        grade_logits = self.grade_head(feat)
        type_logits = self.type_head(feat)

        # Decoder Path for Sub-region Segmentation
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        seg_logits = self.seg_head(d1)  # (B, 3, 64, 64, 64) for ET, TC, WT

        if return_attention:
            sequence_weights = {
                'T1': float(input_channel_weights[0, 0].item()),
                'T1c': float(input_channel_weights[0, 1].item()) if input_channel_weights.shape[1] > 1 else 0.5,
                'T2': float(input_channel_weights[0, 2].item()) if input_channel_weights.shape[1] > 2 else 0.5,
                'FLAIR': float(input_channel_weights[0, 3].item()) if input_channel_weights.shape[1] > 3 else 0.5
            }
            return cls_logits, seg_logits, grade_logits, type_logits, sequence_weights

        return cls_logits, seg_logits, grade_logits

def test_model():
    model = BrainTumorClassifier(input_channels=4, volume_size=(64, 64, 64))
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    cls_logits, seg_logits, grade_logits, type_logits, seq_w = model(dummy_input, return_attention=True)
    print(f"Cls Logits Shape: {cls_logits.shape}")
    print(f"Seg Logits Shape (ET/TC/WT): {seg_logits.shape}")
    print(f"Grade Logits Shape: {grade_logits.shape}")
    print(f"Type Logits Shape: {type_logits.shape}")
    print(f"Sequence Attention Weights: {seq_w}")

if __name__ == "__main__":
    test_model()

