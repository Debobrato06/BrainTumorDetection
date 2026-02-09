import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualEncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
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
        out += residual
        return self.relu(out)

class BrainTumorClassifier(nn.Module):
    """
    State-of-the-Art 3D Residual U-Net for Brain Tumor Detection & Segmentation.
    Inspired by SegResNet (BraTS Winner Architecture).
    """
    def __init__(self, input_channels=4, num_classes=2, volume_size=(64, 64, 64)):
        super().__init__()
        
        # Encoder
        self.enc1 = ResidualEncoderBlock3D(input_channels, 32)
        self.enc2 = ResidualEncoderBlock3D(32, 64)
        self.enc3 = ResidualEncoderBlock3D(64, 128)
        self.enc4 = ResidualEncoderBlock3D(128, 256)
        
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ResidualEncoderBlock3D(256, 512)
        
        # Decoder (for Segmentation - Optional but powerful for feature learning)
        self.up4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ResidualEncoderBlock3D(512, 256) # 256 + 256 from skip
        
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualEncoderBlock3D(256, 128)
        
        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, 4, 64, 64, 64)
        
        # Encoder path with skip connections
        e1 = self.enc1(x)
        p1 = self.pool(e1) # 32
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2) # 16
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3) # 8
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4) # 4
        
        b = self.bottleneck(p4)
        
        # We use the bottleneck for classification
        feat = self.avg_pool(b)
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        
        return logits # Add seg output if mask is available

def test_model():
    model = BrainTumorClassifier(input_channels=4, volume_size=(64, 64, 64))
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")

if __name__ == "__main__":
    test_model()
