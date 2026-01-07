import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """
    Standard ResNet-style 3D block
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BrainTumorClassifier(nn.Module):
    """
    Improved 3D CNN (ResNet-10 style) for volumetric MRI classification.
    More robust than simple Transformers for small datasets.
    """
    def __init__(self, input_channels=4, num_classes=2, volume_size=(64, 64, 64)):
        super().__init__()
        
        self.in_channels = 32
        
        # Initial Conv
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) # 64 -> 32
        
        # ResBlocks
        self.layer1 = ResidualBlock3D(32, 64, stride=2)   # 32 -> 16
        self.layer2 = ResidualBlock3D(64, 128, stride=2)  # 16 -> 8
        self.layer3 = ResidualBlock3D(128, 256, stride=2) # 8 -> 4
        
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, 4, 64, 64, 64)
        out = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def test_model():
    model = BrainTumorClassifier(input_channels=4, volume_size=(64, 64, 64))
    dummy_input = torch.randn(1, 4, 64, 64, 64)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")

if __name__ == "__main__":
    test_model()
