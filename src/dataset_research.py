import torch
from torch.utils.data import Dataset
import numpy as np

class ResearchMockDataset(Dataset):
    """
    Mock Dataset for Joint Segmentation and Classification.
    Returns: 
        x: (4, 64, 64, 64) dummy MRI
        target: tuple (class_label, segmentation_mask)
    """
    def __init__(self, length=20, img_size=(64, 64, 64)):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # random 3D data (4 channels)
        data = torch.randn(4, *self.img_size)
        
        # Binary Classification Label (0 or 1)
        label = torch.randint(0, 2, (1,)).item()
        
        # Segmentation Mask (Same spatial dim as input)
        # If label is 0 (No Tumor), mask is all zeros
        # If label is 1 (Tumor), mask has a random blob
        mask = torch.zeros((1, *self.img_size))
        
        if label == 1:
            # Create a simplified random blob
            center = np.random.randint(16, 48, size=3)
            radius = np.random.randint(5, 12)
            
            # Meshgrid for mask generation (simplified)
            # Just filling a cube for speed
            d, h, w = center
            r = radius
            mask[:, max(0, d-r):min(64, d+r), 
                    max(0, h-r):min(64, h+r), 
                    max(0, w-r):min(64, w+r)] = 1.0
            
        return data, (torch.tensor(label, dtype=torch.long), mask)
