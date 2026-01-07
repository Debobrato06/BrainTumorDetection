import torch
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
from glob import glob
from PIL import Image
from torchvision import transforms

class MRIDataset(Dataset):
    """
    Dataset class for Multi-Modal MRI Brain Tumor Detection.
    Assumes data structure:
        root_dir/
            train/ (or val/)
                Subject_001/
                    Subject_001_t1.nii.gz
                    Subject_001_t1ce.nii.gz
                    Subject_001_t2.nii.gz
                    Subject_001_flair.nii.gz
                Subject_002/
                    ...
    """
    def __init__(self, root_dir, split='train', img_size=(64, 64, 64), transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # Find all subject folders
        self.subjects = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        # In a real scenario, you'd load labels from a csv file maping SubjectID -> Label (0/1)
        # For this template, we will assign random labels or look for a label.csv
        # self.labels = load_labels(...)
        print(f"Found {len(self.subjects)} subjects in {self.split} set.")

    def __len__(self):
        return len(self.subjects)

    def normalize(self, volume):
        """Standard Score Normalization"""
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            return (volume - mean) / std
        return volume - mean

    def resize_volume(self, volume):
        """
        Simple resizing by cropping or padding to img_size.
        For a specialized resize, scipy.ndimage.zoom could be used (slower).
        """
        D, H, W = volume.shape
        tD, tH, tW = self.img_size
        
        # Initialize canvas
        new_vol = np.zeros(self.img_size, dtype=np.float32)
        
        # Calculate start/end indices to center crop/pad
        min_d, max_d = max(0, (D - tD) // 2), min(D, (D + tD) // 2)
        
        out_min_d = max(0, (tD - D) // 2)
        
        # Slicing logic can be complex, using simple crop for now
        # Assuming input is larger than target; if smaller, this needs padding logic.
        # Simplification: Random crop or Center crop
        
        # Let's do Center Crop for simplicity in this template
        c_d, c_h, c_w = D//2, H//2, W//2
        s_d, s_h, s_w = tD//2, tH//2, tW//2
        
        # Handle boundaries
        start_d = max(0, c_d - s_d)
        end_d = min(D, start_d + tD)
        
        # Assign to new volume (adjusting output slice if input was smaller)
        # This is strictly a 'crop', padding implementation skipped for brevity.
        
        crop = volume[start_d:end_d, 
                      max(0, c_h - s_h):min(H, c_h - s_h + tH), 
                      max(0, c_w - s_w):min(W, c_w - s_w + tW)]
        
        # Place into new_vol (handling shape mismatches if crop < target)
        new_vol[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        
        return new_vol

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        subj_dir = os.path.join(self.root_dir, subject)
        
        # Example filenames: suffix could be _t1.nii.gz, etc.
        # We need to load 4 modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        channels = []
        
        for mod in modalities:
            # Flexible search for files
            search_path = os.path.join(subj_dir, f"*{mod}*.nii.gz")
            matches = glob(search_path)
            if not matches:
                # Fallback or error
                # print(f"Warning: {mod} not found for {subject}")
                vol = np.zeros(self.img_size, dtype=np.float32) # Missing modality
            else:
                filepath = matches[0]
                nii_img = nib.load(filepath)
                vol = nii_img.get_fdata().astype(np.float32)
                vol = self.normalize(vol)
                vol = self.resize_volume(vol)
                
            channels.append(vol)
            
        # Stack channels: (4, D, H, W)
        data = np.stack(channels, axis=0)
        data_tensor = torch.from_numpy(data)
        
        # Mock Label for template (Replace with real label lookup)
        label = 0 
        if "Tumor" in subject: label = 1
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return data_tensor, label_tensor

class RandomGeneratorDataset(Dataset):
    """
    For testing without real data.
    """
    def __init__(self, length=10, img_size=(64, 64, 64)):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random 3D data (4 channels)
        data = torch.randn(4, *self.img_size)
        label = torch.randint(0, 2, (1,)).item()
        return data, label

class Image2DDataset(Dataset):
    """
    Dataset for 2D images (JPG/PNG).
    Expected structure:
    root_dir/
       tumor/
          img1.jpg
          ...
       no_tumor/
          img2.jpg
          ...
    """
    def __init__(self, root_dir, img_size=(64, 64), transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.files = []
        
        # Scan for images
        for label, class_name in enumerate(['no_tumor', 'tumor']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            img_files = glob(os.path.join(class_dir, "*.*"))
            img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for f in img_files:
                self.files.append((f, label))
                
        print(f"Found {len(self.files)} 2D images in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        
        # Load Image
        img = Image.open(img_path).convert('L') # Grayscale
        
        # Resize
        resize = transforms.Resize(self.img_size)
        img_tensor = transforms.ToTensor()(resize(img)) # (1, H, W)
        
        # Convert to pseudo-3D for the model architecture
        # (1, H, W) -> (1, D=16, H, W) -> Repeat or stack?
        # Our model expects (4, D, H, W).
        # Strategy: Repeat the single 2D slice across depth and channels
        
        # (1, 1, H, W) -> repeat depth 64
        vol = img_tensor.unsqueeze(1).repeat(4, 64, 1, 1) 
        
        return vol, torch.tensor(label, dtype=torch.long)
