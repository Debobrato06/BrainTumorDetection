import torch
from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image
from torchvision import transforms

def image_to_volume(img, img_size=(64, 64, 64)):
    """The detection model's input for one 2D image: grayscale, resized to (H, W), scaled to
    [0, 1] (no z-scoring), copied over depth D and the 4 channels -> (4, D, H, W).
    Used by training, the web app and evaluation, so all three see the same representation."""
    d, h, w = img_size
    t = transforms.ToTensor()(transforms.Resize((h, w))(img.convert('L')))  # (1, H, W)
    return t.unsqueeze(1).repeat(4, d, 1, 1)


class RandomGeneratorDataset(Dataset):
    """
    Random tensors with random labels: a smoke test that the code runs, never a result.
    For real 3D data use brats_dataset.py with training_pipeline.py.
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
    def __init__(self, root_dir, img_size=(64, 64), transform=None, files=None, depth=64):
        self.root_dir = root_dir
        self.img_size = img_size
        self.depth = depth
        self.transform = transform
        self.files = []
        if files is not None:  # explicit (path, label) list, e.g. one split
            self.files = list(files)
            return
        
        # Scan for images
        for label, class_name in enumerate(['no_tumor', 'tumor']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            img_files = glob(os.path.join(class_dir, "*.*"))
            img_files = [f for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for f in sorted(img_files):
                self.files.append((f, label))
                
        print(f"Found {len(self.files)} 2D images in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        
        # Pseudo-3D input: the 2D slice repeated over depth and the 4 channels
        with Image.open(img_path) as img:
            vol = image_to_volume(img, (self.depth, *self.img_size))
        return vol, torch.tensor(label, dtype=torch.long)
