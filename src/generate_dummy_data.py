import numpy as np
from PIL import Image
import os
import pathlib

root = pathlib.Path(r"D:\DebobratoResearch\BrainTumorDetection\dummy_data\Training")
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

for c in classes:
    p = root / c
    p.mkdir(parents=True, exist_ok=True)
    
    # Create 5 dummy images per class
    for i in range(5):
        img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(p / f"mock_{c}_{i}.png")
        
print("Created dummy images for verification.")
