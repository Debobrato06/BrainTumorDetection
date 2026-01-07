import torch
import nibabel as nib
import numpy as np
import argparse
from model import BrainTumorClassifier

def load_volume(path, img_size=(64, 64, 64)):
    # Simple loader for inference - expects 4D file or folder with 4 files? 
    # For simplicity, assume user provides path to a 4D stacked Nifti or we just use one modality to demo.
    # In reality, need t1, t1ce, t2, flair. 
    # Let's assume input is a 4D Nifti file (C, D, H, W) or (D, H, W, C)
    
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    
    # Handle dimensions (Standardize to C, D, H, W)
    if data.ndim == 3:
        # Single modality? Or 3D volume
        # We need 4 channels. 
        # This is tricky without exact files. 
        # For this template, we'll placeholder a single modality repeated 4 times
        data = np.stack([data]*4, axis=0) 
    elif data.ndim == 4:
        # If (D, H, W, C) -> Transpose to (C, D, H, W)
        if data.shape[3] == 4:
             data = data.transpose(3, 0, 1, 2)
    
    # Resize Logic (Simplified)
    # ... (Same as dataset.py)
    # Just returning raw resized for now
    
    # Normalize
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / (std + 1e-8)
    
    # Add Batch Dim
    tensor = torch.from_numpy(data).unsqueeze(0) # (1, 4, D, H, W)
    
    # Interpolate to target size if needed (simplest way for inference)
    tensor = torch.nn.functional.interpolate(tensor, size=img_size, mode='trilinear')
    
    return tensor

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BrainTumorClassifier(input_channels=4, num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    input_tensor = load_volume(args.input_path).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    print(f"Prediction: {'Tumor Detected' if predicted_class == 1 else 'No Tumor'}")
    print(f"Confidence: {probabilities[0][predicted_class].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to .nii.gz file")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    args = parser.parse_args()
    
    predict(args)
