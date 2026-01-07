# Transformer-Based Lightweight Model for Early Detection of Brain Tumors

This project implements a lightweight 3D Transformer-based model (CNN-Transformer Hybrid) for analyzing Multi-Modal MRI scans (T1, T1ce, T2, FLAIR) to detect Brain Tumors.

## 📂 Project Structure

```
BrainTumorDetection/
├── data/               # Place your dataset here
├── src/
│   ├── model.py        # Lightweight 3D Transformer Model Definition
│   ├── dataset.py      # Data Loader for NIfTI files
│   ├── train.py        # Training Loop
│   ├── inference.py    # Inference Script
│   └── __init__.py
├── requirements.txt    # Python Dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Install Dependencies

Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

*Note: You may need to install PyTorch separately depending on your CUDA version. Visit [pytorch.org](https://pytorch.org).*

### 2. Dataset

This project is designed to work with Multi-Modal MRI data, similar to the **BraTS (Brain Tumor Segmentation) Challenge** dataset.
Each subject should have 4 sequences: **T1, T1ce, T2, FLAIR**.

Expected Directory Structure:
```
data/
    train/
        Subject_001/
            Subject_001_t1.nii.gz
            Subject_001_t1ce.nii.gz
            ...
        Subject_002/
            ...
    val/
        ...
```

### 3. Usage

#### Training
To train the model (uses synthetic data by default if no data found):
```bash
python src/train.py --epochs 20 --batch_size 4
```

To train on real data:
```bash
python src/train.py --data_dir ./data --epochs 50
```

#### Inference
To run prediction on a specific MRI file:
```bash
python src/inference.py --input_path data/sample.nii.gz
```

## 🧠 Model Architecture

The model uses a **Lightweight Hybrid Architecture**:
1. **3D Patch Embedding**: Uses 3D Convolution to reduce spatial dimensions and extract local features.
2. **Transformer Encoder**: 3D Attention mechanism to capture global context across the MRI volume.
3. **Efficiency**: Designed with fewer parameters than standard 3D ResNets or ViT formulations, suitable for faster inference.

## 📝 Configuration

You can adjust hyperparameters in `src/train.py` or pass them as arguments:
- `--embed_dim`: Embedding dimension (default: 64)
- `--layers`: Number of Transformer layers (default: 4)
- `--heads`: Number of Attention heads (default: 4)

---
*Created by Antigravity*
