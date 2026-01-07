# High-Precision Joint Tumor Classification & Segmentation Framework

## Overview
This repository contains a research-grade framework designed for **Joint Tumor Classification and Segmentation** using 3D MRI/CT scans. It outperforms standard U-Net baselines by leveraging a **Hybrid CNN-Transformer Architecture**.

## Key Innovations

### 1. Hybrid Architecture (CNN + Transformer)
- **Implementation**: `src/hybrid_model.py` -> `HybridTumorModel`
- **Logic**: Uses a CNN Stem (ResNet-style) for local feature extraction and a Vision Transformer (ViT/Swin-like) bottleneck to capture long-range global dependencies.

### 2. Self-Supervised Pre-training (SSL)
- **Implementation**: `src/lightning_module.py` (Mode: `pretrain`)
- **Logic**: Supports Masked Autoencoders (MAE). The model can train on unlabeled data by masking patches and reconstructing the volume, learning robust anatomical features before fine-tuning.

### 3. Uncertainty Estimation (Aleatoric & Epistemic)
- **Implementation**: `src/hybrid_model.py` -> `get_uncertainty_map()`
- **Logic**: Monte Carlo Dropout is active during inference to generate a "Confidence Map" (Variance of predictions), quantifying the model's certainty.

### 4. Advanced Loss Functions
- **Implementation**: `src/losses.py`
- **Logic**: Combines **Tversky Loss** (handling class imbalance for small lesions) with standard Cross-Entropy for classification.

### 5. Cross-Modality Transfer Learning
- **Implementation**: `src/hybrid_model.py` -> `inflate_2d_weights()`
- **Logic**: Placeholder for initializing 3D kernels from 2D Pre-trained (ImageNet/X-Ray) weights.

### 6. Explainability (XAI)
- **Implementation**: `HybridTumorModel` returns Attention Maps.
- **Logic**: `return_attention=True` allows visualization of transformer attention weights to interpret global focus.

## Project Structure
- `src/hybrid_model.py`: Core architecture.
- `src/lightning_module.py`: PyTorch Lightning module for training dynamics.
- `src/losses.py`: Custom loss functions.
- `src/training_pipeline.py`: Main entry point for training with DDP/Mixed Precision.
- `src/dataset_research.py`: Mock dataset generator for joint tasks.

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```
(Requires `pytorch-lightning`)

### 1. Self-Supervised Pre-training (SSL)
```bash
python src/training_pipeline.py --mode pretrain --epochs 50 --mask_ratio 0.75
```

### 2. Supervised Fine-Tuning (Joint Task)
```bash
python src/training_pipeline.py --mode finetune --epochs 100 --lr 1e-4
```

### 3. Distributed Training (DDP)
The script automatically detects GPUs. For multi-GPU:
```bash
python src/training_pipeline.py --epochs 100 --batch_size 16
```
