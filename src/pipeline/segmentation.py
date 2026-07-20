"""
Stage 2: Tumor Segmentation — Mask Extraction & Overlay Generation
- Applies configurable probability threshold to seg_logits → binary mask
- Generates base64 segmentation overlay image (original MRI + color mask)
- Works with both 3D model output and 2D heuristic-based masks
"""

import numpy as np
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from skimage import morphology, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


DEFAULT_THRESHOLD = 0.5


def apply_threshold(seg_logits, threshold=DEFAULT_THRESHOLD):
    """
    Convert segmentation logits → probability → binary mask.
    
    Args:
        seg_logits: torch tensor (B, 1, D, H, W) or numpy array
        threshold: float [0,1], configurable sensitivity (default 0.5)
    
    Returns:
        binary_mask: numpy array (D, H, W), dtype bool
        prob_map: numpy array (D, H, W), float32 probability values
    """
    try:
        import torch
        if isinstance(seg_logits, torch.Tensor):
            prob_map = torch.sigmoid(seg_logits).squeeze().detach().cpu().numpy()
        else:
            # Sigmoid manually
            prob_map = 1.0 / (1.0 + np.exp(-np.array(seg_logits, dtype=np.float32)))
            prob_map = prob_map.squeeze()
    except ImportError:
        # No torch — treat input as already a probability array
        prob_map = np.array(seg_logits, dtype=np.float32).squeeze()
        prob_map = np.clip(prob_map, 0.0, 1.0)

    binary_mask = (prob_map >= threshold).astype(bool)

    # Post-processing: remove very small noise regions (< 5 voxels)
    if SKIMAGE_AVAILABLE and binary_mask.any():
        try:
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=5)
        except Exception:
            pass

    return binary_mask, prob_map


def extract_tumor_mask_from_heuristic(volume_np, asym_peak, avg_asym, is_tumor):
    """
    Fallback: Generate an approximate tumor mask from heuristic metrics
    when model seg output is unavailable (PyTorch not loaded).
    Uses intensity thresholding + asymmetry weighting.
    
    Args:
        volume_np: numpy (C, D, H, W) or (D, H, W)
        asym_peak: float, peak asymmetry from heuristic engine
        avg_asym: float, average asymmetry
        is_tumor: bool
    
    Returns:
        binary_mask: numpy (D, H, W)
        prob_map: numpy (D, H, W)
    """
    if not is_tumor:
        if volume_np.ndim == 4:
            shape = volume_np.shape[1:]
        else:
            shape = volume_np.shape
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32)

    try:
        from skimage import filters, exposure, morphology as morph
        # Use first channel (or modality 3 if available for FLAIR)
        if volume_np.ndim == 4:
            img = volume_np[min(3, volume_np.shape[0] - 1)]
        else:
            img = volume_np

        D, H, W = img.shape
        prob_map = np.zeros((D, H, W), dtype=np.float32)
        binary_mask = np.zeros((D, H, W), dtype=bool)

        # Process mid-slices
        center_d = D // 2
        for d_off in range(-8, 9):
            d = center_d + d_off
            if d < 0 or d >= D:
                continue
            sl = img[d]
            try:
                sl_enh = exposure.equalize_adapthist(sl, clip_limit=0.04)
                thresh_hi = np.percentile(sl_enh, 85)
                thresh_lo = np.percentile(sl_enh, 15)
                anomaly = (sl_enh > thresh_hi) | (sl_enh < thresh_lo)
                # Weight by asymmetry strength
                weight = min(1.0, asym_peak * 2.0)
                prob_map[d][anomaly] = weight * 0.8
                binary_mask[d] = (prob_map[d] > 0.4)
            except Exception:
                pass

        # Clean up small regions
        if SKIMAGE_AVAILABLE and binary_mask.any():
            try:
                binary_mask = morph.remove_small_objects(binary_mask, min_size=10)
            except Exception:
                pass

        return binary_mask, prob_map

    except Exception:
        if volume_np.ndim == 4:
            shape = volume_np.shape[1:]
        else:
            shape = volume_np.shape
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32)


def generate_segmentation_overlay(volume_np, binary_mask, channel=0):
    """
    Generate base64 side-by-side: original MRI slice | overlay with tumor mask.
    
    Args:
        volume_np: numpy (C, D, H, W) or (D, H, W)
        binary_mask: numpy (D, H, W) bool
        channel: which MRI channel to display (default 0 = T1)
    
    Returns:
        base64-encoded PNG string, or empty string if generation fails
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""

    try:
        # Extract volume
        if volume_np.ndim == 4:
            img_3d = volume_np[channel]
        else:
            img_3d = volume_np

        # Find best slice: most tumor voxels
        D = img_3d.shape[0]
        if binary_mask.any():
            slice_counts = binary_mask.sum(axis=(1, 2))
            best_d = int(np.argmax(slice_counts))
        else:
            best_d = D // 2

        orig_slice = img_3d[best_d]
        mask_slice = binary_mask[best_d] if binary_mask.ndim == 3 else np.zeros_like(orig_slice, dtype=bool)

        # Normalize slice to [0, 1]
        s_min, s_max = orig_slice.min(), orig_slice.max()
        if s_max > s_min:
            orig_norm = (orig_slice - s_min) / (s_max - s_min)
        else:
            orig_norm = orig_slice.copy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#0d1117')

        # Left: Original
        axes[0].imshow(orig_norm, cmap='gray', interpolation='bilinear')
        axes[0].set_title('MRI Slice', color='#8b949e', fontsize=11, pad=8)
        axes[0].axis('off')

        # Right: Overlay
        axes[1].imshow(orig_norm, cmap='gray', interpolation='bilinear')
        if mask_slice.any():
            # Tumor mask in translucent red
            overlay_rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            overlay_rgba[mask_slice, 0] = 1.0   # R
            overlay_rgba[mask_slice, 1] = 0.2   # G
            overlay_rgba[mask_slice, 2] = 0.2   # B
            overlay_rgba[mask_slice, 3] = 0.55  # Alpha
            axes[1].imshow(overlay_rgba, interpolation='bilinear')

        axes[1].set_title(f'Segmentation Overlay (Slice {best_d})', color='#8b949e', fontsize=11, pad=8)
        axes[1].axis('off')

        plt.tight_layout(pad=1.0)

        buf = BytesIO()
        plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight',
                    facecolor='#0d1117', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    except Exception as e:
        print(f"[Segmentation] Overlay generation failed: {e}")
        return ""


def get_mask_stats(binary_mask, prob_map=None, threshold=0.5):
    """Quick stats about the mask for downstream use."""
    if not binary_mask.any():
        return {
            'voxel_count': 0,
            'has_tumor': False,
            'bounding_box': None,
            'centroid': None,
            'seg_quality_score': 1.0
        }

    voxel_count = int(binary_mask.sum())
    indices = np.argwhere(binary_mask)
    centroid = indices.mean(axis=0).tolist()

    bb_min = indices.min(axis=0).tolist()
    bb_max = indices.max(axis=0).tolist()

    if prob_map is not None:
        # Scale quality score from [threshold, 1.0] to [0.0, 1.0]
        mean_prob = np.mean(prob_map[binary_mask])
        seg_quality_score = float((mean_prob - threshold) / (1.0 - threshold + 1e-8))
        seg_quality_score = max(0.0, min(1.0, seg_quality_score))
    else:
        seg_quality_score = 0.85

    return {
        'voxel_count': voxel_count,
        'has_tumor': True,
        'bounding_box': {'min': bb_min, 'max': bb_max},
        'centroid': centroid,
        'seg_quality_score': round(seg_quality_score, 3)
    }
