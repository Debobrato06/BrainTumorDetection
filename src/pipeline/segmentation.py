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
    Convert segmentation logits → probability → binary mask & BraTS sub-region masks (ET, TC, WT).
    
    Args:
        seg_logits: torch tensor (B, C, D, H, W) or numpy array
        threshold: float [0,1], configurable sensitivity (default 0.5)
    
    Returns:
        binary_mask: numpy array (D, H, W), dtype bool (Whole Tumor)
        prob_map: numpy array (D, H, W), float32 probability values
        sub_masks: dict containing {'ET': et_mask, 'TC': tc_mask, 'WT': wt_mask}
    """
    try:
        import torch
        if isinstance(seg_logits, torch.Tensor):
            prob_map_raw = torch.sigmoid(seg_logits).squeeze(0).detach().cpu().numpy()
        else:
            prob_map_raw = 1.0 / (1.0 + np.exp(-np.array(seg_logits, dtype=np.float32))).squeeze()
    except ImportError:
        prob_map_raw = np.array(seg_logits, dtype=np.float32).squeeze()
        prob_map_raw = np.clip(prob_map_raw, 0.0, 1.0)

    sub_masks = {}
    if prob_map_raw.ndim == 4 and prob_map_raw.shape[0] >= 3:
        # Multi-channel output: [0: ET, 1: TC, 2: WT]
        prob_et = prob_map_raw[0]
        prob_tc = prob_map_raw[1]
        prob_wt = prob_map_raw[2]

        et_mask = (prob_et >= threshold).astype(bool)
        tc_mask = (prob_tc >= threshold).astype(bool) | et_mask
        wt_mask = (prob_wt >= threshold).astype(bool) | tc_mask

        binary_mask = wt_mask
        prob_map = prob_wt

        sub_masks = {
            'ET': et_mask,
            'TC': tc_mask,
            'WT': wt_mask
        }
    else:
        prob_map = prob_map_raw.squeeze()
        binary_mask = (prob_map >= threshold).astype(bool)
        sub_masks = {
            'ET': binary_mask,
            'TC': binary_mask,
            'WT': binary_mask
        }

    # Post-processing: remove very small noise regions (< 5 voxels)
    if SKIMAGE_AVAILABLE and binary_mask.any():
        try:
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=5)
            for k in sub_masks:
                if sub_masks[k].any():
                    sub_masks[k] = morphology.remove_small_objects(sub_masks[k], min_size=5)
        except Exception:
            pass

    return binary_mask, prob_map, sub_masks


def extract_tumor_mask_from_heuristic(volume_np, asym_peak, avg_asym, is_tumor):
    """
    Generate an image-adaptive tumor mask and BraTS sub-region masks (ET, TC, WT)
    directly from image intensity distribution and spatial asymmetry.
    """
    if volume_np.ndim == 4:
        shape = volume_np.shape[1:]
        img_3d = volume_np[min(3, volume_np.shape[0] - 1)]
    else:
        shape = volume_np.shape
        img_3d = volume_np

    if not is_tumor:
        empty_mask = np.zeros(shape, dtype=bool)
        empty_prob = np.zeros(shape, dtype=np.float32)
        return empty_mask, empty_prob, {'ET': empty_mask, 'TC': empty_mask, 'WT': empty_mask}

    try:
        from skimage import filters, exposure, morphology as morph, measure
        D, H, W = img_3d.shape
        prob_map = np.zeros((D, H, W), dtype=np.float32)
        binary_mask = np.zeros((D, H, W), dtype=bool)

        center_d = D // 2
        slice_indices = range(max(0, center_d - 6), min(D, center_d + 7))

        for d in slice_indices:
            sl = img_3d[d].copy()
            s_min, s_max = sl.min(), sl.max()
            if s_max <= s_min:
                continue
            sl_norm = (sl - s_min) / (s_max - s_min + 1e-8)
            sl_norm = np.clip(sl_norm, 0.0, 1.0)

            # Brain tissue mask (exclude dark background)
            bg_thresh = np.percentile(sl_norm, 20)
            brain_mask = sl_norm > bg_thresh

            if not np.any(brain_mask):
                continue

            try:
                sl_enh = exposure.equalize_adapthist(sl_norm, clip_limit=0.04)
            except Exception:
                sl_enh = sl_norm

            brain_pixels = sl_enh[brain_mask]
            p85 = np.percentile(brain_pixels, 85)
            p95 = np.percentile(brain_pixels, 95)

            # Identify hyperintense anomaly region (typical SOL/lesion behavior)
            lesion_candidates = (sl_enh >= p85) & brain_mask

            # Measure spatial asymmetry to focus on the anomalous side
            mid_w = W // 2
            left = sl_enh[:, :mid_w]
            right = np.fliplr(sl_enh[:, W - mid_w:])
            w_min = min(left.shape[1], right.shape[1])
            asym_map = np.abs(left[:, :w_min] - right[:, :w_min])

            asym_full = np.zeros_like(sl_enh)
            asym_full[:, :w_min] += asym_map
            asym_full[:, W - w_min:] += np.fliplr(asym_map)

            asym_thresh = np.percentile(asym_full[brain_mask], 75) if np.any(brain_mask) else 0.2
            lesion_mask_2d = lesion_candidates & (asym_full >= asym_thresh)

            if not np.any(lesion_mask_2d):
                lesion_mask_2d = (sl_enh >= p95) & brain_mask

            # Keep largest connected components (actual lesion)
            labeled = measure.label(lesion_mask_2d)
            if labeled.max() > 0:
                regions = measure.regionprops(labeled)
                regions.sort(key=lambda r: r.area, reverse=True)
                valid_mask = np.zeros_like(lesion_mask_2d, dtype=bool)
                for r in regions[:2]:  # Keep top 2 lesion components
                    if r.area >= 15:
                        valid_mask[labeled == r.label] = True
                lesion_mask_2d = valid_mask

            binary_mask[d] = lesion_mask_2d
            prob_map[d] = (sl_enh * 0.7) + (asym_full * 0.3)

        # 3D Post-processing: cleanup small noise
        if SKIMAGE_AVAILABLE and binary_mask.any():
            try:
                binary_mask = morph.remove_small_objects(binary_mask, min_size=15)
            except Exception:
                pass

        # Build realistic sub-region masks (ET inside TC inside WT)
        wt_mask = binary_mask.copy()
        tc_mask = np.zeros_like(wt_mask)
        et_mask = np.zeros_like(wt_mask)

        if wt_mask.any():
            try:
                for d in range(D):
                    if wt_mask[d].any():
                        tc_mask[d] = morph.binary_erosion(wt_mask[d], morph.disk(1))
                        et_mask[d] = morph.binary_erosion(wt_mask[d], morph.disk(2))
            except Exception:
                tc_mask = wt_mask.copy()
                et_mask = wt_mask.copy()

        # Ensure non-empty sub-masks if WT exists
        if wt_mask.any() and not tc_mask.any():
            tc_mask = wt_mask.copy()
        if wt_mask.any() and not et_mask.any():
            et_mask = tc_mask.copy()

        sub_masks = {
            'ET': et_mask,
            'TC': tc_mask,
            'WT': wt_mask
        }
        return binary_mask, prob_map, sub_masks

    except Exception as e:
        print(f"[Heuristic Mask] Exception during extraction: {e}")
        empty_mask = np.zeros(shape, dtype=bool)
        empty_prob = np.zeros(shape, dtype=np.float32)
        return empty_mask, empty_prob, {'ET': empty_mask, 'TC': empty_mask, 'WT': empty_mask}


def generate_segmentation_overlay(volume_np, binary_mask, channel=0, sub_masks=None):
    """
    Generate base64 side-by-side: original MRI slice | color-coded multi-region overlay.
    - ET (Enhancing Tumor): Red (#EF4444)
    - TC (Tumor Core): Yellow (#F59E0B)
    - WT / Edema (Peritumoral zone): Green (#10B981)
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""

    try:
        if volume_np.ndim == 4:
            img_3d = volume_np[channel]
        else:
            img_3d = volume_np

        D = img_3d.shape[0]
        if binary_mask.any():
            slice_counts = binary_mask.sum(axis=(1, 2))
            best_d = int(np.argmax(slice_counts))
        else:
            best_d = D // 2

        orig_slice = img_3d[best_d]
        mask_slice = binary_mask[best_d] if binary_mask.ndim == 3 else np.zeros_like(orig_slice, dtype=bool)

        s_min, s_max = orig_slice.min(), orig_slice.max()
        if s_max > s_min:
            orig_norm = (orig_slice - s_min) / (s_max - s_min)
        else:
            orig_norm = orig_slice.copy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), facecolor='#0d1117')

        # Left: Original MRI
        axes[0].imshow(orig_norm, cmap='gray', interpolation='bilinear')
        axes[0].set_title('Multi-Modal MRI Slice', color='#8b949e', fontsize=11, pad=8)
        axes[0].axis('off')

        # Right: Overlay
        axes[1].imshow(orig_norm, cmap='gray', interpolation='bilinear')
        if mask_slice.any():
            overlay_rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)

            et_slice = sub_masks['ET'][best_d] if sub_masks and 'ET' in sub_masks and sub_masks['ET'].ndim == 3 else mask_slice
            tc_slice = sub_masks['TC'][best_d] if sub_masks and 'TC' in sub_masks and sub_masks['TC'].ndim == 3 else mask_slice
            wt_slice = sub_masks['WT'][best_d] if sub_masks and 'WT' in sub_masks and sub_masks['WT'].ndim == 3 else mask_slice

            # WT / Vasogenic Edema -> Green (#10B981)
            overlay_rgba[wt_slice, 0] = 0.06
            overlay_rgba[wt_slice, 1] = 0.72
            overlay_rgba[wt_slice, 2] = 0.50
            overlay_rgba[wt_slice, 3] = 0.40

            # TC / Tumor Core -> Yellow (#F59E0B)
            overlay_rgba[tc_slice, 0] = 0.96
            overlay_rgba[tc_slice, 1] = 0.62
            overlay_rgba[tc_slice, 2] = 0.04
            overlay_rgba[tc_slice, 3] = 0.60

            # ET / Enhancing Tumor -> Red (#EF4444)
            overlay_rgba[et_slice, 0] = 0.93
            overlay_rgba[et_slice, 1] = 0.26
            overlay_rgba[et_slice, 2] = 0.26
            overlay_rgba[et_slice, 3] = 0.80

            axes[1].imshow(overlay_rgba, interpolation='bilinear')

            # Legend
            patch_et = mpatches.Patch(color='#EF4444', label='ET (Enhancing Tumor)')
            patch_tc = mpatches.Patch(color='#F59E0B', label='TC (Tumor Core)')
            patch_wt = mpatches.Patch(color='#10B981', label='WT (Whole Tumor / Edema)')
            axes[1].legend(handles=[patch_et, patch_tc, patch_wt], loc='lower right',
                           facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', fontsize=8)

        axes[1].set_title(f'BraTS SE-Attention Overlay (Slice {best_d})', color='#8b949e', fontsize=11, pad=8)
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
