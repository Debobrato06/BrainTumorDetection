"""
Stage 4: Radiomics Feature Extraction
- Shape features: sphericity, irregularity (3D convex hull analysis)
- Enhancement pattern: contrast channel intensity ratio inside vs outside mask
- Peritumoral edema: expanded mask zone analysis
- Anatomical location: centroid-based atlas mapping
- Texture features: GLCM (homogeneity, entropy, contrast, correlation)

Decision-support only. Not for clinical malignancy diagnosis.
"""

import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
    from skimage import measure, morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ─── Shape Features ──────────────────────────────────────────────────────────

def compute_shape_features(binary_mask):
    """
    Compute 3D shape features from binary tumor mask.
    
    Features:
        - sphericity: how sphere-like (1.0 = perfect sphere, <0.6 = irregular/elongated)
        - irregularity: surface irregularity index (higher = more complex boundary)
        - solidity: ratio of object volume to convex hull volume
        - extent: ratio of object to bounding box
        - elongation: max_axis / min_axis ratio
    
    Returns:
        dict of shape features
    """
    if not SKIMAGE_AVAILABLE or not binary_mask.any():
        return _default_shape_features()

    try:
        # Use regionprops_3D via skimage measure
        labeled = measure.label(binary_mask)
        regions = measure.regionprops(labeled)

        if not regions:
            return _default_shape_features()

        # Take the largest region
        region = max(regions, key=lambda r: r.area)

        volume_vox = float(region.area)
        bbox = region.bbox  # (d0, h0, w0, d1, h1, w1)
        
        # Filled area (convex-approximation via filled_image)
        try:
            convex_vol = float(region.convex_area) if hasattr(region, 'convex_area') else volume_vox
        except Exception:
            convex_vol = volume_vox

        solidity = round(volume_vox / (convex_vol + 1e-6), 3)

        # Sphericity: (π^(1/3) × (6V)^(2/3)) / A
        # Approximate surface area from voxel count (simplified)
        # True surface area would need marching cubes — approximate here
        surf_approx = _approximate_surface_area(binary_mask)
        if surf_approx > 0:
            sphericity = round(
                (np.pi ** (1/3)) * ((6 * volume_vox) ** (2/3)) / (surf_approx + 1e-6),
                3
            )
            sphericity = min(1.0, max(0.0, sphericity))
        else:
            sphericity = 0.5

        # Elongation: max / min bounding box side
        sides = [
            bbox[3] - bbox[0],  # depth
            bbox[4] - bbox[1],  # height
            bbox[5] - bbox[2],  # width
        ]
        sides = [max(1, s) for s in sides]
        elongation = round(max(sides) / min(sides), 3)

        # Extent: fraction of bounding box filled
        bbox_vol = sides[0] * sides[1] * sides[2]
        extent = round(volume_vox / (bbox_vol + 1e-6), 3)

        # Irregularity = 1 - solidity (irregular shape has lower solidity)
        irregularity = round(1.0 - solidity, 3)

        return {
            'sphericity': sphericity,
            'irregularity': irregularity,
            'solidity': solidity,
            'extent': extent,
            'elongation': elongation,
            'shape_label': _classify_shape(sphericity, irregularity),
            'surface_area_approx_vox': round(surf_approx, 1)
        }

    except Exception as e:
        print(f"[Radiomics] Shape feature extraction failed: {e}")
        return _default_shape_features()


def _approximate_surface_area(mask):
    """Voxel-face-counting approximation for surface area."""
    try:
        # Count faces exposed at the boundary
        # A voxel face is "surface" if the adjacent voxel is background
        sa = 0
        padded = np.pad(mask, 1, mode='constant', constant_values=False)
        for shift, axis in [
            ((1, 0, 0), 0), ((-1, 0, 0), 0),
            ((0, 1, 0), 1), ((0, -1, 0), 1),
            ((0, 0, 1), 2), ((0, 0, -1), 2)
        ]:
            shifted = np.roll(padded, shift[axis], axis=axis)[1:-1, 1:-1, 1:-1]
            sa += np.sum(mask & ~shifted)
        return float(sa)
    except Exception:
        return float(mask.sum() ** (2/3)) * 4.84  # Sphere approximation


def _classify_shape(sphericity, irregularity):
    if sphericity >= 0.8 and irregularity < 0.25:
        return 'Well-defined, round (low aggression indicator)'
    elif sphericity >= 0.6:
        return 'Moderately irregular border'
    else:
        return 'Highly irregular / infiltrative border (high aggression indicator)'


def _default_shape_features():
    return {
        'sphericity': 0.0, 'irregularity': 0.0, 'solidity': 0.0,
        'extent': 0.0, 'elongation': 1.0,
        'shape_label': 'Insufficient mask data',
        'surface_area_approx_vox': 0.0
    }


# ─── Enhancement Pattern ─────────────────────────────────────────────────────

def estimate_enhancement_pattern(volume_np, binary_mask):
    """
    Estimate contrast enhancement pattern from MRI channels.
    
    In multi-channel MRI (T1, T1CE, T2, FLAIR):
        Enhancement ratio = mean(T1CE inside mask) / mean(T1CE outside mask + ε)
    
    Enhancement categories:
        > 2.5 → Strong ring/nodular enhancement (high-grade indicator)
        1.5–2.5 → Moderate enhancement
        < 1.5 → Minimal/no enhancement (low-grade indicator)
    
    For single-channel input, uses intensity percentile analysis.
    """
    try:
        if volume_np.ndim == 4:
            # Multi-channel: T1CE is typically channel index 1 (T1, T1CE, T2, FLAIR)
            n_ch = volume_np.shape[0]
            t1ce_idx = min(1, n_ch - 1)
            t1ce = volume_np[t1ce_idx]
        else:
            t1ce = volume_np

        if not binary_mask.any():
            return _default_enhancement()

        # Normalize
        t1ce = t1ce.astype(np.float32)
        p1, p99 = np.percentile(t1ce, 1), np.percentile(t1ce, 99)
        if p99 > p1:
            t1ce_norm = (t1ce - p1) / (p99 - p1 + 1e-8)
        else:
            t1ce_norm = t1ce

        inside_mean = float(t1ce_norm[binary_mask].mean())
        outside_mask = ~binary_mask
        if outside_mask.any():
            outside_mean = float(t1ce_norm[outside_mask].mean())
        else:
            outside_mean = 0.3

        enhancement_ratio = round(inside_mean / (outside_mean + 1e-6), 3)
        enhancement_ratio = min(5.0, enhancement_ratio)  # Cap for display

        # Classify pattern
        if enhancement_ratio >= 2.5:
            pattern = 'Strong / Ring Enhancement'
            severity = 'high'
        elif enhancement_ratio >= 1.5:
            pattern = 'Moderate Enhancement'
            severity = 'moderate'
        else:
            pattern = 'Minimal / No Enhancement'
            severity = 'low'

        # Ring enhancement detection: check if periphery > core intensity
        is_ring = _detect_ring_enhancement(t1ce_norm, binary_mask)

        return {
            'enhancement_ratio': enhancement_ratio,
            'enhancement_pattern': pattern,
            'is_ring_enhancement': is_ring,
            'severity': severity,
            'inside_mean_intensity': round(inside_mean, 4),
            'outside_mean_intensity': round(outside_mean, 4)
        }

    except Exception as e:
        print(f"[Radiomics] Enhancement estimation failed: {e}")
        return _default_enhancement()


def _detect_ring_enhancement(t1ce_norm, binary_mask):
    """Check if intensity is higher at tumor periphery than core (ring-enhancement pattern)."""
    try:
        if not SKIMAGE_AVAILABLE or not binary_mask.any():
            return False
        # Erode mask to get core
        core = morphology.binary_erosion(binary_mask, footprint=morphology.ball(2))
        periphery = binary_mask & ~core
        if not core.any() or not periphery.any():
            return False
        core_mean = t1ce_norm[core].mean()
        periphery_mean = t1ce_norm[periphery].mean()
        return bool(periphery_mean > core_mean * 1.2)  # 20% brighter periphery
    except Exception:
        return False


def _default_enhancement():
    return {
        'enhancement_ratio': 0.0,
        'enhancement_pattern': 'Cannot be determined',
        'is_ring_enhancement': False,
        'severity': 'unknown',
        'inside_mean_intensity': 0.0,
        'outside_mean_intensity': 0.0
    }


# ─── Peritumoral Edema ───────────────────────────────────────────────────────

def estimate_edema_extent(binary_mask, volume_np):
    """
    Estimate peritumoral edema by analyzing a dilated zone around the tumor mask.
    
    In FLAIR channel, edema appears as hyper-intense halo around the tumor core.
    Edema ratio = peritumoral zone volume / tumor core volume
    
    Returns:
        edema_ratio: float (>2.0 = extensive edema, mass effect indicator)
        edema_volume_vox: approximate edema voxel count
        edema_label: clinical severity label
    """
    try:
        if not SKIMAGE_AVAILABLE or not binary_mask.any():
            return _default_edema()

        # Dilate mask to get peritumoral zone (2–3 voxel ring)
        dilated = morphology.binary_dilation(binary_mask, footprint=morphology.ball(3))
        peritumoral_zone = dilated & ~binary_mask

        tumor_vol = int(binary_mask.sum())
        edema_zone_vol = int(peritumoral_zone.sum())

        # Check FLAIR channel (index 3 for T1/T1CE/T2/FLAIR, or last available)
        if volume_np.ndim == 4:
            flair_idx = min(3, volume_np.shape[0] - 1)
            flair = volume_np[flair_idx].astype(np.float32)
        else:
            flair = volume_np.astype(np.float32)

        # Normalize FLAIR
        p1, p99 = np.percentile(flair, 1), np.percentile(flair, 99)
        if p99 > p1:
            flair_norm = (flair - p1) / (p99 - p1 + 1e-8)
        else:
            flair_norm = flair

        # Edema appears hyper-intense in FLAIR
        if peritumoral_zone.any():
            zone_intensity = float(flair_norm[peritumoral_zone].mean())
        else:
            zone_intensity = 0.0

        background_mask = ~dilated
        if background_mask.any():
            bg_intensity = float(flair_norm[background_mask].mean())
        else:
            bg_intensity = 0.5

        # Edema severity: peritumoral intensity elevation above background
        edema_intensity_ratio = round(zone_intensity / (bg_intensity + 1e-6), 3)

        # Edema volume ratio: how large is the edema zone relative to tumor
        edema_vol_ratio = round(edema_zone_vol / (tumor_vol + 1e-6), 2)

        # Combined edema score
        edema_score = (edema_intensity_ratio * 0.5) + (min(edema_vol_ratio, 5.0) / 5.0 * 0.5)

        if edema_vol_ratio >= 3.0 or edema_intensity_ratio >= 1.5:
            label = 'Extensive Edema — Significant Mass Effect'
            severity = 'high'
        elif edema_vol_ratio >= 1.5:
            label = 'Moderate Peritumoral Edema'
            severity = 'moderate'
        else:
            label = 'Minimal Edema'
            severity = 'low'

        return {
            'edema_ratio': edema_vol_ratio,
            'edema_intensity_ratio': edema_intensity_ratio,
            'edema_volume_vox': edema_zone_vol,
            'edema_score': round(edema_score, 3),
            'edema_label': label,
            'severity': severity
        }

    except Exception as e:
        print(f"[Radiomics] Edema estimation failed: {e}")
        return _default_edema()


def _default_edema():
    return {
        'edema_ratio': 0.0, 'edema_intensity_ratio': 0.0,
        'edema_volume_vox': 0, 'edema_score': 0.0,
        'edema_label': 'Cannot be determined', 'severity': 'unknown'
    }


# ─── Anatomical Location ─────────────────────────────────────────────────────

# Eloquent areas (high surgical risk) vs non-eloquent
ELOQUENT_REGIONS = {
    'Brainstem', 'Thalamus / Deep Structures', 'Motor Cortex (Paracentral)',
    'Wernicke Area (Left Temporal)', 'Broca Area (Left Frontal)'
}


def infer_anatomical_location(binary_mask, volume_shape):
    """
    Infer approximate anatomical location from tumor mask centroid.
    Maps 3D centroid coordinates to anatomical regions using simplified atlas.
    
    Args:
        binary_mask: numpy bool (D, H, W)
        volume_shape: tuple (D, H, W) — full volume dimensions
    
    Returns:
        dict with:
            - region: anatomical region name
            - laterality: 'Left' | 'Right' | 'Midline'
            - is_eloquent: bool (surgical risk indicator)
            - depth_category: 'Cortical' | 'Subcortical' | 'Deep'
    """
    if not binary_mask.any():
        return _default_location()

    try:
        indices = np.argwhere(binary_mask)
        centroid = indices.mean(axis=0)  # [D, H, W]

        D, H, W = volume_shape[-3], volume_shape[-2], volume_shape[-1]

        # Normalized centroid [0, 1]
        norm_d = centroid[0] / max(D - 1, 1)  # Superior-Inferior
        norm_h = centroid[1] / max(H - 1, 1)  # Anterior-Posterior
        norm_w = centroid[2] / max(W - 1, 1)  # Left-Right

        # Laterality
        if norm_w < 0.4:
            laterality = 'Right'   # In standard MRI orientation
        elif norm_w > 0.6:
            laterality = 'Left'
        else:
            laterality = 'Midline'

        # Depth category (based on distance from surface approximation)
        depth_category = _infer_depth(norm_d, norm_h, norm_w)

        # Region mapping (simplified atlas based on normalized coordinates)
        region = _map_to_region(norm_d, norm_h, norm_w)

        is_eloquent = region in ELOQUENT_REGIONS

        return {
            'region': region,
            'laterality': laterality,
            'full_location': f'{laterality} {region}' if laterality != 'Midline' else region,
            'is_eloquent': is_eloquent,
            'depth_category': depth_category,
            'surgical_risk': 'High' if is_eloquent or depth_category == 'Deep' else 'Moderate' if depth_category == 'Subcortical' else 'Lower',
            'centroid_norm': [round(norm_d, 3), round(norm_h, 3), round(norm_w, 3)]
        }

    except Exception as e:
        print(f"[Radiomics] Location inference failed: {e}")
        return _default_location()


def _map_to_region(norm_d, norm_h, norm_w):
    """Simplified atlas mapping from normalized centroid → anatomical region."""
    # Superior-Inferior: norm_d 0=top, 1=bottom
    # Anterior-Posterior: norm_h 0=front, 1=back
    # Left-Right: norm_w

    # Brainstem / cerebellum (inferior)
    if norm_d > 0.78:
        return 'Brainstem'

    # Thalamus / deep structures (central)
    if 0.35 < norm_d < 0.65 and 0.4 < norm_h < 0.65 and 0.35 < norm_w < 0.65:
        return 'Thalamus / Deep Structures'

    # Cerebellum (inferior-posterior)
    if norm_d > 0.65 and norm_h > 0.6:
        return 'Cerebellum'

    # Frontal lobe (anterior, superior)
    if norm_h < 0.4:
        if norm_d < 0.5:
            if 0.3 < norm_w < 0.5:
                return 'Broca Area (Left Frontal)'
            return 'Frontal Lobe'
        return 'Frontal Lobe'

    # Occipital lobe (posterior)
    if norm_h > 0.72:
        return 'Occipital Lobe'

    # Temporal lobe (inferior, mid-posterior)
    if norm_d > 0.5 and 0.45 < norm_h < 0.72:
        if norm_w > 0.55:
            return 'Wernicke Area (Left Temporal)'
        return 'Temporal Lobe'

    # Parietal lobe (superior-posterior)
    if norm_h >= 0.4:
        if norm_d < 0.35:
            return 'Motor Cortex (Paracentral)'
        return 'Parietal Lobe'

    return 'Frontal Lobe'  # Default


def _infer_depth(norm_d, norm_h, norm_w):
    """Infer cortical / subcortical / deep based on distance from center."""
    center_dist = np.sqrt((norm_d - 0.5)**2 + (norm_h - 0.5)**2 + (norm_w - 0.5)**2)
    if center_dist < 0.2:
        return 'Deep'
    elif center_dist < 0.35:
        return 'Subcortical'
    else:
        return 'Cortical'


def _default_location():
    return {
        'region': 'Undetermined', 'laterality': 'Unknown',
        'full_location': 'Undetermined', 'is_eloquent': False,
        'depth_category': 'Unknown', 'surgical_risk': 'Unknown',
        'centroid_norm': [0.5, 0.5, 0.5]
    }


# ─── Texture Features (GLCM) ─────────────────────────────────────────────────

def compute_texture_features(volume_np, binary_mask):
    """
    Compute GLCM-based texture features inside the tumor mask.
    
    Features (from Gray Level Co-occurrence Matrix):
        - homogeneity: low → heterogeneous tumor (more aggressive indicator)
        - contrast: high → high intensity variation
        - correlation: spatial intensity correlation
        - energy: textural uniformity
        - entropy: randomness (high entropy → heterogeneous texture)
    
    Returns:
        dict of texture features
    """
    if not SKIMAGE_AVAILABLE or not binary_mask.any():
        return _default_texture()

    try:
        # Use middle axial slice with most tumor voxels
        if volume_np.ndim == 4:
            img_3d = volume_np[0]  # T1 channel
        else:
            img_3d = volume_np

        # Find best slice
        slice_counts = binary_mask.sum(axis=(1, 2))
        if slice_counts.max() == 0:
            return _default_texture()
        best_d = int(np.argmax(slice_counts))

        img_slice = img_3d[best_d].astype(np.float32)
        mask_slice = binary_mask[best_d]

        if not mask_slice.any():
            return _default_texture()

        # Quantize to 32 levels for GLCM
        p1, p99 = np.percentile(img_slice, 1), np.percentile(img_slice, 99)
        if p99 > p1:
            img_norm = (img_slice - p1) / (p99 - p1 + 1e-8)
        else:
            img_norm = img_slice

        img_quant = (np.clip(img_norm, 0, 1) * 31).astype(np.uint8)

        # Extract ROI
        rows, cols = np.where(mask_slice)
        if len(rows) < 4:
            return _default_texture()

        r_min, r_max = rows.min(), rows.max() + 1
        c_min, c_max = cols.min(), cols.max() + 1
        roi = img_quant[r_min:r_max, c_min:c_max]

        if roi.shape[0] < 2 or roi.shape[1] < 2:
            return _default_texture()

        # Compute GLCM
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(roi, distances=distances, angles=angles,
                            levels=32, symmetric=True, normed=True)

        homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
        contrast = float(graycoprops(glcm, 'contrast').mean())
        correlation = float(graycoprops(glcm, 'correlation').mean())
        energy = float(graycoprops(glcm, 'energy').mean())

        # Entropy from GLCM (not a direct graycoprops property — compute manually)
        glcm_mean = glcm.mean(axis=(2, 3))
        glcm_mean_safe = glcm_mean + 1e-10
        entropy = float(-np.sum(glcm_mean_safe * np.log2(glcm_mean_safe)))

        # Heterogeneity label
        if homogeneity < 0.4:
            texture_label = 'Highly Heterogeneous (aggressive indicator)'
        elif homogeneity < 0.65:
            texture_label = 'Moderately Heterogeneous'
        else:
            texture_label = 'Relatively Homogeneous'

        return {
            'homogeneity': round(homogeneity, 4),
            'contrast': round(contrast, 4),
            'correlation': round(correlation, 4),
            'energy': round(energy, 4),
            'entropy': round(entropy, 4),
            'texture_label': texture_label
        }

    except Exception as e:
        print(f"[Radiomics] Texture feature extraction failed: {e}")
        return _default_texture()


def _default_texture():
    return {
        'homogeneity': 0.0, 'contrast': 0.0, 'correlation': 0.0,
        'energy': 0.0, 'entropy': 0.0,
        'texture_label': 'Insufficient data for texture analysis'
    }


# ─── Main Feature Bundle ──────────────────────────────────────────────────────

def extract_all_radiomics(volume_np, binary_mask):
    """
    Run the full radiomics pipeline and return all features in one dict.
    Safe to call even if individual features fail (graceful degradation).
    """
    volume_shape = volume_np.shape
    return {
        'shape': compute_shape_features(binary_mask),
        'enhancement': estimate_enhancement_pattern(volume_np, binary_mask),
        'edema': estimate_edema_extent(binary_mask, volume_np),
        'location': infer_anatomical_location(binary_mask, volume_shape),
        'texture': compute_texture_features(volume_np, binary_mask)
    }
