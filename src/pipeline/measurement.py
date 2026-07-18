"""
Stage 3: Size & Volume Measurement Logic
- Implements RANO/RECIST-criteria measurements
- Volume formula: voxel_count × (px × py × st) / 1000 → mL
- Longest diameter from bounding box (RANO standard)
- Growth tracking: % change vs previous scan
"""

import numpy as np


def compute_volume_ml(binary_mask, pixel_spacing_x, pixel_spacing_y, slice_thickness):
    """
    Compute tumor volume in milliliters (mL / cm³).
    
    Formula (RANO standard):
        Volume (mL) = voxel_count × (pixel_spacing_x × pixel_spacing_y × slice_thickness) / 1000
    
    Args:
        binary_mask: numpy bool array (D, H, W)
        pixel_spacing_x: float, mm per pixel in X
        pixel_spacing_y: float, mm per pixel in Y
        slice_thickness: float, mm per slice in Z
    
    Returns:
        volume_ml: float, rounded to 2 decimal places
        voxel_count: int
    """
    if not binary_mask.any():
        return 0.0, 0

    voxel_count = int(binary_mask.sum())
    voxel_volume_mm3 = pixel_spacing_x * pixel_spacing_y * slice_thickness
    volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0

    return round(volume_ml, 2), voxel_count


def compute_longest_diameter_mm(binary_mask, pixel_spacing_x, pixel_spacing_y):
    """
    Compute the longest tumor diameter in mm using RANO criteria.
    
    RANO standard: longest diameter measured in the axial plane
    (max of bounding box dimensions in X, Y directions).
    
    Args:
        binary_mask: numpy bool array (D, H, W)
        pixel_spacing_x: float, mm/pixel X
        pixel_spacing_y: float, mm/pixel Y
    
    Returns:
        longest_diameter_mm: float
        perpendicular_diameter_mm: float (for RANO bi-dimensional)
        bounding_box_dims: dict
    """
    if not binary_mask.any():
        return 0.0, 0.0, {}

    indices = np.argwhere(binary_mask)
    bb_min = indices.min(axis=0)  # [D_min, H_min, W_min]
    bb_max = indices.max(axis=0)  # [D_max, H_max, W_max]

    # Dimensions in voxels
    d_vox = int(bb_max[0] - bb_min[0] + 1)
    h_vox = int(bb_max[1] - bb_min[1] + 1)
    w_vox = int(bb_max[2] - bb_min[2] + 1)

    # Convert to mm
    d_mm = d_vox * 5.0            # Depth uses slice thickness (simplified — caller should use actual)
    h_mm = h_vox * pixel_spacing_y
    w_mm = w_vox * pixel_spacing_x

    # RANO: longest axial diameter = max(H, W)
    axial_diameters = sorted([h_mm, w_mm], reverse=True)
    longest_diameter_mm = axial_diameters[0]
    perpendicular_diameter_mm = axial_diameters[1]

    return (
        round(longest_diameter_mm, 1),
        round(perpendicular_diameter_mm, 1),
        {
            'depth_vox': d_vox, 'height_vox': h_vox, 'width_vox': w_vox,
            'depth_mm': round(d_mm, 1), 'height_mm': round(h_mm, 1), 'width_mm': round(w_mm, 1)
        }
    )


def compute_longest_diameter_rano(binary_mask, pixel_spacing_x, pixel_spacing_y, slice_thickness):
    """
    Full RANO computation with proper slice thickness in all 3 axes.
    Returns longest diameter, perpendicular, and full dimension info.
    """
    if not binary_mask.any():
        return 0.0, 0.0, {}

    indices = np.argwhere(binary_mask)
    bb_min = indices.min(axis=0)
    bb_max = indices.max(axis=0)

    d_vox = int(bb_max[0] - bb_min[0] + 1)
    h_vox = int(bb_max[1] - bb_min[1] + 1)
    w_vox = int(bb_max[2] - bb_min[2] + 1)

    d_mm = round(d_vox * slice_thickness, 1)
    h_mm = round(h_vox * pixel_spacing_y, 1)
    w_mm = round(w_vox * pixel_spacing_x, 1)

    # RANO longest diameter: max axial plane dimension
    all_dims = sorted([h_mm, w_mm, d_mm], reverse=True)
    longest_diameter_mm = all_dims[0]
    perpendicular_diameter_mm = all_dims[1]

    return (
        longest_diameter_mm,
        perpendicular_diameter_mm,
        {
            'depth_mm': d_mm, 'height_mm': h_mm, 'width_mm': w_mm,
            'depth_vox': d_vox, 'height_vox': h_vox, 'width_vox': w_vox
        }
    )


def compute_growth_change(current_vol_ml, previous_vol_ml):
    """
    Compute % volume change vs previous scan (for follow-up reports).
    
    Criteria:
        RECIST: ≥20% increase → progression
        RANO: ≥25% increase in sum of perpendicular diameters
    
    Args:
        current_vol_ml: float
        previous_vol_ml: float
    
    Returns:
        dict with:
            - percent_change: float
            - trend: 'growth' | 'reduction' | 'stable'
            - clinical_significance: str
    """
    if previous_vol_ml is None or previous_vol_ml <= 0:
        return {
            'percent_change': None,
            'trend': 'baseline',
            'clinical_significance': 'No previous scan available — baseline measurement.'
        }

    delta = current_vol_ml - previous_vol_ml
    pct = (delta / previous_vol_ml) * 100.0

    if pct >= 25.0:
        trend = 'growth'
        significance = f'PROGRESSION: Volume increased {pct:.1f}% (RANO criteria: ≥25% = progression).'
    elif pct <= -25.0:
        trend = 'reduction'
        significance = f'RESPONSE: Volume reduced {abs(pct):.1f}% (partial/complete response indicator).'
    elif abs(pct) < 10.0:
        trend = 'stable'
        significance = f'STABLE DISEASE: Volume change {pct:+.1f}% (within ±10% threshold).'
    elif pct > 0:
        trend = 'mild_growth'
        significance = f'MILD GROWTH: Volume increased {pct:.1f}% — monitor closely.'
    else:
        trend = 'mild_reduction'
        significance = f'MILD REDUCTION: Volume decreased {abs(pct):.1f}% — possibly treatment response.'

    return {
        'percent_change': round(pct, 1),
        'absolute_change_ml': round(delta, 2),
        'trend': trend,
        'clinical_significance': significance
    }


def get_size_category(diameter_mm):
    """
    Clinical size categorization for risk scoring.
    Based on general clinical thresholds (not WHO-specific).
    """
    if diameter_mm <= 0:
        return 'unmeasurable', 0
    elif diameter_mm < 15:
        return 'small (<15mm)', 1
    elif diameter_mm < 30:
        return 'moderate (15-30mm)', 2
    elif diameter_mm < 40:
        return 'large (30-40mm)', 3
    else:
        return 'very large (>40mm)', 4
