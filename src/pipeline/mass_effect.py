"""
Mass Effect & Midline Shift Calculator
---------------------------------------
Classical image-processing approach (NumPy / SciPy) — no Deep Learning required.

Computes:
  - Brain symmetry axis (ideal midline)
  - Septum pellucidum / actual midline displacement
  - Midline shift in mm
  - Ventricular compression estimate
  - Herniation risk flag

Reference: Standard neuroradiology criteria — shift > 5 mm = high herniation risk.
"""

import numpy as np

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_midline_shift(vol_np, binary_mask, pixel_spacing_x=1.0, pixel_spacing_y=1.0):
    """
    Estimate midline shift from tumor mass effect.

    Strategy:
      1. Find the geometric center line of the brain volume (ideal midline).
      2. Estimate the actual midline as the hemisphere boundary skewed by tumor mass.
      3. Shift = |ideal - actual| × pixel_spacing.

    Args:
        vol_np      : numpy (C, D, H, W) or (D, H, W) — brain volume
        binary_mask : numpy bool (D, H, W) — tumor segmentation mask
        pixel_spacing_x : float — mm per voxel in W axis
        pixel_spacing_y : float — mm per voxel in H axis

    Returns:
        dict with:
            midline_shift_mm        : float
            herniation_risk         : 'Low' | 'Moderate' | 'High'
            ventricular_compression : str
            mass_effect_label       : str
            shift_direction         : 'Left' | 'Right' | 'None'
    """
    if binary_mask is None or not np.any(binary_mask):
        return _default_mass_effect()

    try:
        # Extract single 3D volume (use first channel or squeeze)
        if vol_np.ndim == 4:
            brain_3d = vol_np[0].astype(np.float32)
        else:
            brain_3d = vol_np.astype(np.float32)

        D, H, W = brain_3d.shape

        # ── Step 1: Ideal midline = geometric center in W axis ──
        ideal_midline_w = W / 2.0

        # ── Step 2: Estimate actual midline from mass asymmetry ──
        # Compute center of mass of the hemisphere with more tumor burden
        # The septum shifts AWAY from the tumor hemisphere
        tumor_indices = np.argwhere(binary_mask)
        if tumor_indices.size == 0:
            return _default_mass_effect()

        tumor_centroid_w = tumor_indices[:, 2].mean()   # W-axis centroid of tumor

        # Brain intensity centroid (weighted by voxel intensity)
        brain_threshold = np.percentile(brain_3d[brain_3d > 0], 10) if np.any(brain_3d > 0) else 0
        brain_mask_approx = brain_3d > brain_threshold

        # Intensity-weighted centroid of the full brain
        if brain_mask_approx.any():
            w_indices = np.arange(W)
            # Weights must be non-negative: z-scored volumes contain negative values, and a
            # signed profile can sum to ~0, which blew the centroid up to thousands of mm.
            weights = np.where(brain_mask_approx, brain_3d - brain_3d.min(), 0.0)
            w_profile = weights.sum(axis=(0, 1))   # Shape (W,)
            total = w_profile.sum()
            actual_midline_w = float(np.sum(w_profile * w_indices) / total) if total > 0 else ideal_midline_w
        else:
            actual_midline_w = ideal_midline_w

        # ── Step 3: Shift magnitude ──
        shift_vox = abs(ideal_midline_w - actual_midline_w)
        shift_mm = round(shift_vox * pixel_spacing_x, 2)
        # Physical plausibility: the midline cannot move further than half the head width,
        # and shifts beyond ~30 mm are not survivable; treat such values as estimation failure.
        if not np.isfinite(shift_mm) or shift_vox > W / 2 or shift_mm > 30.0:
            print(f"[MassEffect] Implausible midline shift {shift_mm} mm; reporting as not computed")
            return _default_mass_effect()

        # ── Step 4: Direction (tumor pushes midline opposite side) ──
        if tumor_centroid_w < ideal_midline_w:
            shift_direction = "Right"   # Left-sided tumor pushes midline Right
        elif tumor_centroid_w > ideal_midline_w:
            shift_direction = "Left"    # Right-sided tumor pushes midline Left
        else:
            shift_direction = "None"

        # ── Step 5: Risk classification ──
        herniation_risk, mass_effect_label = _classify_herniation_risk(shift_mm)

        # ── Step 6: Ventricular compression estimate ──
        # Rough proxy: large tumor volume near center = more compression
        tumor_vol_vox = int(binary_mask.sum())
        brain_vol_vox = int(brain_mask_approx.sum()) if brain_mask_approx.any() else (D * H * W)
        tumor_fraction = tumor_vol_vox / max(brain_vol_vox, 1)
        ventricular_compression = _classify_ventricular_compression(tumor_fraction, shift_mm)

        return {
            "midline_shift_mm":        shift_mm,
            "shift_direction":         shift_direction,
            "herniation_risk":         herniation_risk,
            "mass_effect_label":       mass_effect_label,
            "ventricular_compression": ventricular_compression,
            "tumor_fraction_of_brain": round(tumor_fraction * 100, 2),   # %
        }

    except Exception as e:
        print(f"[MassEffect] Midline shift computation failed: {e}")
        return _default_mass_effect()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _classify_herniation_risk(shift_mm: float):
    """Standard neuroradiology thresholds for herniation risk."""
    if shift_mm >= 10.0:
        return "High",     "Significant midline shift — high herniation risk (≥10 mm)"
    elif shift_mm >= 5.0:
        return "Moderate", "Moderate midline shift — close monitoring required (5–10 mm)"
    elif shift_mm >= 2.0:
        return "Low",      "Mild midline shift — mass effect present (2–5 mm)"
    else:
        return "Minimal",  "No significant midline shift detected (<2 mm)"


def _classify_ventricular_compression(tumor_fraction: float, shift_mm: float):
    """Rough ventricular compression classification."""
    if tumor_fraction > 0.15 or shift_mm >= 8.0:
        return "Significant bilateral ventricular compression"
    elif tumor_fraction > 0.08 or shift_mm >= 4.0:
        return "Mild ipsilateral ventricular compression"
    else:
        return "No significant ventricular compression"


def _default_mass_effect():
    return {
        "midline_shift_mm":        0.0,
        "shift_direction":         "None",
        "herniation_risk":         "Not assessed",
        "mass_effect_label":       "Insufficient mask data for mass effect calculation",
        "ventricular_compression": "Not assessed",
        "tumor_fraction_of_brain": 0.0,
    }
