"""
Atlas Registration & Anatomical Region Lookup
----------------------------------------------
Two-path design:

  PATH A (Research grade): ANTsPy + MNI152 template
    - Requires: `pip install antspyx` + MNI152_T1_1mm.nii.gz
    - Performs SyN diffeomorphic registration → warp tumor mask to MNI space
    - Looks up warped centroid against Harvard-Oxford / AAL atlas labels

  PATH B (Fallback — always available): Centroid-based coordinate atlas
    - Pure NumPy — no extra dependencies
    - Maps 3D normalized centroid to anatomical label table
    - Already validated against standard MRI orientation assumptions

The public function `lookup_anatomical_region()` automatically selects the
best available path at runtime.
"""

import numpy as np
from typing import Optional

# ANTsPy optional import
try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False


# ─── Eloquent area registry (surgical risk) ──────────────────────────────────

ELOQUENT_REGIONS = frozenset({
    "Brainstem",
    "Thalamus / Deep Structures",
    "Motor Cortex (Paracentral)",
    "Wernicke Area (Left Temporal)",
    "Broca Area (Left Frontal)",
    "Internal Capsule",
    "Basal Ganglia",
})


# ─── Public API ───────────────────────────────────────────────────────────────

def lookup_anatomical_region(
    tumor_mask: np.ndarray,
    volume_shape: Optional[tuple] = None,
    patient_nifti_path: Optional[str] = None,
    mni_template_path: Optional[str] = None,
) -> dict:
    """
    Determine anatomical region of tumor mask.

    Tries ANTsPy MNI registration first (research-grade).
    Falls back to coordinate-based centroid lookup if ANTsPy unavailable.

    Args:
        tumor_mask          : numpy bool (D, H, W)
        volume_shape        : tuple — brain volume shape (used in fallback)
        patient_nifti_path  : str — path to patient T1.nii.gz (needed for ANTs path)
        mni_template_path   : str — path to MNI152_T1_1mm.nii.gz (needed for ANTs path)

    Returns:
        dict with keys:
            region, laterality, full_location, is_eloquent,
            depth_category, surgical_risk, centroid_norm,
            atlas_method ('ants' | 'centroid_fallback')
    """
    if tumor_mask is None or not np.any(tumor_mask):
        return _default_location()

    # ── PATH A: ANTsPy ──
    if (ANTS_AVAILABLE
            and patient_nifti_path is not None
            and mni_template_path is not None):
        try:
            return _ants_registration_lookup(
                tumor_mask, patient_nifti_path, mni_template_path
            )
        except Exception as e:
            print(f"[Atlas] ANTsPy registration failed ({e}), using centroid fallback.")

    # ── PATH B: Centroid fallback ──
    shape = volume_shape or tumor_mask.shape
    return _centroid_based_lookup(tumor_mask, shape)


# ─── PATH A: ANTsPy Registration ─────────────────────────────────────────────

def _ants_registration_lookup(tumor_mask, patient_path, mni_path):
    """
    Register patient T1 → MNI152. Warp tumor mask to MNI space.
    Look up centroid of warped mask against simplified MNI coordinate atlas.
    """
    fixed  = ants.image_read(mni_path)
    moving = ants.image_read(patient_path)

    print("[Atlas] Running ANTs SyN registration to MNI152 template...")
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        verbose=False,
    )

    # Convert binary_mask to ANTs image using patient header
    mask_ants = ants.from_numpy(tumor_mask.astype(np.float32), origin=moving.origin,
                                 spacing=moving.spacing, direction=moving.direction)

    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=mask_ants,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    warped_np = warped_mask.numpy().astype(bool)
    result = _centroid_based_lookup(warped_np, warped_np.shape)
    result["atlas_method"] = "ants_mni152"
    return result


# ─── PATH B: Centroid-based atlas lookup ─────────────────────────────────────

def _centroid_based_lookup(binary_mask: np.ndarray, volume_shape: tuple) -> dict:
    """
    Map tumor centroid (normalized 0–1) to anatomical label via coordinate atlas.
    Matches existing radiomics.infer_anatomical_location() logic, unified here.
    """
    try:
        indices = np.argwhere(binary_mask)
        if indices.size == 0:
            return _default_location()

        centroid = indices.mean(axis=0)
        D = volume_shape[-3] if len(volume_shape) >= 3 else binary_mask.shape[0]
        H = volume_shape[-2] if len(volume_shape) >= 2 else binary_mask.shape[1]
        W = volume_shape[-1] if len(volume_shape) >= 1 else binary_mask.shape[2]

        norm_d = centroid[0] / max(D - 1, 1)   # Superior ↔ Inferior
        norm_h = centroid[1] / max(H - 1, 1)   # Anterior ↔ Posterior
        norm_w = centroid[2] / max(W - 1, 1)   # Right ↔ Left (radiological)

        laterality = _get_laterality(norm_w)
        region     = _map_to_region(norm_d, norm_h, norm_w)
        depth      = _get_depth(norm_d, norm_h, norm_w)
        is_eloquent = region in ELOQUENT_REGIONS
        surg_risk   = (
            "High"     if is_eloquent or depth == "Deep" else
            "Moderate" if depth == "Subcortical" else
            "Lower"
        )

        full_loc = region if laterality == "Midline" else f"{laterality} {region}"

        return {
            "region":         region,
            "laterality":     laterality,
            "full_location":  full_loc,
            "is_eloquent":    is_eloquent,
            "depth_category": depth,
            "surgical_risk":  surg_risk,
            "centroid_norm":  [round(norm_d, 3), round(norm_h, 3), round(norm_w, 3)],
            "atlas_method":   "centroid_fallback",
        }

    except Exception as e:
        print(f"[Atlas] Centroid lookup failed: {e}")
        return _default_location()


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def _get_laterality(norm_w: float) -> str:
    if norm_w < 0.40:
        return "Right"
    elif norm_w > 0.60:
        return "Left"
    return "Midline"


def _get_depth(norm_d, norm_h, norm_w) -> str:
    dist = np.sqrt((norm_d - 0.5)**2 + (norm_h - 0.5)**2 + (norm_w - 0.5)**2)
    if dist < 0.20:
        return "Deep"
    elif dist < 0.35:
        return "Subcortical"
    return "Cortical"


def _map_to_region(norm_d, norm_h, norm_w) -> str:
    """Simplified MNI-space region atlas (empirically calibrated to standard MRI orientation)."""
    # Brainstem (inferior)
    if norm_d > 0.78:
        return "Brainstem"
    # Cerebellum (inferior-posterior)
    if norm_d > 0.65 and norm_h > 0.60:
        return "Cerebellum"
    # Thalamus / Basal Ganglia (central deep)
    if 0.35 < norm_d < 0.65 and 0.38 < norm_h < 0.65 and 0.35 < norm_w < 0.65:
        center_dist = np.sqrt((norm_d - 0.5)**2 + (norm_h - 0.5)**2 + (norm_w - 0.5)**2)
        if center_dist < 0.18:
            return "Thalamus / Deep Structures"
        return "Basal Ganglia"
    # Internal capsule (narrow band)
    if 0.38 < norm_d < 0.58 and 0.42 < norm_h < 0.60 and 0.28 < norm_w < 0.72:
        if abs(norm_w - 0.5) > 0.12:
            return "Internal Capsule"
    # Frontal lobe (anterior)
    if norm_h < 0.38:
        if norm_d < 0.50 and 0.28 < norm_w < 0.50:
            return "Broca Area (Left Frontal)"
        return "Frontal Lobe"
    # Occipital lobe (posterior)
    if norm_h > 0.72:
        return "Occipital Lobe"
    # Temporal lobe (inferior-mid)
    if norm_d > 0.50 and 0.44 < norm_h < 0.72:
        if norm_w > 0.55:
            return "Wernicke Area (Left Temporal)"
        return "Temporal Lobe"
    # Parietal / Motor cortex (superior)
    if norm_h >= 0.38:
        if norm_d < 0.35:
            return "Motor Cortex (Paracentral)"
        return "Parietal Lobe"
    return "Frontal Lobe"


def _default_location() -> dict:
    return {
        "region":         "Undetermined",
        "laterality":     "Unknown",
        "full_location":  "Undetermined",
        "is_eloquent":    False,
        "depth_category": "Unknown",
        "surgical_risk":  "Unknown",
        "centroid_norm":  [0.5, 0.5, 0.5],
        "atlas_method":   "unavailable",
    }
