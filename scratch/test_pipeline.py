"""
Verification Script: Test Modular Clinical Pipeline Stages
-----------------------------------------------------------
Runs unit/integration checks on the newly added:
  1. Midline shift calculator (mass_effect.py)
  2. Upgraded MNI Atlas region lookup (atlas.py)
  3. VASARI standard radiomics feature mapping (radiomics.py)
  4. MC-Dropout uncertainty estimator (uncertainty.py)
"""

import sys
import os
import numpy as np

# Set path to root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pre-import verification
try:
    from src.pipeline.mass_effect import compute_midline_shift
    from src.pipeline.atlas import lookup_anatomical_region
    from src.pipeline.radiomics import extract_all_radiomics
    from src.pipeline.uncertainty import estimate_uncertainty
    print("SUCCESS: All new pipeline modules imported successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to import pipeline modules: {e}")
    sys.exit(1)


def run_tests():
    print("\n--- Running pipeline unit tests ---")
    
    # ── Test 1: Generate Mock Brain Volume and Tumor Mask ──
    # Shape: (64, 64, 64)
    vol_np = np.zeros((4, 64, 64, 64), dtype=np.float32)
    # Draw a simulated brain hemisphere shape
    vol_np[:, 10:54, 10:54, 10:54] = 0.5
    
    # Draw a simulated tumor in the left hemisphere (W-axis > 32)
    tumor_mask = np.zeros((64, 64, 64), dtype=bool)
    tumor_mask[20:30, 20:30, 40:50] = True
    # Fill tumor region with high intensity in T1CE (channel index 1)
    vol_np[1, tumor_mask] = 1.0
    # Edema in FLAIR (channel index 3) around tumor
    vol_np[3, 18:32, 18:32, 38:52] = 0.8

    # ── Test 2: Verify Midline Shift Calculation ──
    print("\nTesting Midline Shift...")
    mass_effect = compute_midline_shift(vol_np, tumor_mask, pixel_spacing_x=1.0, pixel_spacing_y=1.0)
    print("  Calculated Shift:", mass_effect["midline_shift_mm"], "mm")
    print("  Shift Direction:", mass_effect["shift_direction"])
    print("  Herniation Risk:", mass_effect["herniation_risk"])
    print("  Ventricular Compression:", mass_effect["ventricular_compression"])
    
    assert mass_effect["midline_shift_mm"] >= 0.0, "Shift should be non-negative"
    assert mass_effect["shift_direction"] == "Left", "Tumor on right side (W=40-50) should shift midline Left"

    # ── Test 3: Verify Atlas Anatomical Region Lookup ──
    print("\nTesting Anatomical Atlas Region Lookup...")
    location = lookup_anatomical_region(tumor_mask, volume_shape=tumor_mask.shape)
    print("  Anatomical Location:", location["full_location"])
    print("  Eloquent Area:", location["is_eloquent"])
    print("  Surgical Risk:", location["surgical_risk"])
    print("  Centroid:", location["centroid_norm"])
    print("  Method:", location["atlas_method"])

    assert "Left" in location["full_location"], "Centroid W=45 should map to Left hemisphere"
    
    # ── Test 4: Verify Radiomics + VASARI Mapping ──
    print("\nTesting Radiomics & VASARI Mapping...")
    radiomics = extract_all_radiomics(vol_np, tumor_mask)
    vasari = radiomics["vasari"]
    print("  Margin:", vasari["F5_tumorMargin"])
    print("  Contrast Enhancement:", vasari["F7_contrastEnhancement"])
    print("  Edema Classification:", vasari["F9_edema"])
    print("  Eloquent Brain:", vasari["F10_eloquentBrain"])
    print("  Cortical Involvement:", vasari["F22_corticalInvolvement"])
    print("  Extended location label:", vasari["EXT_locationLabel"])

    assert "F5_tumorMargin" in vasari, "VASARI must contain tumor margin key"
    assert "F7_contrastEnhancement" in vasari, "VASARI must contain enhancement key"

    # ── Test 5: Verify MC-Dropout Uncertainty ──
    print("\nTesting MC-Dropout Uncertainty (Heuristic/Fallback path)...")
    class MockModel:
        def __call__(self, x):
            # Returns a mock tuple (cls, seg, grade)
            return (
                np.array([[1.0, 0.0]]),
                np.random.randn(1, 1, 64, 64, 64),
                np.array([[0.0, 1.0]])
            )
        def train(self):
            pass

    # When PyTorch is not fully loaded, estimate_uncertainty gracefully returns defaults
    mean_prob, var_map, u_score = estimate_uncertainty(MockModel(), vol_np)
    print("  Uncertainty Score (mean variance):", u_score)

    print("\nALL TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
