"""Patch script: Add VASARI mapping to radiomics.py"""
import re

filepath = r"d:\DebobratoResearch\BrainTumorDetection\src\pipeline\radiomics.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Locate the function to replace (from the header comment to end of file)
marker = "# \u2500\u2500\u2500 Main Feature Bundle"
idx = content.find(marker)
if idx == -1:
    print("ERROR: Could not find 'Main Feature Bundle' section")
    exit(1)

# Keep everything before this section, replace from here onward
prefix = content[:idx]

new_tail = '''# \u2500\u2500\u2500 Main Feature Bundle \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def extract_all_radiomics(volume_np, binary_mask):
    """
    Run the full radiomics pipeline and return all features in one dict.
    Safe to call even if individual features fail (graceful degradation).
    Includes VASARI-standard feature mapping for research-grade reporting.
    """
    volume_shape = volume_np.shape
    shape_feats  = compute_shape_features(binary_mask)
    enh_feats    = estimate_enhancement_pattern(volume_np, binary_mask)
    edema_feats  = estimate_edema_extent(binary_mask, volume_np)
    loc_feats    = infer_anatomical_location(binary_mask, volume_shape)
    tex_feats    = compute_texture_features(volume_np, binary_mask)
    return {
        "shape":       shape_feats,
        "enhancement": enh_feats,
        "edema":       edema_feats,
        "location":    loc_feats,
        "texture":     tex_feats,
        "vasari":      to_vasari(shape_feats, enh_feats, edema_feats, loc_feats, tex_feats),
    }


# \u2500\u2500\u2500 VASARI Standard Feature Mapping \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def to_vasari(shape: dict, enhancement: dict, edema: dict, location: dict, texture: dict) -> dict:
    """
    Map internal feature keys to VASARI (Visually AcceSsible Rembrandt Images) standard.
    Reference: Gutman et al. (2013) doi:10.1371/journal.pone.0070966
    """
    sph       = shape.get("sphericity", 0.0)
    irr       = shape.get("irregularity", 0.0)
    enh_ratio = enhancement.get("enhancement_ratio", 0.0)
    ring_enh  = enhancement.get("is_ring_enhancement", False)
    enh_sev   = enhancement.get("severity", "low")
    edema_rat = edema.get("edema_ratio", 0.0)
    is_eloquent = location.get("is_eloquent", False)
    depth     = location.get("depth_category", "Unknown")
    homog     = texture.get("homogeneity", 0.0)

    # F5: Tumor margin
    if sph >= 0.80 and irr < 0.25:
        f5 = "1 - Well-defined / Smooth"
    elif sph >= 0.60:
        f5 = "2 - Moderately irregular"
    else:
        f5 = "3 - Highly irregular / infiltrative"

    # F7: Contrast enhancement quality
    if ring_enh:
        f7 = "4 - Ring enhancement"
    elif enh_ratio >= 2.5:
        f7 = "3 - Avid / marked"
    elif enh_ratio >= 1.5:
        f7 = "2 - Mild / moderate"
    else:
        f7 = "1 - No / minimal"

    # F8: Proportion enhancing
    f8 = ">67%" if enh_sev == "high" else ("34-67%" if enh_sev == "moderate" else "<33%")

    # F9: Edema
    if edema_rat >= 3.0:
        f9 = "4 - Extensive"
    elif edema_rat >= 1.5:
        f9 = "3 - Moderate"
    elif edema_rat >= 0.5:
        f9 = "2 - Mild"
    else:
        f9 = "1 - None"

    # F14: Necrosis
    f14 = "Present" if ring_enh else "Absent"

    # F22: Cortical involvement
    f22 = "Yes" if depth == "Cortical" else ("Partial" if depth == "Subcortical" else "No")

    # Extended texture
    tex_v = ("3 - Highly heterogeneous" if homog < 0.40
             else ("2 - Moderately heterogeneous" if homog < 0.65
                   else "1 - Homogeneous"))

    return {
        "F5_tumorMargin":           f5,
        "F7_contrastEnhancement":   f7,
        "F8_proportionEnhancing":   f8,
        "F9_edema":                 f9,
        "F10_eloquentBrain":        "Yes" if is_eloquent else "No",
        "F14_necrosisPresence":     f14,
        "F16_multifocal":           "Cannot determine",
        "F22_corticalInvolvement":  f22,
        "EXT_textureHeterogeneity": tex_v,
        "EXT_ringEnhancement":      "Yes" if ring_enh else "No",
        "EXT_sphericity":           round(sph, 3),
        "EXT_glcmHomogeneity":      round(homog, 4),
        "EXT_edemaRatio":           round(edema_rat, 2),
        "EXT_enhancementRatio":     round(enh_ratio, 3),
        "EXT_locationLabel":        location.get("full_location", "N/A"),
        "EXT_surgicalRisk":         location.get("surgical_risk", "Unknown"),
    }
'''

new_content = prefix + new_tail

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"SUCCESS: radiomics.py updated. Total lines: {len(new_content.splitlines())}")
