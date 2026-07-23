"""Patch: add uncertainty_score + mass_effect to report_generator.py"""

filepath = r"d:\DebobratoResearch\BrainTumorDetection\src\pipeline\report_generator.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# --- 1. Update function signature to include new params ---
old_sig = '''def generate_structured_report(
    patient_meta,
    scan_meta,
    detection_result,
    mask_stats,
    measurements,
    growth_data,
    radiomics,
    risk_data,
    grade_label,
    confidence,
    overlay_b64="",
    original_img_b64="",
    insufficient_mask_data=False
):'''

new_sig = '''def generate_structured_report(
    patient_meta,
    scan_meta,
    detection_result,
    mask_stats,
    measurements,
    growth_data,
    radiomics,
    risk_data,
    grade_label,
    confidence,
    overlay_b64="",
    original_img_b64="",
    insufficient_mask_data=False,
    uncertainty_score=None,
    mass_effect=None,
):'''

# --- 2. Update pipeline_stages to include new stages ---
old_stages = '''    pipeline_stages = {
        'preprocessing': preprocessing_status,
        'segmentation': segmentation_status,
        'radiomics': radiomics_status,
        'risk_scoring': risk_scoring_status,
        'grading': grading_status
    }'''

new_stages = '''    uncertainty_status = "pass" if uncertainty_score is not None else "skipped"
    mass_effect_status = "pass" if mass_effect is not None else "skipped"

    pipeline_stages = {
        'preprocessing':  preprocessing_status,
        'segmentation':   segmentation_status,
        'radiomics':      radiomics_status,
        'risk_scoring':   risk_scoring_status,
        'grading':        grading_status,
        'uncertainty':    uncertainty_status,
        'mass_effect':    mass_effect_status,
    }'''

# --- 3. Add uncertainty + mass_effect sections to assembled report ---
old_report_tail = '''    # \u2500\u2500 Section 9: Images \u2500\u2500
    images = {
        'original_slice_b64': original_img_b64,
        'segmentation_overlay_b64': overlay_b64,
    }'''

new_report_tail = '''    # \u2500\u2500 Section 9: Uncertainty (MC-Dropout) \u2500\u2500
    uncertainty = None
    if uncertainty_score is not None:
        conf_level = (
            "High Confidence"   if uncertainty_score < 0.02 else
            "Moderate Confidence" if uncertainty_score < 0.06 else
            "Low Confidence \u2014 Manual Review Recommended"
        )
        uncertainty = {
            "score":         round(float(uncertainty_score), 5),
            "confidence_level": conf_level,
            "mc_samples":    20,
            "method":        "Monte Carlo Dropout (variance of segmentation mask)",
        }

    # \u2500\u2500 Section 10: Mass Effect \u2500\u2500
    mass_eff = mass_effect if mass_effect else {
        "midline_shift_mm":        None,
        "herniation_risk":         "Not assessed",
        "ventricular_compression": "Not assessed",
        "mass_effect_label":       "Mass effect analysis not performed",
    }

    # \u2500\u2500 Section 11: Images \u2500\u2500
    images = {
        'original_slice_b64': original_img_b64,
        'segmentation_overlay_b64': overlay_b64,
    }'''

# --- 4. Add new fields to the returned report dict ---
old_return_fields = "        'pipeline_stages': pipeline_stages,\n        'insufficient_mask_data': insufficient_mask_data\n    }"

new_return_fields = """        'pipeline_stages':     pipeline_stages,
        'insufficient_mask_data': insufficient_mask_data,
        'uncertainty':         uncertainty,
        'mass_effect':         mass_eff,
        'uncertainty_score':   round(float(uncertainty_score), 5) if uncertainty_score is not None else None,
        'midline_shift_mm':    mass_eff.get('midline_shift_mm'),
    }"""

# --- 5. Update impression function call to pass new args ---
old_impression = '''    impression = _generate_impression(is_tumor, meas, radiomics_summary, risk, grade_label) if not insufficient_mask_data else "Analysis Incomplete \u2014 Manual Review Required"'''

new_impression = '''    impression = (
        _generate_impression(is_tumor, meas, radiomics_summary, risk, grade_label,
                             uncertainty_score=uncertainty_score, mass_effect=mass_effect)
        if not insufficient_mask_data
        else "Analysis Incomplete \u2014 Manual Review Required"
    )'''

# Apply all patches in sequence
patches = [
    (old_sig,           new_sig,           "function signature"),
    (old_stages,        new_stages,        "pipeline_stages"),
    (old_impression,    new_impression,    "impression call"),
    (old_report_tail,   new_report_tail,   "images section / new sections"),
    (old_return_fields, new_return_fields, "return fields"),
]

for old, new, label in patches:
    if old in content:
        content = content.replace(old, new)
        print(f"  PATCHED: {label}")
    else:
        print(f"  WARNING: not found - {label}")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nSUCCESS: report_generator.py updated. Lines: {len(content.splitlines())}")
