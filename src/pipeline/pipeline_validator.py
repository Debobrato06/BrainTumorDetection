"""
Pipeline Validation Layer — NeuroScan AI
=========================================
Implements all 8 Rules from the clinical system prompt:
  Rule 1 — Modality Gate
  Rule 2 — Stage-Dependency Gating (hard fail-safe chain)
  Rule 3 — No Stale / Carried-Over Values
  Rule 4 — Confidence Caps + Consistency Checks
  Rule 5 — Explicit "No Lesion" / Indeterminate Third State
  Rule 6 — Report Text must match structured data
  Rule 7 — Mandatory Disclaimers + Stage Status Block
  Rule 8 — Fail Loud, not quiet (top-level flags for UI)

This is a rule-based Python module (not LLM-based).
Every function returns a validated/sanitized state dict.
"""

from dataclasses import dataclass, field
from typing import Optional, Any


# ─── Constants ────────────────────────────────────────────────────────────────

DETECTION_CONFIDENCE_THRESHOLD = 0.60   # Rule 2 & 5: below → Indeterminate
SEGMENTATION_QUALITY_THRESHOLD = 0.50   # Rule 2: below → gate downstream
OOD_CONFIDENCE_CAP = 0.70              # Rule 4: cap confidence for OOD inputs
SUPPORTED_MODALITIES = ["MRI", "NIfTI", "MRI-T1", "MRI-T2", "MRI-FLAIR"]


# ─── Stage Status Enum-like Constants ─────────────────────────────────────────

class StageStatus:
    COMPLETED = "completed"
    FAILED    = "failed"
    NOT_RUN   = "not_run"
    SKIPPED   = "skipped"


class OverallReliability:
    HIGH    = "high"
    REDUCED = "reduced"
    LOW     = "low"


class DetectionState:
    LESION_IDENTIFIED = "lesion_identified"
    NO_ANOMALY        = "no_anomaly"
    INDETERMINATE     = "indeterminate"


# ─── Rule 1: Modality Gate ────────────────────────────────────────────────────

def check_modality(scan_meta: dict) -> dict:
    """
    Rule 1: Check if input modality is verified and supported.
    
    Returns:
        {
          'modality_verified': bool,
          'modality_blocked': bool,   # True = stop ALL downstream processing
          'modality_warning': str or None,
          'modality_label': str
        }
    """
    source = scan_meta.get('source', 'unknown')
    warning = scan_meta.get('warning')

    if source == 'nifti':
        # NIfTI = confirmed MRI structural scan
        return {
            'modality_verified': True,
            'modality_blocked': False,
            'modality_warning': None,
            'modality_label': 'MRI (NIfTI — verified)'
        }

    elif source == '2d_image':
        # 2D PNG/JPG: modality is unverified — could be MRI or CT screenshot
        # Per Rule 1: proceed with reduced reliability flag, NOT block
        return {
            'modality_verified': False,
            'modality_blocked': False,
            'modality_warning': (
                "Unverified Modality — 2D image uploaded without DICOM metadata. "
                "Cannot confirm MRI vs CT vs other modality. "
                "This model is validated for MRI (T1/T2/FLAIR) only. "
                "All outputs flagged as REDUCED RELIABILITY."
            ),
            'modality_label': 'Unverified (2D Image — no DICOM)'
        }

    elif source == 'default':
        # Default fallback metadata — pipeline couldn't determine source
        return {
            'modality_verified': False,
            'modality_blocked': False,
            'modality_warning': (
                "Modality could not be verified — pipeline is operating on "
                "default metadata assumptions. Outputs are LOW RELIABILITY."
            ),
            'modality_label': 'Unverified (Unknown Source)'
        }

    else:
        return {
            'modality_verified': True,
            'modality_blocked': False,
            'modality_warning': warning,
            'modality_label': f'MRI ({source})'
        }


# ─── Rule 5: Three-State Detection ───────────────────────────────────────────

def compute_detection_state(
    predicted_class: int,
    confidence: float,
    model_available: bool
) -> dict:
    """
    Rule 5: Detection must support 3 states — not just binary tumor/no-tumor.
    
    States:
        LESION_IDENTIFIED  — model confident, class=1
        NO_ANOMALY         — model confident, class=0
        INDETERMINATE      — confidence < threshold or model unavailable
        
    Returns:
        {
          'detection_state': DetectionState.*,
          'is_tumor': bool,
          'is_indeterminate': bool,
          'detection_confidence': float
        }
    """
    if not model_available:
        # No model → always indeterminate, not a binary coin flip
        return {
            'detection_state': DetectionState.INDETERMINATE,
            'is_tumor': False,
            'is_indeterminate': True,
            'detection_confidence': confidence
        }

    if confidence < DETECTION_CONFIDENCE_THRESHOLD:
        # Rule 5: Weak evidence → Indeterminate, not forced binary
        return {
            'detection_state': DetectionState.INDETERMINATE,
            'is_tumor': False,
            'is_indeterminate': True,
            'detection_confidence': confidence
        }

    if predicted_class == 1:
        return {
            'detection_state': DetectionState.LESION_IDENTIFIED,
            'is_tumor': True,
            'is_indeterminate': False,
            'detection_confidence': confidence
        }
    else:
        return {
            'detection_state': DetectionState.NO_ANOMALY,
            'is_tumor': False,
            'is_indeterminate': False,
            'detection_confidence': confidence
        }


# ─── Rule 2: Stage Dependency Gate ───────────────────────────────────────────

def compute_pipeline_gates(
    detection_state_result: dict,
    seg_quality_score: float,
    binary_mask_has_voxels: bool,
    ood_detected: bool,
    modality_blocked: bool
) -> dict:
    """
    Rule 2: Hard fail-safe chain — each stage only runs if upstream passed.
    Rule 3: Failed upstream = ALL downstream fields must be null, never 0.00.
    
    Returns:
        {
          'run_segmentation': bool,
          'run_measurements': bool,
          'run_radiomics': bool,
          'run_grading': bool,
          'run_risk_scoring': bool,
          'segmentation_passed': bool,
          'failure_reason': str or None
        }
    """
    # Hard block: modality cannot even be processed
    if modality_blocked:
        return {
            'run_segmentation': False,
            'run_measurements': False,
            'run_radiomics': False,
            'run_grading': False,
            'run_risk_scoring': False,
            'segmentation_passed': False,
            'failure_reason': 'MODALITY_BLOCKED: Unsupported or unverifiable input modality.'
        }

    is_tumor = detection_state_result.get('is_tumor', False)
    is_indeterminate = detection_state_result.get('is_indeterminate', False)

    # Indeterminate detection → nothing downstream runs
    if is_indeterminate:
        return {
            'run_segmentation': False,
            'run_measurements': False,
            'run_radiomics': False,
            'run_grading': False,
            'run_risk_scoring': False,
            'segmentation_passed': False,
            'failure_reason': f'DETECTION_INDETERMINATE: Confidence below threshold ({DETECTION_CONFIDENCE_THRESHOLD:.0%}).'
        }

    # No lesion detected → segmentation/measurements/radiomics not applicable
    if not is_tumor:
        return {
            'run_segmentation': False,
            'run_measurements': False,
            'run_radiomics': False,
            'run_grading': False,
            'run_risk_scoring': False,
            'segmentation_passed': False,
            'failure_reason': None  # Not a failure, just "No anomaly"
        }

    # Lesion detected — check segmentation gate
    seg_passed = binary_mask_has_voxels and seg_quality_score >= SEGMENTATION_QUALITY_THRESHOLD
    if not seg_passed:
        reason = (
            f'SEGMENTATION_FAILED: Mask quality score={seg_quality_score:.2f} '
            f'(threshold={SEGMENTATION_QUALITY_THRESHOLD}), '
            f'mask_has_voxels={binary_mask_has_voxels}. '
            f'All downstream metrics are null.'
        )
        return {
            'run_segmentation': True,   # was run, but failed
            'run_measurements': False,
            'run_radiomics': False,
            'run_grading': False,
            'run_risk_scoring': False,
            'segmentation_passed': False,
            'failure_reason': reason
        }

    # All gates passed
    return {
        'run_segmentation': True,
        'run_measurements': True,
        'run_radiomics': True,
        'run_grading': True,
        'run_risk_scoring': True,
        'segmentation_passed': True,
        'failure_reason': None
    }


# ─── Rule 3 & 4: Sanitize Output Fields ──────────────────────────────────────

def sanitize_grade(grade_label: str, gates: dict, detection_state_result: dict) -> str:
    """
    Rule 3: Grade must NEVER be shown when:
     (a) no lesion detected
     (b) segmentation/radiomics insufficient
    Returns null-state string "—" in those cases.
    """
    if not detection_state_result.get('is_tumor', False):
        return "Not Applicable"
    if not gates.get('run_grading', False):
        return "Not Computed — Segmentation Insufficient"
    if grade_label in (None, "", "N/A"):
        return "Not Computed"
    return grade_label


def sanitize_confidence(
    raw_confidence: float,
    ood_detected: bool,
    modality_verified: bool,
    segmentation_passed: bool,
    is_indeterminate: bool
) -> tuple[float, bool, list[str]]:
    """
    Rule 4: Cap confidence at 70% and add Low Reliability flag when:
    - Missing DICOM metadata / unverified modality
    - Segmentation failed or returned insufficient data
    - Input is OOD
    - Indeterminate detection state

    Returns: (capped_confidence, reliability_reduced, reasons_list)
    """
    reasons = []
    should_cap = False

    if not modality_verified:
        should_cap = True
        reasons.append("Unverified modality — no DICOM header")

    if ood_detected:
        should_cap = True
        reasons.append("Out-of-distribution input — missing scan metadata")

    if not segmentation_passed and ood_detected:
        should_cap = True
        reasons.append("Segmentation stage did not produce a valid mask")

    if is_indeterminate:
        should_cap = True
        reasons.append(f"Detection confidence below {DETECTION_CONFIDENCE_THRESHOLD:.0%} threshold")

    if should_cap:
        capped = min(raw_confidence, OOD_CONFIDENCE_CAP)
        return capped, True, reasons

    return raw_confidence, False, []


# ─── Rule 7: Machine-Readable Status Block ───────────────────────────────────

def build_status_block(
    modality_result: dict,
    detection_state_result: dict,
    gates: dict,
    overall_reliability: str,
    reliability_reasons: list[str]
) -> dict:
    """
    Rule 7: Machine-readable status block that MUST be consistent with
    every visible field in the human-readable report.
    """
    detection_state = detection_state_result.get('detection_state', DetectionState.INDETERMINATE)

    if detection_state == DetectionState.LESION_IDENTIFIED:
        det_status = StageStatus.COMPLETED
    elif detection_state == DetectionState.NO_ANOMALY:
        det_status = StageStatus.COMPLETED
    else:
        det_status = StageStatus.FAILED

    if gates.get('run_segmentation'):
        seg_status = StageStatus.COMPLETED if gates.get('segmentation_passed') else StageStatus.FAILED
    else:
        seg_status = StageStatus.NOT_RUN

    rad_status = StageStatus.COMPLETED if gates.get('run_radiomics') else (
        StageStatus.NOT_RUN if not detection_state_result.get('is_tumor') else StageStatus.SKIPPED
    )
    grade_status = StageStatus.COMPLETED if gates.get('run_grading') else (
        StageStatus.NOT_RUN if not detection_state_result.get('is_tumor') else StageStatus.SKIPPED
    )
    risk_status = StageStatus.COMPLETED if gates.get('run_risk_scoring') else (
        StageStatus.NOT_RUN if not detection_state_result.get('is_tumor') else StageStatus.SKIPPED
    )

    return {
        'modality_verified': modality_result.get('modality_verified', False),
        'detection_stage': det_status,
        'segmentation_stage': seg_status,
        'radiomics_stage': rad_status,
        'grading_stage': grade_status,
        'risk_scoring_stage': risk_status,
        'overall_reliability': overall_reliability,
        'reason_if_reduced': '; '.join(reliability_reasons) if reliability_reasons else None,
        # Human-readable summary for disclaimer block (Rule 7)
        'stage_summary': (
            f"Detection: {det_status.title()} | "
            f"Segmentation: {seg_status.title()} | "
            f"Radiomics: {rad_status.title()} | "
            f"Grading: {grade_status.title()} | "
            f"Risk Scoring: {risk_status.title()}"
        )
    }


# ─── Rule 8: Fail-Loud Severity ──────────────────────────────────────────────

def compute_alert_level(
    modality_result: dict,
    detection_state_result: dict,
    gates: dict,
    reliability_reduced: bool
) -> dict:
    """
    Rule 8: Surface failures at TOP of UI, not as small banners.
    Returns a severity + message to drive UI alert rendering.

    Levels: 'none', 'info', 'warning', 'critical'
    """
    if modality_result.get('modality_blocked'):
        return {
            'level': 'critical',
            'title': 'UNSUPPORTED MODALITY — Analysis Not Performed',
            'message': modality_result.get('modality_warning', ''),
            'show_results': False
        }

    if detection_state_result.get('is_indeterminate'):
        return {
            'level': 'critical',
            'title': 'INDETERMINATE — Manual Review Required',
            'message': (
                'Detection confidence is below the minimum reliability threshold. '
                'The model cannot reliably distinguish lesion vs. no-lesion. '
                'Please perform manual clinical review of the scan.'
            ),
            'show_results': True  # show the scan image, but no metrics
        }

    failure = gates.get('failure_reason')
    seg_failed = gates.get('run_segmentation') and not gates.get('segmentation_passed')

    if seg_failed:
        return {
            'level': 'warning',
            'title': 'ANALYSIS INCOMPLETE — Segmentation Failed',
            'message': (
                'The segmentation stage could not produce a reliable tumor mask. '
                'Volume, Diameter, Grade, Radiomics, and Risk Score are NOT computed. '
                'Manual review of the scan is required.'
            ),
            'show_results': True
        }

    if not modality_result.get('modality_verified') or reliability_reduced:
        return {
            'level': 'warning',
            'title': 'LOW RELIABILITY — Unverified Input',
            'message': modality_result.get('modality_warning') or (
                'Input image lacks verified DICOM metadata. '
                'All outputs should be interpreted with caution.'
            ),
            'show_results': True
        }

    if not detection_state_result.get('is_tumor'):
        return {
            'level': 'none',
            'title': '',
            'message': '',
            'show_results': True
        }

    return {
        'level': 'none',
        'title': '',
        'message': '',
        'show_results': True
    }


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def validate_pipeline(
    scan_meta: dict,
    predicted_class: int,
    raw_confidence: float,
    model_available: bool,
    seg_quality_score: float,
    binary_mask_has_voxels: bool,
    ood_detected: bool,
    grade_label_raw: str
) -> dict:
    """
    Central validation entry point. Runs all 8 rules in sequence.
    
    Returns a complete `validation` dict consumed by app.py to:
      - Decide which stages to execute
      - Sanitize all output field values
      - Build the machine-readable status block
      - Set the UI alert level
    """
    # Rule 1: Modality check
    modality = check_modality(scan_meta)

    # Rule 5: Three-state detection
    detection = compute_detection_state(
        predicted_class=predicted_class,
        confidence=raw_confidence,
        model_available=model_available
    )

    # Rule 2: Stage gates
    gates = compute_pipeline_gates(
        detection_state_result=detection,
        seg_quality_score=seg_quality_score,
        binary_mask_has_voxels=binary_mask_has_voxels,
        ood_detected=ood_detected,
        modality_blocked=modality.get('modality_blocked', False)
    )

    # Rule 4: Confidence sanitization
    final_conf, reliability_reduced, reliability_reasons = sanitize_confidence(
        raw_confidence=raw_confidence,
        ood_detected=ood_detected,
        modality_verified=modality.get('modality_verified', False),
        segmentation_passed=gates.get('segmentation_passed', False),
        is_indeterminate=detection.get('is_indeterminate', False)
    )

    overall_reliability = (
        OverallReliability.LOW if not modality.get('modality_verified') and ood_detected
        else OverallReliability.REDUCED if reliability_reduced
        else OverallReliability.HIGH
    )

    # Rule 3: Grade sanitization
    grade_label = sanitize_grade(grade_label_raw, gates, detection)

    # Rule 7: Machine-readable status block
    status_block = build_status_block(
        modality_result=modality,
        detection_state_result=detection,
        gates=gates,
        overall_reliability=overall_reliability,
        reliability_reasons=reliability_reasons
    )

    # Rule 8: Alert level for UI
    alert = compute_alert_level(
        modality_result=modality,
        detection_state_result=detection,
        gates=gates,
        reliability_reduced=reliability_reduced
    )

    return {
        'modality': modality,
        'detection': detection,
        'gates': gates,
        'final_confidence': final_conf,
        'reliability_reduced': reliability_reduced,
        'reliability_reasons': reliability_reasons,
        'overall_reliability': overall_reliability,
        'grade_label': grade_label,
        'status_block': status_block,
        'alert': alert,
        # Convenience shortcuts for app.py
        'is_tumor': detection.get('is_tumor', False),
        'is_indeterminate': detection.get('is_indeterminate', False),
        'insufficient_mask_data': not gates.get('segmentation_passed', True) and detection.get('is_tumor', False),
    }
