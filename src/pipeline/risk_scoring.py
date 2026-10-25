"""
Stage 5: Risk / Severity Classification Logic
- Composite risk score (0–100) from 6 weighted factors
- Produces: Low / Moderate / High / Critical risk buckets
- NOT a malignancy diagnosis — decision-support flag only
- WHO Grade (I–IV) requires biopsy/histopathology confirmation

Factor Weights:
    Volume / Size                : 25 pts max
    Enhancement pattern          : 20 pts max
    Peritumoral edema            : 15 pts max
    Anatomical location          : 20 pts max
    Shape irregularity           : 10 pts max
    Grade (HGG indicator)        : 10 pts max
    ─────────────────────────────
    Total                        : 100 pts max
"""

import math


# ─── Risk Factor Weights ──────────────────────────────────────────────────────

FACTOR_WEIGHTS = {
    'volume_size':    25,
    'enhancement':    20,
    'edema':          15,
    'location':       20,
    'shape':          10,
    'grade':          10,
}

RISK_BUCKETS = [
    (0,  24,  'Low',      '#22c55e', 'green',    'Low-risk finding. Continued monitoring recommended.'),
    (25, 49,  'Moderate', '#f59e0b', 'yellow',   'Moderate-risk finding. Prompt specialist referral advised.'),
    (50, 74,  'High',     '#f97316', 'orange',   'High-risk finding. Urgent neuro-oncology review required.'),
    (75, 100, 'Critical', '#ef4444', 'red',      'Critical-risk finding. Immediate neurosurgical consultation required.'),
]


def compute_risk_score(
    volume_ml,
    diameter_mm,
    enhancement_data,
    edema_data,
    location_data,
    shape_data,
    grade_label,
    confidence=0.5
):
    """
    Compute weighted composite risk score (0–100).
    
    Args:
        volume_ml: float — tumor volume in mL
        diameter_mm: float — longest diameter in mm
        enhancement_data: dict from radiomics.estimate_enhancement_pattern()
        edema_data: dict from radiomics.estimate_edema_extent()
        location_data: dict from radiomics.infer_anatomical_location()
        shape_data: dict from radiomics.compute_shape_features()
        grade_label: str — "High-Grade (HGG)" | "Low-Grade (LGG)" | "N/A"
        confidence: float — model confidence [0,1]
    
    Returns:
        dict with:
            - total_score: int (0–100)
            - risk_level: str
            - risk_color: str (CSS color)
            - factor_scores: dict of individual factor contributions
            - factor_breakdown: list of dicts for UI rendering
            - clinical_summary: str
    """
    factor_scores = {}

    # ── Factor 1: Volume / Size (25 pts) ──
    size_score = _score_volume_size(volume_ml, diameter_mm)
    factor_scores['volume_size'] = size_score

    # ── Factor 2: Enhancement (20 pts) ──
    enh_score = _score_enhancement(enhancement_data)
    factor_scores['enhancement'] = enh_score

    # ── Factor 3: Edema (15 pts) ──
    edema_score = _score_edema(edema_data)
    factor_scores['edema'] = edema_score

    # ── Factor 4: Location (20 pts) ──
    loc_score = _score_location(location_data)
    factor_scores['location'] = loc_score

    # ── Factor 5: Shape Irregularity (10 pts) ──
    shape_score = _score_shape(shape_data)
    factor_scores['shape'] = shape_score

    # ── Factor 6: Grade (10 pts) ──
    grade_score = _score_grade(grade_label)
    factor_scores['grade'] = grade_score

    # ── Total ──
    raw_total = sum(factor_scores.values())
    total_score = max(0, min(100, int(round(raw_total))))

    # ── Separate Dual Risk Dimensions (Biological vs Surgical/Functional) ──────
    # 1. Biological Risk (Malignancy & WHO CNS5 Grade)
    bio_enh = factor_scores.get('enhancement', 0)
    bio_edema = factor_scores.get('edema', 0)
    bio_shape = factor_scores.get('shape', 0)
    bio_grade = factor_scores.get('grade', 0)
    bio_score = min(100, int(round((bio_enh * 1.5 + bio_edema * 1.5 + bio_shape * 2.0 + bio_grade * 3.0) * 1.2)))

    # 2. Surgical / Functional Risk (Eloquent area, Resectability, Mass Effect)
    surg_vol = factor_scores.get('volume_size', 0)
    surg_loc = factor_scores.get('location', 0)
    surg_score = min(100, int(round((surg_vol * 1.8 + surg_loc * 2.5))))

    # ── Classify Risk Level ──
    risk_level, risk_color, risk_css, clinical_note = _classify_risk(total_score)

    # ── Factor Breakdown for UI ──
    factor_breakdown = _build_factor_breakdown(factor_scores, FACTOR_WEIGHTS)

    # ── Clinical Summary ──
    clinical_summary = _build_clinical_summary(
        total_score, risk_level, volume_ml, diameter_mm,
        enhancement_data, edema_data, location_data, grade_label, clinical_note
    )

    return {
        'total_score': total_score,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_css_class': risk_css,
        'biological_risk': {
            'score': bio_score,
            'level': 'High Biological Malignancy' if bio_score > 60 else ('Moderate' if bio_score > 30 else 'Low')
        },
        'surgical_risk': {
            'score': surg_score,
            'level': 'High Surgical/Functional Risk' if surg_score > 60 else ('Moderate' if surg_score > 30 else 'Lower Risk')
        },
        'factor_scores': factor_scores,
        'factor_breakdown': factor_breakdown,
        'clinical_summary': clinical_summary,
        'clinical_note': clinical_note,
        'confidence_adjusted_score': _confidence_adjust(total_score, confidence)
    }


def classify_risk_level(score):
    """Simple lookup for risk bucket from score."""
    level, color, css, note = _classify_risk(score)
    return level


# ─── Factor Scoring Helpers ───────────────────────────────────────────────────

def _score_volume_size(volume_ml, diameter_mm):
    """25 pts max: larger = higher risk"""
    # Volume scoring
    if volume_ml <= 0 and diameter_mm <= 0:
        return 0
    if volume_ml >= 65:       # >65mL → very large
        vol_pts = 25
    elif volume_ml >= 30:     # 30–65mL → large (>4cm equivalent)
        vol_pts = 20
    elif volume_ml >= 10:     # 10–30mL → moderate
        vol_pts = 13
    elif volume_ml >= 3:      # 3–10mL → small-moderate
        vol_pts = 7
    elif volume_ml > 0:
        vol_pts = 3
    else:
        vol_pts = 0

    # Diameter bonus
    if diameter_mm >= 40:
        diam_bonus = 5
    elif diameter_mm >= 30:
        diam_bonus = 3
    elif diameter_mm >= 20:
        diam_bonus = 1
    else:
        diam_bonus = 0

    return min(25, vol_pts + diam_bonus)


def _score_enhancement(enhancement_data):
    """20 pts max: strong/ring enhancement → higher risk"""
    if not enhancement_data:
        return 0
    severity = enhancement_data.get('severity', 'unknown')
    is_ring = enhancement_data.get('is_ring_enhancement', False)

    base = {
        'high': 16, 'moderate': 10, 'low': 3, 'unknown': 0
    }.get(severity, 0)

    ring_bonus = 4 if is_ring else 0
    return min(20, base + ring_bonus)


def _score_edema(edema_data):
    """15 pts max: extensive edema → mass effect → higher risk"""
    if not edema_data:
        return 0
    severity = edema_data.get('severity', 'unknown')
    return {
        'high': 15, 'moderate': 8, 'low': 2, 'unknown': 0
    }.get(severity, 0)


def _score_location(location_data):
    """20 pts max: eloquent / deep areas → higher surgical risk"""
    if not location_data:
        return 0
    is_eloquent = location_data.get('is_eloquent', False)
    depth = location_data.get('depth_category', 'Cortical')
    surgical_risk = location_data.get('surgical_risk', 'Lower')

    if is_eloquent:
        eloquent_pts = 12
    else:
        eloquent_pts = 0

    depth_pts = {'Deep': 8, 'Subcortical': 5, 'Cortical': 2, 'Unknown': 0}.get(depth, 0)

    return min(20, eloquent_pts + depth_pts)


def _score_shape(shape_data):
    """10 pts max: irregular shape → higher aggression indicator"""
    if not shape_data:
        return 0
    irregularity = shape_data.get('irregularity', 0.0)
    sphericity = shape_data.get('sphericity', 1.0)
    elongation = shape_data.get('elongation', 1.0)

    # High irregularity = lower sphericity = more aggressive
    irr_pts = round(irregularity * 7)        # max 7
    sph_pts = round((1.0 - sphericity) * 2)  # max 2
    elong_pts = 1 if elongation > 2.5 else 0

    return min(10, irr_pts + sph_pts + elong_pts)


def _score_grade(grade_label):
    """10 pts max: HGG indicator → higher risk flag"""
    if grade_label and 'High-Grade' in grade_label:
        return 10
    elif grade_label and 'Low-Grade' in grade_label:
        return 2
    return 5  # Unknown → neutral midpoint


def _classify_risk(score):
    for lo, hi, level, color, css, note in RISK_BUCKETS:
        if lo <= score <= hi:
            return level, color, css, note
    return 'Critical', '#ef4444', 'red', 'Critical-risk finding. Immediate consultation required.'


def _confidence_adjust(score, confidence):
    """Confidence-weighted score for display (acknowledges model uncertainty)."""
    if confidence < 0.6:
        adjusted = score * (0.5 + confidence * 0.5)
    else:
        adjusted = score
    return max(0, min(100, int(round(adjusted))))


# ─── Breakdown & Summary Builders ────────────────────────────────────────────

def _build_factor_breakdown(factor_scores, weights):
    """Build a list of factor dicts for the UI bar chart."""
    labels = {
        'volume_size': 'Volume & Size',
        'enhancement': 'Enhancement Pattern',
        'edema': 'Peritumoral Edema',
        'location': 'Anatomical Location',
        'shape': 'Shape Irregularity',
        'grade': 'Grade Indicator',
    }
    breakdown = []
    for key, max_pts in weights.items():
        score = factor_scores.get(key, 0)
        pct = round((score / max_pts) * 100) if max_pts > 0 else 0
        breakdown.append({
            'factor': labels.get(key, key),
            'score': score,
            'max': max_pts,
            'percent': pct,
            'color': _factor_color(pct)
        })
    return breakdown


def _factor_color(pct):
    if pct >= 75:
        return '#ef4444'
    elif pct >= 50:
        return '#f97316'
    elif pct >= 25:
        return '#f59e0b'
    else:
        return '#22c55e'


def _build_clinical_summary(score, risk_level, volume_ml, diameter_mm,
                             enhancement_data, edema_data, location_data,
                             grade_label, clinical_note):
    """Build a clinical summary paragraph for the structured report."""
    loc = location_data.get('full_location', 'undetermined region') if location_data else 'undetermined region'
    enh = enhancement_data.get('enhancement_pattern', 'undetermined') if enhancement_data else 'undetermined'
    edema_lbl = edema_data.get('edema_label', 'undetermined') if edema_data else 'undetermined'
    is_ring = enhancement_data.get('is_ring_enhancement', False) if enhancement_data else False
    is_eloquent = location_data.get('is_eloquent', False) if location_data else False

    ring_note = " Ring enhancement pattern noted, which may indicate necrotic core characteristic of high-grade lesions." if is_ring else ""
    eloquent_note = f" Location in an eloquent cortical/deep region ({loc}) raises surgical risk consideration." if is_eloquent else ""
    if volume_ml and diameter_mm:  # the app passes 0 for "not measured"
        size_note = f"The lesion measures approximately {volume_ml:.1f}mL (longest diameter {diameter_mm:.1f}mm). "
    else:
        size_note = "Lesion size was not measured (requires a 3D scan with known voxel spacing). "
    grade_note = f" AI grade indicator suggests {grade_label.lower()}." if grade_label and grade_label != 'N/A' else ""

    return (
        f"AI analysis yields a composite risk score of {score}/100 ({risk_level} risk). "
        f"{size_note}"
        f"The lesion is located in the {loc}. "
        f"Enhancement pattern: {enh}.{ring_note} "
        f"Peritumoral edema: {edema_lbl}.{eloquent_note}{grade_note} "
        f"{clinical_note}"
    )
