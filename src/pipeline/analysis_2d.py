"""Measurements, location and a rule-based risk estimate for one 2D MRI slice, computed from
the tumor probability map of TumorSegNet2D (src/seg2d.py).

All geometry is on the model's square input grid (INPUT_SIZE px). Millimetres:
  - If the caller knows the pixel spacing (e.g. DICOM-derived), it is used as given.
  - Otherwise the scale is estimated from the width of the head on the image, assuming the
    typical head width measured on the figshare slices (known 0.49 mm pixels). The typical
    error of that estimate is reported with the result (SCALE_REL_ERROR).
Sides use the radiological convention of axial and coronal MRI: the image's left side is the
patient's right side. On a sagittal slice left/right cannot be determined.
"""
import numpy as np
from scipy import ndimage

# Typical width of the head (scalp included) on an axial or coronal slice through the brain.
# Adult head breadth is about 145-155 mm (anthropometric surveys); it is smaller on slices near
# the top of the head, and the width shown on an image also depends on the slice level, so a
# size derived from it is approximate (SCALE_REL_ERROR). The figshare README states 0.49 mm
# pixels, which would make the median head 180 mm wide (calibrate_head_width), so that nominal
# spacing is not used for calibration.
HEAD_WIDTH_MM = 150.0
SCALE_REL_ERROR = 0.15
# Left-right mirror correlation of the head's intensities (view_symmetry). Measured on figshare
# slices: sagittal slices score below ~0.5, axial and coronal slices above ~0.7, both views occur
# in between.
SAGITTAL_MAX = 0.5
UNCERTAIN_VIEW_MAX = 0.7


# ─── Head mask / view ─────────────────────────────────────────────────────────

def head_mask(slice_u8):
    """Largest bright connected region (the head incl. scalp), holes filled."""
    from skimage.filters import threshold_otsu
    img = ndimage.gaussian_filter(slice_u8.astype(np.float32), 2)
    if img.max() <= img.min():
        return np.zeros(slice_u8.shape, bool)
    m = img > 0.5 * threshold_otsu(img)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_closing(m, iterations=3)
    return ndimage.binary_fill_holes(m)


def head_geometry(hm):
    ys, xs = np.nonzero(hm)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # Midline = vertical centre line of the head's bounding box
    return {'x0': int(x0), 'x1': int(x1), 'y0': int(y0), 'y1': int(y1), 'mid_x': (x0 + x1) / 2.0,
            'width_px': int(x1 - x0 + 1), 'height_px': int(y1 - y0 + 1), 'area_px': int(hm.sum())}


def view_symmetry(slice_u8, geo):
    """Correlation between the head region and its mirror image (1 = perfectly symmetric)."""
    import cv2
    crop = slice_u8[geo['y0']:geo['y1'] + 1, geo['x0']:geo['x1'] + 1].astype(np.float32)
    crop = cv2.GaussianBlur(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    a, b = crop.ravel(), crop[:, ::-1].ravel()
    if a.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calibrate_head_width(images, spacings):
    """Median head width in mm and median relative deviation, for slices with known spacing."""
    widths = []
    for img, sp in zip(images, spacings):
        g = head_geometry(head_mask(img))
        if g and view_symmetry(img, g) >= UNCERTAIN_VIEW_MAX:
            widths.append(g['width_px'] * sp)
    w = np.array(widths)
    med = float(np.median(w))
    return med, float(np.median(np.abs(w - med)) / med), len(w)


# ─── Tumor mask / measurements ────────────────────────────────────────────────

def tumor_mask_from_prob(prob, hm=None, threshold=0.5, min_area_px=25):
    """Binary tumor mask: thresholded, inside the head, components smaller than min_area_px or
    than 10% of the largest component removed."""
    m = prob >= threshold
    if hm is not None and hm.any():
        m &= ndimage.binary_dilation(hm, iterations=3)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= max(min_area_px, 0.1 * sizes.max())]
    return np.isin(lab, keep)


def feret_diameters(mask):
    """Longest diameter (px) of the mask, its end points, and the largest extent perpendicular to it."""
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return 0.0, 0.0, None
    try:
        from scipy.spatial import ConvexHull
        pts = pts[ConvexHull(pts).vertices]
    except Exception:
        pass
    pts = pts.astype(np.float64)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    longest = d[i, j]
    axis = (pts[j] - pts[i]) / max(longest, 1e-9)
    normal = np.array([-axis[1], axis[0]])
    proj = pts @ normal
    perp = proj.max() - proj.min()
    # +1 px: a mask one pixel wide is one pixel across, not zero
    return float(longest + 1), float(perp + 1), (pts[i], pts[j])


def ring_pattern(slice_u8, mask):
    """Rim brighter than the core (ring-like enhancement on contrast T1) -> (is_ring, rim/core ratio)."""
    # Bands of the distance to the mask edge, so that a mask drawn a little too large (its
    # outermost pixels are normal tissue) does not decide the rim intensity
    dist = ndimage.distance_transform_edt(mask)
    if dist.max() < 4:
        return False, None
    core = dist > 0.55 * dist.max()
    rim = (dist > 0.2 * dist.max()) & ~core
    if core.sum() < 8 or rim.sum() < 8:
        return False, None
    img = slice_u8.astype(np.float32)
    ratio = float(img[rim].mean() / max(img[core].mean(), 1.0))
    return ratio > 1.25, ratio


def analyze_slice(slice_u8, prob, cls_probs, classes, threshold=0.5, pixel_spacing_mm=None):
    """Everything the report needs from one slice. Returns a dict; 'tumor_found' False when the
    mask is empty."""
    hm = head_mask(slice_u8)
    geo = head_geometry(hm)
    mask = tumor_mask_from_prob(prob, hm, threshold)
    out = {'tumor_found': bool(mask.any()), 'mask': mask, 'head': geo}
    if geo is None:
        out['view'], out['view_symmetry'] = 'unknown', None
    else:
        vs = view_symmetry(slice_u8, geo)
        out['view_symmetry'] = vs
        out['view'] = ('sagittal' if vs < SAGITTAL_MAX else
                       'uncertain' if vs < UNCERTAIN_VIEW_MAX else 'axial_or_coronal')

    # Pixel scale
    if pixel_spacing_mm:
        mm_per_px, scale_source, rel_err = float(pixel_spacing_mm), 'provided', 0.0
    elif geo is not None and out['view'] == 'axial_or_coronal':
        mm_per_px, scale_source, rel_err = HEAD_WIDTH_MM / geo['width_px'], 'estimated_from_head_width', SCALE_REL_ERROR
    elif geo is not None and out['view'] == 'sagittal':
        # Sagittal: front-to-back length ~ 1.25 x breadth (adult head length / breadth ratio)
        mm_per_px, scale_source, rel_err = 1.25 * HEAD_WIDTH_MM / geo['width_px'], 'estimated_from_head_length', 0.2
    elif geo is not None:
        mm_per_px, scale_source, rel_err = HEAD_WIDTH_MM / geo['width_px'], 'estimated_from_head_width', 0.25
    else:
        mm_per_px, scale_source, rel_err = None, 'unknown', None
    out.update({'mm_per_px': mm_per_px, 'scale_source': scale_source, 'scale_rel_error': rel_err})

    ci = int(np.argmax(cls_probs))
    out['type_probs'] = {c: round(float(p), 4) for c, p in zip(classes, cls_probs)}
    out['type_label'] = classes[ci]
    out['type_confidence'] = float(cls_probs[ci])
    if not out['tumor_found']:
        return out

    area_px = int(mask.sum())
    longest_px, perp_px, ends = feret_diameters(mask)
    cy, cx = ndimage.center_of_mass(mask)
    out.update({'area_px': area_px, 'centroid': (float(cy), float(cx)), 'diameter_ends': ends,
                'longest_px': longest_px, 'perp_px': perp_px,
                'mean_prob': float(prob[mask].mean())})
    if mm_per_px:
        L, P = longest_px * mm_per_px, perp_px * mm_per_px
        out['longest_mm'] = round(L, 1)
        out['perpendicular_mm'] = round(P, 1)
        out['area_mm2'] = round(area_px * mm_per_px ** 2, 1)
        # Ellipsoid estimate with the unseen third axis taken equal to the perpendicular diameter
        out['volume_est_ml'] = round(np.pi / 6 * L * P * P / 1000.0, 2)

    # Location
    is_ring, ring_ratio = ring_pattern(slice_u8, mask)
    out['ring_like'] = is_ring
    out['ring_ratio'] = ring_ratio
    solidity = area_px / max(ndimage.binary_fill_holes(_convex_hull(mask)).sum(), 1)
    out['solidity'] = float(solidity)
    if geo is None:
        return out
    half_w = geo['width_px'] / 2.0
    offset = (cx - geo['mid_x']) / half_w          # -1 image left ... +1 image right
    out['lateral_offset'] = float(offset)
    xs = np.nonzero(mask)[1]
    out['crosses_midline'] = bool(xs.min() < geo['mid_x'] - 2 and xs.max() > geo['mid_x'] + 2)
    if out['view'] == 'sagittal':
        out['side'] = 'Not determinable (sagittal slice)'
    elif out['crosses_midline'] and abs(offset) < 0.15:
        out['side'] = 'Midline (both sides)'
    elif abs(offset) < 0.08:
        out['side'] = 'Midline / central'
    else:
        # Radiological convention: image left = patient right
        out['side'] = 'Right side of the brain' if offset < 0 else 'Left side of the brain'
        if out['view'] != 'axial_or_coronal':
            out['side'] += ' (if this is an axial or coronal slice)'
    rel_y = (cy - geo['y0']) / max(geo['height_px'], 1)
    out['vertical_band'] = 'upper' if rel_y < 0.4 else ('lower' if rel_y > 0.65 else 'middle')
    dist = ndimage.distance_transform_edt(hm)
    depth = float(dist[int(round(cy)), int(round(cx))] / max(dist.max(), 1))
    out['depth_ratio'] = depth
    out['depth_category'] = 'Deep' if depth > 0.6 else ('Intermediate' if depth > 0.35 else 'Superficial')
    out['head_fraction'] = float(area_px / max(geo['area_px'], 1))
    return out


def _convex_hull(mask):
    from skimage.morphology import convex_hull_image
    return convex_hull_image(mask)


# ─── Rule-based risk estimate ─────────────────────────────────────────────────

RISK_BUCKETS = [
    (0, 24, 'Low', '#22c55e', 'Low-concern finding on this slice. Specialist review still required.'),
    (25, 49, 'Moderate', '#f59e0b', 'Moderate concern. Prompt specialist (neurology / neurosurgery) review advised.'),
    (50, 74, 'High', '#f97316', 'High concern. Urgent neuro-oncology review advised.'),
    (75, 100, 'Critical', '#ef4444', 'Very high concern. Immediate specialist consultation advised.'),
]

TYPE_POINTS = {'Glioma': 26, 'Meningioma': 10, 'Pituitary tumor': 8}


def risk_estimate_2d(a):
    """Rule-based concern score (0-100) from size, predicted type, location and appearance on one
    slice. Weights reflect general clinical knowledge (gliomas are usually infiltrative and often
    high-grade; most meningiomas and pituitary adenomas are benign; size, deep location and
    midline involvement raise risk). Not validated against outcomes."""
    factors = []
    L = a.get('longest_mm') or 0.0
    size = 5 if L < 10 else 10 if L < 20 else 16 if L < 30 else 22 if L < 40 else 26 if L < 50 else 30
    factors.append(('Size (longest diameter)', size, 30))

    t = a.get('type_label')
    type_pts = TYPE_POINTS.get(t, 15)
    if a.get('type_confidence', 0) < 0.6:
        type_pts = max(type_pts, 15)     # uncertain type -> neutral
    factors.append(('Predicted tumor type', type_pts, 30))

    depth = a.get('depth_category')
    loc = {'Deep': 16, 'Intermediate': 10, 'Superficial': 5}.get(depth, 10)
    if t == 'Pituitary tumor':
        loc = 12        # sellar region: close to the optic chiasm
    if a.get('crosses_midline') and t == 'Glioma':
        loc += 4
    factors.append(('Location / depth', min(loc, 20), 20))

    frac = a.get('head_fraction', 0)
    mass = 10 if frac > 0.08 else 6 if frac > 0.04 else 3 if frac > 0.02 else 1
    factors.append(('Mass relative to head', mass, 10))

    pattern = 0
    if a.get('ring_like'):
        pattern += 6
    if a.get('solidity', 1) < 0.85:
        pattern += 4 if a['solidity'] < 0.75 else 2
    factors.append(('Margin / ring pattern', min(pattern, 10), 10))

    total = int(round(sum(f[1] for f in factors)))
    level, color, note = 'Critical', '#ef4444', RISK_BUCKETS[-1][4]
    for lo, hi, lv, col, nt in RISK_BUCKETS:
        if lo <= total <= hi:
            level, color, note = lv, col, nt
            break
    breakdown = []
    for name, s, mx in factors:
        pct = round(100 * s / mx)
        breakdown.append({'factor': name, 'score': s, 'max': mx, 'percent': pct,
                          'color': '#ef4444' if pct >= 75 else '#f97316' if pct >= 50 else '#f59e0b' if pct >= 25 else '#22c55e'})
    bio = min(100, round(100 * (type_pts + factors[4][1]) / 40))
    surg = min(100, round(100 * (size + factors[2][1] + mass) / 60))
    return {
        'total_score': total, 'risk_level': level, 'risk_color': color, 'clinical_note': note,
        'factor_breakdown': breakdown, 'confidence_adjusted_score': total,
        'biological_risk': {'score': bio, 'level': 'High' if bio > 60 else 'Moderate' if bio > 30 else 'Low'},
        'surgical_risk': {'score': surg, 'level': 'High' if surg > 60 else 'Moderate' if surg > 30 else 'Lower'},
    }


# ─── Overlay ──────────────────────────────────────────────────────────────────

def render_overlay(orig_img, a, info, input_size):
    """Original image (PIL) with the tumor outline, a translucent fill and the longest-diameter
    line drawn on it; returns PNG base64."""
    import base64
    import io
    import cv2
    rgb = np.asarray(orig_img.convert('RGB')).copy()
    h, w = rgb.shape[:2]
    side = max(h, w)
    m = cv2.resize(a['mask'].astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
    m = m[info['pad_top']:info['pad_top'] + h, info['pad_left']:info['pad_left'] + w]
    color = np.array([239, 68, 68], dtype=np.float32)
    sel = m > 0
    rgb[sel] = (0.6 * rgb[sel] + 0.4 * color).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thick = max(1, side // 256)
    cv2.drawContours(rgb, contours, -1, (255, 230, 0), thick)
    if a.get('diameter_ends') is not None:
        s = side / input_size
        (y1, x1), (y2, x2) = a['diameter_ends']
        p1 = (int(x1 * s - info['pad_left']), int(y1 * s - info['pad_top']))
        p2 = (int(x2 * s - info['pad_left']), int(y2 * s - info['pad_top']))
        cv2.line(rgb, p1, p2, (0, 220, 255), thick)
        if a.get('longest_mm'):
            cv2.putText(rgb, f"{a['longest_mm']:.0f} mm", (min(p1[0], p2[0]), max(12, min(p1[1], p2[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, (0, 220, 255), thick, cv2.LINE_AA)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(rgb).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')
