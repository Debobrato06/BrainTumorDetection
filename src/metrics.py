"""
Evaluation metrics, computed per case. Conventions follow the BraTS challenge:
- Dice = 1 when prediction and reference are both empty.
- HD95 = 0 when both are empty; 373.13 mm (image diagonal) when exactly one is empty.
"""
import numpy as np
from scipy import ndimage

REGIONS = ("wt", "tc", "et")
HD95_EMPTY_PENALTY = 373.13


def dice(pred, ref):
    pred, ref = pred.astype(bool), ref.astype(bool)
    denom = pred.sum() + ref.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, ref).sum() / denom)


def _surface(mask):
    return mask ^ ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


def hd95(pred, ref, spacing=(1.0, 1.0, 1.0)):
    pred, ref = pred.astype(bool), ref.astype(bool)
    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return HD95_EMPTY_PENALTY
    sp, sr = _surface(pred), _surface(ref)
    dt_ref = ndimage.distance_transform_edt(~sr, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_ref[sp], dt_pred[sr]])
    return float(np.percentile(d, 95))


def sensitivity_specificity(pred, ref):
    pred, ref = pred.astype(bool), ref.astype(bool)
    tp = np.logical_and(pred, ref).sum()
    tn = np.logical_and(~pred, ~ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    sens = float(tp / (tp + fn)) if (tp + fn) else 1.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 1.0
    return sens, spec


def region_metrics(pred_regions, ref_regions, spacing=(1.0, 1.0, 1.0)):
    """pred_regions / ref_regions: (3, X, Y, Z) binary in WT, TC, ET order."""
    out = {}
    for i, r in enumerate(REGIONS):
        out[f"dice_{r}"] = dice(pred_regions[i], ref_regions[i])
        out[f"hd95_{r}"] = hd95(pred_regions[i], ref_regions[i], spacing)
        out[f"sens_{r}"], out[f"spec_{r}"] = sensitivity_specificity(pred_regions[i], ref_regions[i])
    return out


def expected_calibration_error(probs, labels, n_bins=15):
    """Binary ECE: sum over equal-width bins of P(prob in bin) * |mean prob - observed positive rate|."""
    centers, freq, counts = reliability_curve(probs, labels, n_bins)
    if counts.sum() == 0:
        return float("nan")
    return float(np.sum(counts / counts.sum() * np.abs(centers - freq)))


def reliability_curve(probs, labels, n_bins=10):
    """Returns (bin centers, observed positive frequency, counts) for a reliability diagram."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    centers, freq, counts = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            centers.append(probs[sel].mean())
            freq.append(labels[sel].mean())
            counts.append(int(sel.sum()))
    return np.array(centers), np.array(freq), np.array(counts)
