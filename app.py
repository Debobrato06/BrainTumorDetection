from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import threading
import time
import json
import datetime
import uuid
import os
import sys
from io import BytesIO

# Standard non-torch image processing
try:
    from PIL import Image
except ImportError:
    print("Error: PIL (Pillow) not found.")
    Image = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: Matplotlib not found.")
    plt = None

# Torch-specific imports with fallback
import traceback
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    print(f"SUCCESS: PyTorch {torch.__version__} loaded successfully.")
except Exception as e:
    print("-" * 50)
    print(f"CRITICAL ERROR: PyTorch could not be loaded!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    print("Full Debug Traceback:")
    traceback.print_exc()
    print("-" * 50)
    print("Application will continue in HEURISTIC-ONLY mode.")
    TORCH_AVAILABLE = False
    torch = None
    transforms = None

# Load local modules safely
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
if TORCH_AVAILABLE:
    try:
        # from hybrid_model import HybridTumorModel as BrainTumorClassifier
        from model import BrainTumorClassifier
        from inference import load_volume
        from train import train
    except Exception as e:
        print(f"Failed to load Hybrid ML modules: {e}")
        try:
            from model import BrainTumorClassifier
        except:
            TORCH_AVAILABLE = False
            BrainTumorClassifier = None
        load_volume = train = None
else:
    BrainTumorClassifier = load_volume = train = None

# ─── Load Full Clinical Pipeline Modules ──────────────────────────────────────
try:
    from pipeline.preprocessing import (
        extract_metadata_from_nifti,
        extract_metadata_from_2d,
        normalize_intensity
    )
    from pipeline.segmentation import (
        apply_threshold,
        generate_segmentation_overlay,
        get_mask_stats
    )
    from pipeline.measurement import (
        compute_volume_ml,
        compute_longest_diameter_rano,
        compute_growth_change,
        get_size_category
    )
    from pipeline.radiomics import extract_all_radiomics
    from pipeline.risk_scoring import compute_risk_score
    from pipeline.report_generator import generate_structured_report
    from pipeline.pipeline_validator import validate_pipeline   # ← এই লাইনটা নতুন যোগ করো
    PIPELINE_AVAILABLE = True
    print("SUCCESS: Full clinical pipeline modules loaded.")
except Exception as e:
    print(f"WARNING: Clinical pipeline modules failed to load: {e}")
    PIPELINE_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # largest accepted upload (a 4-sequence NIfTI fits)

# ─── Custom JSON Encoder / Provider for Numpy compatibility ───────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if TORCH_AVAILABLE and torch is not None and isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.detach().cpu().numpy().tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder  # Flask < 2.2 fallback

try:
    from flask.json.provider import DefaultJSONProvider
    class CustomJSONProvider(DefaultJSONProvider):
        def default(self, o):
            if TORCH_AVAILABLE and torch is not None and isinstance(o, torch.Tensor):
                return o.item() if o.numel() == 1 else o.detach().cpu().numpy().tolist()
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    app.json = CustomJSONProvider(app)
except ImportError:
    pass


# ─── Global Settings (configurable via /save_settings) ───────────────────────
SYSTEM_SETTINGS = {
    'segmentation_threshold': 0.5,
    'gpu_acceleration': True,
    'heuristic_fallback': True,
    'inference_precision': 'fp32',
    'anonymize_patient_data': True,
    'audit_logging': False,
    'auto_sync_reports': False,
    'api_endpoint': '',
    'auto_check_updates': True,
}

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'settings.json')

def load_settings():
    """Load persisted settings from disk."""
    global SYSTEM_SETTINGS
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                saved = json.load(f)
                SYSTEM_SETTINGS.update(saved)
        except Exception:
            pass

def save_settings_to_disk():
    """Persist settings to disk."""
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(SYSTEM_SETTINGS, f, indent=2)

load_settings()

# ─── Global Training State ────────────────────────────────────────────────────
training_state = {
    'is_running': False,
    'progress': 0,
    'logs': [],
    'epoch': 0,
    'accuracy': 0.0
}
training_thread = None

# ─── Initialize Model ─────────────────────────────────────────────────────────
model = None
device = None
# Heads that received a training loss, and the softmax temperature fitted on the
# validation set (src/train.py stores both). Older checkpoints only trained the
# detection head and have no fitted temperature.
MODEL_TRAINED_HEADS = {'cls'}
MODEL_TEMPERATURE = 1.0
MODEL_IMG_SIZE = (64, 64, 64)


def read_checkpoint_meta(checkpoint):
    heads, temperature = {'cls'}, 1.0
    if isinstance(checkpoint, dict):
        heads = set(checkpoint.get('trained_heads', heads))
        temperature = float(checkpoint.get('temperature', temperature))
    return heads, temperature

if TORCH_AVAILABLE:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BrainTumorClassifier(in_channels=4, num_classes=2)
        model_path = "best_model.pth"
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # ── DIAGNOSTIC: Check checkpoint compatibility ──
                model_dict = model.state_dict()
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    ckpt_dict = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict):
                    ckpt_dict = checkpoint
                else:
                    ckpt_dict = {}
                    
                matched = {k: v for k, v in ckpt_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                unmatched = [k for k in ckpt_dict if k not in model_dict or model_dict[k].shape != ckpt_dict[k].shape]
                missing = [k for k in model_dict if k not in ckpt_dict]
                
                print(f"[CHECKPOINT CHECK] Total layers in model: {len(model_dict)}")
                print(f"[CHECKPOINT CHECK] Matched: {len(matched)} | Unmatched: {len(unmatched)} | Missing: {len(missing)}")
                if unmatched:
                    print(f"[CHECKPOINT CHECK] UNMATCHED keys (first 5): {unmatched[:5]}")
                if missing:
                    print(f"[CHECKPOINT CHECK] MISSING keys (first 5): {missing[:5]}")
                
                if len(matched) < len(model_dict) * 0.5:
                    print("[CHECKPOINT CHECK] WARNING: Less than 50% layers matched! Model will behave like random weights.")
                
                model.load_state_dict(ckpt_dict, strict=False)
                MODEL_TRAINED_HEADS, MODEL_TEMPERATURE = read_checkpoint_meta(checkpoint)
                if isinstance(checkpoint, dict) and 'img_size' in checkpoint:
                    MODEL_IMG_SIZE = tuple(checkpoint['img_size'])
                print(f"[CHECKPOINT CHECK] trained heads: {sorted(MODEL_TRAINED_HEADS)}, temperature: {MODEL_TEMPERATURE}")

            except Exception as e:
                print(f"Failed to load model: {e}. Using random weights.")
        else:
            print("No model weights found. Using randomly initialized model.")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error initializing model: {e}")
        TORCH_AVAILABLE = False

# ─── 2D tumor segmentation + type model (src/seg2d.py, trained by src/train_seg2d.py) ──
# Gives the tumor outline on JPG/PNG slices, from which size, side, depth and the risk
# estimate are computed (src/pipeline/analysis_2d.py).
SEG2D_MODEL, SEG2D_META, CLASSES_2D = None, {}, None
SEG2D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'seg2d_best.pth')
if TORCH_AVAILABLE and os.path.exists(SEG2D_PATH):
    try:
        from seg2d import load_seg2d, CLASSES as CLASSES_2D
        SEG2D_MODEL, SEG2D_META = load_seg2d(SEG2D_PATH, device)
        print(f"[SEG2D] loaded {SEG2D_PATH} (epoch {SEG2D_META.get('epoch')}, "
              f"val Dice {SEG2D_META.get('val_dice')})")
    except Exception as e:
        print(f"[SEG2D] could not load {SEG2D_PATH}: {e}")
        SEG2D_MODEL = None
else:
    print("[SEG2D] no 2D segmentation model found; 2D uploads get detection only.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_slice_b64(volume_tensor):
    if hasattr(volume_tensor, 'squeeze'):
        try:
            vol_np = volume_tensor.squeeze(0).detach().cpu().numpy()
        except:
            vol_np = volume_tensor
    else:
        vol_np = volume_tensor

    mid_idx = vol_np.shape[1] // 2
    slice_img = vol_np[0, mid_idx, :, :]
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    buf = BytesIO()
    try:
        import matplotlib.pyplot as plt
        plt.imsave(buf, slice_img, cmap='gray')
    except:
        return ""
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def process_2d_image(image_path, target_size=None):
    from PIL import Image
    # ROOT CAUSE OF THE OOM CRASH: this used to hardcode target_size=(64,64,64)
    # regardless of device. On CPU (no MKLDNN), Conv3d falls back to an
    # im2col-based kernel, and the dec1 layer (64 channels @ 64^3 spatial)
    # needs a ~1.8GB unfolded buffer just for that one conv — exactly the
    # allocation size in the crash log. Shrinking the volume on CPU cuts that
    # buffer down by the cube of the size ratio (32^3 instead of 64^3 = ~8x
    # smaller), same fix already applied for training in train.py.
    if target_size is None:
        if TORCH_AVAILABLE and device is not None and device.type == 'cpu':
            target_size = (32, 32, 32)
        else:
            target_size = (64, 64, 64)
    img = Image.open(image_path).convert('L')
    if TORCH_AVAILABLE:
        resize = transforms.Resize((target_size[1], target_size[2]))
        img_tensor = transforms.ToTensor()(resize(img))
        
        # Zero-mean unit-variance normalization (Z-score) matching model training distribution
        mean = img_tensor.mean()
        std = img_tensor.std()
        img_tensor = (img_tensor - mean) / (std + 1e-8)
        
        vol_tensor = img_tensor.unsqueeze(1).repeat(1, target_size[0], 1, 1)
        vol_tensor = vol_tensor.repeat(4, 1, 1, 1)
        return vol_tensor.unsqueeze(0)
    else:
        img = img.resize((target_size[2], target_size[1]))
        img_np = np.array(img).astype(np.float32)
        
        # Zero-mean unit-variance normalization (Z-score)
        mean = np.mean(img_np)
        std = np.std(img_np)
        img_np = (img_np - mean) / (std + 1e-8)
        
        vol = np.zeros((4, target_size[0], target_size[1], target_size[2]), dtype=np.float32)
        for c in range(4):
            for d in range(target_size[0]):
                vol[c, d] = img_np
        return vol


def model_input_2d(image_path):
    """Classifier input for a 2D image, built exactly as in training (dataset.Image2DDataset:
    grayscale, resized, scaled to [0, 1], copied over depth and the 4 channels). Z-scoring
    this input instead made every image look tumor-free to the model (sensitivity 0 on the
    Kaggle test set, see AUDIT_REPORT.md)."""
    from dataset import image_to_volume
    with Image.open(image_path) as img:
        return image_to_volume(img, MODEL_IMG_SIZE).unsqueeze(0)


def load_nifti_for_app(path, img_size=None):
    """NIfTI -> ((1, 4, D, H, W) tensor, grid spacing). Axes are arranged the way the rest of
    the pipeline assumes: D = axial slices (inferior -> superior), H = anterior -> posterior,
    W = patient right -> left (radiological view). Each channel is z-scored on non-zero voxels
    and resized to the model grid; the returned spacing is the mm size of one grid voxel, so
    volumes and diameters measured on the resized mask are in real units.

    Accepts a 4D file with 4 sequences in the last axis (T1, T1ce, T2, FLAIR) or a 3D file
    (copied into all 4 channels, which the model was never trained on)."""
    import nibabel as nib
    from brats_dataset import zscore_nonzero
    img_size = tuple(img_size or MODEL_IMG_SIZE)
    img = nib.load(path)
    if img.ndim == 4 and img.shape[-1] != 4:
        raise ValueError(f"Expected 4 sequences in the last axis of a 4D NIfTI, got shape {img.shape}.")
    if img.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D or 4D NIfTI volume, got shape {img.shape}.")
    img = nib.as_closest_canonical(img)                  # RAS+: axes = (L->R, P->A, I->S)
    data = img.get_fdata(dtype=np.float32)
    data = np.moveaxis(data, -1, 0) if data.ndim == 4 else np.stack([data] * 4)
    if not np.isfinite(data).all() or not np.any(data):
        raise ValueError("NIfTI volume is empty or contains non-finite values.")
    sx, sy, sz = (float(z) for z in img.header.get_zooms()[:3])
    X, Y, Z = data.shape[1:]
    data = np.ascontiguousarray(data.transpose(0, 3, 2, 1)[:, :, ::-1, ::-1])   # (C, Z, Y, X), flipped
    data = np.stack([zscore_nonzero(c) for c in data])
    tensor = torch.nn.functional.interpolate(torch.from_numpy(data).unsqueeze(0), size=img_size,
                                             mode='trilinear', align_corners=False)
    D, H, W = img_size
    spacing = {'slice_thickness': sz * Z / D, 'pixel_spacing_y': sy * Y / H, 'pixel_spacing_x': sx * X / W}
    return tensor, spacing


def localize_2d(image_path, pixel_spacing_mm=None, threshold=0.5):
    """Run the 2D segmentation/type model on an uploaded slice and measure the tumor.
    pixel_spacing_mm (mm per pixel of the uploaded image) is optional; without it the scale is
    estimated from the head width. Returns the analyze_slice() dict plus 'info'."""
    from seg2d import prepare_slice, predict_slice, CLASSES, INPUT_SIZE
    from pipeline.analysis_2d import analyze_slice, risk_estimate_2d, render_overlay
    with Image.open(image_path) as img:
        img.load()
        sl, info = prepare_slice(img)
        prob, cls_p = predict_slice(SEG2D_MODEL, sl, device)
        sp = pixel_spacing_mm * info['orig_px_per_input_px'] if pixel_spacing_mm else None
        a = analyze_slice(sl, prob, cls_p, CLASSES, threshold=threshold, pixel_spacing_mm=sp)
        a['info'] = info
        if a['tumor_found']:
            a['risk'] = risk_estimate_2d(a)
            a['overlay_b64'] = render_overlay(img, a, info, INPUT_SIZE)
    return a


# Mean tumor probability inside the outline below this -> the outline is flagged as uncertain.
# On figshare val+test slices it flagged 15 of 39 outlines that missed the expert mask
# (Dice < 0.1) and 11 of 579 that matched it.
LOW_OUTLINE_CONFIDENCE = 0.8

TYPICAL_GRADE = {
    'Glioma': 'Needs biopsy (gliomas range WHO grade 1-4)',
    'Meningioma': 'Needs biopsy (most meningiomas are WHO grade 1)',
    'Pituitary tumor': 'Needs biopsy (most pituitary adenomas are benign)',
}


def build_2d_report(a, tumor_prob):
    """Response fields for a 2D slice with a segmented tumor (overrides the detection-only
    defaults). Sizes are from one slice with an estimated scale, so they are approximate."""
    t, tconf = a['type_label'], a['type_confidence']
    type_text = f"{t} (model confidence {tconf:.0%})" if tconf >= 0.6 else f"Uncertain (most likely {t}, {tconf:.0%})"
    if a.get('mm_per_px') is None:
        scale_note = 'Size could not be estimated (head outline not found).'
    elif a['scale_source'] == 'provided':
        scale_note = 'Sizes use the pixel spacing you provided.'
    else:
        scale_note = (f"No pixel spacing in a JPG/PNG: the mm scale is estimated from the head width "
                      f"(typical error about ±{a['scale_rel_error']:.0%}).")
    side = a.get('side', 'Not determined')
    depth = a.get('depth_category', 'Unknown')
    band = a.get('vertical_band')
    view = {'axial_or_coronal': 'axial/coronal', 'sagittal': 'sagittal', 'uncertain': 'uncertain view'}.get(a['view'], 'unknown view')
    region = f"{side}, {band} part of the slice" if band else side
    if t == 'Pituitary tumor' and tconf >= 0.6:
        region += ' (sellar / pituitary region)'
    L, P, V = a.get('longest_mm'), a.get('perpendicular_mm'), a.get('volume_est_ml')
    risk = a['risk']
    ring = a.get('ring_like')
    solidity = a.get('solidity', 1.0)
    shape_label = 'Irregular margin' if solidity < 0.8 else ('Lobulated' if solidity < 0.9 else 'Well-circumscribed / round')

    low_conf = (a.get('mean_prob') or 0) < LOW_OUTLINE_CONFIDENCE
    size_txt = (f"about {L:.0f} x {P:.0f} mm (longest x perpendicular), estimated volume ~{V:.1f} mL"
                if L is not None else "size not measurable")
    impression = (
        ("CAUTION: low outline confidence — the outlined region may not be the tumor, so location and "
         "size may be wrong. " if low_conf else "") +
        f"AI finding: a tumor-like lesion is outlined on this {view} slice — {region}. "
        f"Size {size_txt}. Most likely type: {type_text}. "
        f"Depth: {depth.lower()}."
        + (" Rim brighter than the centre (ring-like pattern), which can indicate a necrotic core." if ring else "")
        + (" The lesion crosses the midline." if a.get('crosses_midline') else "")
        + f" Rule-based concern level: {risk['risk_level']} ({risk['total_score']}/100). {risk['clinical_note']} "
        f"{scale_note} One 2D slice cannot show the full 3D extent; volume assumes an ellipsoid. "
        "Research prototype — not a diagnosis; a radiologist must review the scan."
    )
    location = {
        'full_location': region, 'region': region, 'laterality': side,
        'depth_category': depth, 'is_eloquent': False,
        'surgical_risk': risk['surgical_risk']['level'] + ' (estimate)',
    }
    radiomics = {
        'location': location,
        'enhancement': {'enhancement_pattern': 'Ring-like rim' if ring else 'Solid / no ring',
                        'is_ring_enhancement': bool(ring), 'enhancement_ratio': a.get('ring_ratio') or 0.0},
        'edema': {'edema_label': 'Not assessable from one image'},
        'shape': {'shape_label': shape_label, 'irregularity': round(1 - solidity, 3)},
        'texture': {},
        'who_cns5_indicators': {
            'margin_type': shape_label,
            'ring_enhancement': 'Ring-like rim present' if ring else 'Not seen',
            'necrosis_present': bool(ring),
            'dwi_diffusion': 'Needs DWI sequence', 'perfusion_rcbv': 'Needs perfusion MRI',
            'mrs_cho_naa': 'Needs MR spectroscopy',
            'butterfly_glioma': 'Crosses midline' if a.get('crosses_midline') else 'No',
        },
        'vasari': {},
    }
    measurements = {
        'volume_ml': V, 'diameter_mm': L, 'perpendicular_mm': P,
        'rano_criteria': f'{L:.1f} × {P:.1f} mm' if L is not None else None,
        'voxel_count': a.get('area_px'), 'bounding_box': {},
        'area_mm2': a.get('area_mm2'), 'estimated': a['scale_source'] != 'provided',
    }
    risk_out = dict(risk, clinical_summary=impression)
    return {
        'tumor_type': type_text, 'grade': TYPICAL_GRADE.get(t, 'Needs biopsy'),
        'overlay_b64': a['overlay_b64'], 'measurements': measurements, 'radiomics': radiomics,
        'risk': risk_out, 'impression': impression, 'location_text': region,
        'tumor_analysis_2d': {
            'side': side, 'region': region, 'view': a['view'], 'depth': depth,
            'longest_diameter_mm': L, 'perpendicular_diameter_mm': P, 'area_mm2': a.get('area_mm2'),
            'estimated_volume_ml': V, 'mm_per_pixel_input': a.get('mm_per_px'),
            'scale_source': a['scale_source'], 'scale_relative_error': a.get('scale_rel_error'),
            'tumor_type': t, 'type_probabilities': a['type_probs'], 'tumor_probability': tumor_prob,
            'ring_like': bool(ring), 'crosses_midline': a.get('crosses_midline'),
            'risk_level': risk['risk_level'], 'risk_score': risk['total_score'],
            'mean_mask_probability': a.get('mean_prob'), 'outline_low_confidence': low_conf,
        },
    }


def detect_tumor_symmetry_heuristic(volume_tensor):
    """
    Advanced Heuristic Engine: SPATIO-TEXTURE ASYMMETRY & VOXEL INTENSITY ANALYSIS
    """
    try:
        from skimage import measure, morphology, filters, exposure, feature
        from skimage.measure import regionprops

        if hasattr(volume_tensor, 'squeeze'):
            vol = volume_tensor.squeeze(0).detach().cpu().numpy()
        else:
            vol = volume_tensor

        modality_idx = 0
        if vol.shape[0] > 3:
            modality_idx = 3

        img = vol[modality_idx]
        D, H, W = img.shape

        slices_to_check = [D//2, D//4, 3*D//4, D//2 - 10, D//2 + 10]
        votes = []
        all_metrics = []

        for d_idx in slices_to_check:
            d_idx = max(0, min(D - 1, d_idx))
            slice_2d = img[d_idx, :, :].copy()

            # ── Fix: equalize_adapthist requires float images in [0, 1] range.
            # Z-score normalized MRI can have values far outside this range.
            # Rescale to [0, 1] before adaptive histogram equalization.
            s_min, s_max = slice_2d.min(), slice_2d.max()
            if s_max > s_min:
                slice_2d = (slice_2d - s_min) / (s_max - s_min + 1e-8)
            else:
                continue  # flat slice — skip
            slice_2d = np.clip(slice_2d, 0.0, 1.0)

            slice_enhanced = exposure.equalize_adapthist(slice_2d, clip_limit=0.04)
            thresh = filters.threshold_otsu(slice_enhanced)
            brain_mask = slice_enhanced > thresh * 0.2
            if not np.any(brain_mask):
                continue

            mid_w = W // 2
            left = slice_enhanced[:, :mid_w]
            right = np.fliplr(slice_enhanced[:, W-mid_w:])
            w_min = min(left.shape[1], right.shape[1])
            asym_map = np.abs(left[:, :w_min] - right[:, :w_min])
            m_left = brain_mask[:, :mid_w][:, :w_min]
            if not np.any(m_left):
                continue

            asym_val = np.mean(asym_map[m_left])
            asym_peak = np.percentile(asym_map[m_left], 98)
            normalized = (slice_enhanced - np.mean(slice_enhanced[brain_mask])) / (np.std(slice_enhanced[brain_mask]) + 1e-8)
            anomaly_mask = (normalized > 2.2) | (normalized < -2.0)
            anomaly_mask &= brain_mask

            labeled = measure.label(anomaly_mask)
            regions = regionprops(labeled)
            has_suspicious_blob = False
            max_area = 0
            for r in regions:
                max_area = max(max_area, r.area)
                if r.area > 60 and r.solidity > 0.4:
                    has_suspicious_blob = True

            is_lesion = (asym_peak > 0.45) or (has_suspicious_blob and asym_val > 0.08) or (max_area > 300)
            votes.append(is_lesion)
            all_metrics.append({'asym': asym_val, 'peak': asym_peak, 'area': max_area})

        final_is_tumor = sum(votes) >= 2
        avg_asym = np.mean([m['asym'] for m in all_metrics]) if all_metrics else 0
        max_peak = np.max([m['peak'] for m in all_metrics]) if all_metrics else 0
        confidence = 0.5 + (sum(votes) / max(len(slices_to_check), 1)) * 0.45

        return final_is_tumor, confidence, avg_asym, max_peak

    except Exception as e:
        print(f"Heuristic Logic Trace: {e}")
        return False, 0.5, 0.0, 0.0


def tensor_to_numpy_volume(input_tensor):
    """Convert model input tensor to numpy for pipeline processing."""
    try:
        if TORCH_AVAILABLE and hasattr(input_tensor, 'squeeze'):
            return input_tensor.squeeze(0).detach().cpu().numpy()
        elif isinstance(input_tensor, np.ndarray):
            return input_tensor.squeeze(0) if input_tensor.ndim == 5 else input_tensor
    except Exception:
        pass
    return None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400

    # Read previous volume for growth tracking (optional, from form data)
    previous_volume_ml = None
    try:
        prev_vol_str = request.form.get('previous_volume_ml', '')
        if prev_vol_str:
            previous_volume_ml = float(prev_vol_str)
    except Exception:
        pass
    # Optional: mm per pixel of an uploaded 2D image (e.g. from the DICOM header). Without it
    # the 2D size estimate uses the head width as the scale.
    pixel_spacing_mm = None
    try:
        ps = float(request.form.get('pixel_spacing_mm', '') or 0)
        pixel_spacing_mm = ps if 0.05 <= ps <= 5 else None
    except ValueError:
        pass

    if file:
        filename = file.filename.lower()
        is_nifti = filename.endswith('.nii') or filename.endswith('.nii.gz')
        if not (is_nifti or filename.endswith(('.jpg', '.jpeg', '.png'))):
            return jsonify({'error': 'Unsupported file type. Upload a .jpg/.jpeg/.png MRI slice or a .nii/.nii.gz volume.'}), 400
        # One file per request: the server is threaded, so a shared name could mix up uploads
        ext = ('.nii.gz' if filename.endswith('.gz') else '.nii') if is_nifti else os.path.splitext(filename)[1]
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'temp_upload_{uuid.uuid4().hex}{ext}')
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)

        if not is_nifti:
            try:
                with Image.open(temp_path) as probe:
                    probe.verify()
                with Image.open(temp_path) as probe:
                    w_px, h_px = probe.size
                if min(w_px, h_px) < 32:
                    raise ValueError(f'image is only {w_px}x{h_px} pixels')
            except Exception as e:
                os.remove(temp_path)
                reason = str(e) if isinstance(e, ValueError) else 'not a readable image file'
                return jsonify({'error': f'Could not read the image ({reason}). Upload an uncorrupted MRI slice.'}), 400

        try:
            threshold = float(SYSTEM_SETTINGS.get('segmentation_threshold', 0.5))

            # ── Step 1: Extract Scan Metadata (Preprocessing) ──────────────────
            scan_meta = None
            nib_img = None
            if PIPELINE_AVAILABLE:
                if is_nifti:
                    try:
                        import nibabel as nib
                        nib_img = nib.load(temp_path)
                        scan_meta = extract_metadata_from_nifti(nib_img)
                    except Exception as e:
                        print(f"NIfTI metadata extraction failed: {e}")
                        scan_meta = extract_metadata_from_2d(temp_path)
                else:
                    scan_meta = extract_metadata_from_2d(temp_path)
            else:
                scan_meta = {
                    'pixel_spacing_x': 1.0, 'pixel_spacing_y': 1.0,
                    'slice_thickness': 5.0, 'voxel_volume_mm3': 5.0,
                    'image_shape': {}, 'orientation': 'AXIAL',
                    'source': 'default', 'warning': 'Pipeline modules not available.'
                }

            # ── Capture High-Res Display Image ─────────────────────────────────
            display_img_b64 = ""
            if not is_nifti:
                with Image.open(temp_path) as d_img:
                    d_img = d_img.convert('RGB')
                    d_img.thumbnail((1024, 1024))
                    buf = BytesIO()
                    d_img.save(buf, format='JPEG', quality=95)
                    display_img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # ── Step 2: Process for Model (64×64×64 volume) ────────────────────
            # The model, the mask and every measurement share this one grid.
            grid_spacing = None
            if is_nifti:
                if not TORCH_AVAILABLE:
                    return jsonify({'error': 'NIfTI processing requires PyTorch.'}), 500
                try:
                    input_tensor, grid_spacing = load_nifti_for_app(temp_path)
                except Exception as e:
                    return jsonify({'error': f'Could not read the NIfTI file: {e}'}), 400
            elif TORCH_AVAILABLE and model is not None:
                input_tensor = model_input_2d(temp_path)
            else:
                input_tensor = process_2d_image(temp_path)

            if TORCH_AVAILABLE and device is not None:
                if hasattr(input_tensor, 'to'):
                    input_tensor = input_tensor.to(device)
                elif isinstance(input_tensor, np.ndarray):
                    input_tensor = torch.from_numpy(input_tensor).to(device)

            # ── Step 3: Model Inference ─────────────────────────────────────────
            # The segmentation head is used only if the checkpoint says it was trained.
            # best_model.pth trains the detection head only; its untrained decoder marks
            # 75-90% of the volume, and the old fallbacks (lower threshold, intensity
            # heuristic, a disk around the brightest pixel) drew "tumors" on the skull.
            seg_head_trained = 'seg' in MODEL_TRAINED_HEADS
            seg_logits = None
            predicted_class = 0
            raw_confidence = 0.5
            grade_label_raw = "N/A"
            tumor_type_label_raw = "Not determined"
            se_channel_attention = None
            model_inference_ok = True
            det_prob_tumor = None   # detection model's P(tumor)

            if TORCH_AVAILABLE and model is not None:
                with torch.no_grad():
                    try:
                        model_in = input_tensor
                        if seg_head_trained:
                            outputs = model(model_in, return_attention=True)
                        else:
                            # The untrained decoder is not used, so do not run it (it needs GBs of RAM on CPU)
                            cls_logits, grade_logits, type_logits, w = model.forward_heads(model_in)
                            outputs = (cls_logits, None, grade_logits, type_logits,
                                       dict(zip(['T1', 'T1c', 'T2', 'FLAIR'], w[0].tolist())))
                        if len(outputs) == 5:
                            cls_logits, seg_logits, grade_logits, type_logits, se_channel_attention = outputs
                        elif len(outputs) == 4:
                            cls_logits, seg_logits, grade_logits, type_logits = outputs
                        else:
                            cls_logits, seg_logits, grade_logits = outputs[0], outputs[1], outputs[2]
                            type_logits = None

                        # Temperature scaling with the value fitted on the validation set (1.0 if none)
                        probs = torch.softmax(cls_logits / MODEL_TEMPERATURE, dim=1)
                        predicted_class = torch.argmax(probs, dim=1).item()
                        raw_confidence = float(probs[0][predicted_class].item())
                        det_prob_tumor = float(probs[0][1].item())

                        if 'grade' in MODEL_TRAINED_HEADS:
                            grade_probs = torch.softmax(grade_logits, dim=1)
                            grade_idx = torch.argmax(grade_probs, dim=1).item()
                            grade_label_raw = "High-Grade (HGG)" if grade_idx == 1 else "Low-Grade (LGG)"

                        # SE weights describe MRI sequences; a 2D JPG copied into 4 channels has none
                        if not is_nifti:
                            se_channel_attention = None

                        if 'type' in MODEL_TRAINED_HEADS and type_logits is not None:
                            type_types = ["Glioma", "Meningioma", "Metastasis", "Medulloblastoma", "Healthy"]
                            type_probs = torch.softmax(type_logits, dim=1)
                            type_idx = torch.argmax(type_probs, dim=1).item()
                            tumor_type_label_raw = type_types[min(type_idx, len(type_types)-1)]
                    except Exception as e:
                        print(f"Model forward pass fallback: {e}")
                        try:
                            cls_logits = model.forward_heads(model_in)[0]
                            seg_logits = None
                            probs = torch.softmax(cls_logits / MODEL_TEMPERATURE, dim=1)
                            predicted_class = torch.argmax(probs, dim=1).item()
                            raw_confidence = float(probs[0][predicted_class].item())
                            det_prob_tumor = float(probs[0][1].item())
                        except Exception as e2:
                            print(f"Model forward pass fallback also failed, using heuristic only: {e2}")
                            model_inference_ok = False
                            # Leave predicted_class/raw_confidence at their
                            # pre-set defaults; the heuristic check below will
                            # still run and the request completes with a 200
                            # instead of crashing the whole Flask process.

            # ── Heuristic Check (fallback only) ─────────────────────────────────
            heuristic_tumor, h_conf, avg_asym, peak_asym = detect_tumor_symmetry_heuristic(input_tensor)

            # If no model available (or model inference failed entirely), use heuristic confidence score.
            # The heuristic flags most tumor-free images as tumor (AUDIT_REPORT.md), so the
            # validator treats this case as "model unavailable" -> INDETERMINATE.
            model_ok = TORCH_AVAILABLE and model is not None and model_inference_ok
            if not model_ok:
                predicted_class = 1 if heuristic_tumor else 0
                raw_confidence = h_conf

            # ── 2D slices: tumor outline, type and size from the 2D segmentation model ──
            loc2d = None
            if not is_nifti and SEG2D_MODEL is not None:
                try:
                    loc2d = localize_2d(temp_path, pixel_spacing_mm, threshold=threshold)
                except Exception as e:
                    traceback.print_exc()
                    print(f"[SEG2D] analysis failed: {e}")
            if loc2d is not None:
                # Tumor decision from the 2D model (P(tumor) = 1 - P(no tumor)). On the Kaggle
                # Testing split it is more accurate alone (0.998) than the 3D detection model (0.937)
                # or the mean of both (0.986), see runs/seg2d_eval/summary.json.
                p_seg2d = 1.0 - float(loc2d['type_probs'][CLASSES_2D[0]])
                tumor_prob_2d = p_seg2d
                predicted_class = int(tumor_prob_2d >= 0.5)
                raw_confidence = tumor_prob_2d if predicted_class == 1 else 1.0 - tumor_prob_2d
                model_ok = True
                if predicted_class == 1 and not loc2d['tumor_found']:
                    # Tumor predicted but nothing above the mask threshold: retry at a lower one
                    try:
                        loc2d = localize_2d(temp_path, pixel_spacing_mm, threshold=0.3)
                    except Exception as e:
                        print(f"[SEG2D] low-threshold retry failed: {e}")


            # ── OOD Detection ────────────────────────────────────────────────────
            ood_detected = (
                scan_meta.get('source') in ['2d_image', 'default']
                or scan_meta.get('warning') is not None
            )

            # ── Step 4: Segmentation → Binary Mask & BraTS Sub-regions ─────────
            vol_np = tensor_to_numpy_volume(input_tensor)
            binary_mask = None
            prob_map = None
            sub_masks = None
            overlay_b64 = ""
            seg_quality_score = 1.0
            binary_mask_has_voxels = False

            if PIPELINE_AVAILABLE and vol_np is not None:
                if seg_logits is not None and seg_head_trained and model_ok and predicted_class == 1:
                    binary_mask, prob_map, sub_masks = apply_threshold(seg_logits, threshold=threshold)
                else:
                    empty = np.zeros(vol_np.shape[1:], dtype=bool)
                    binary_mask, prob_map = empty, None
                    sub_masks = {'ET': empty, 'TC': empty, 'WT': empty}

                mask_stats_data = get_mask_stats(binary_mask, prob_map, threshold)
                seg_quality_score = mask_stats_data.get('seg_quality_score', 1.0)
                binary_mask_has_voxels = bool(binary_mask.any())

                # ── Seg Quality Filter: reject near-full-coverage masks (noise from
                # untrained segmentation head). If mask covers >50% of volume, it's
                # not a real tumor mask — reset to empty and let heuristic take over.
                if binary_mask_has_voxels:
                    coverage_ratio = binary_mask.sum() / max(binary_mask.size, 1)
                    if coverage_ratio > 0.50:
                        print(f"[SEG QUALITY] Mask coverage={coverage_ratio:.1%} > 50% — treating as noise, resetting.")
                        binary_mask = np.zeros_like(binary_mask)
                        if sub_masks:
                            for k in sub_masks:
                                sub_masks[k] = np.zeros_like(sub_masks[k])
                        binary_mask_has_voxels = False
                        seg_quality_score = 0.0
                        mask_stats_data = {'voxel_count': 0, 'has_tumor': False,
                                           'bounding_box': None, 'centroid': None, 'seg_quality_score': 0.0}

                print(f"[SEG DEBUG] binary_mask_has_voxels={binary_mask_has_voxels}, voxels={binary_mask.sum() if binary_mask is not None else 0}, predicted_class={predicted_class}")
            else:
                mask_stats_data = {
                    'voxel_count': 0, 'has_tumor': False,
                    'bounding_box': None, 'centroid': None, 'seg_quality_score': 1.0
                }

            # ════════════════════════════════════════════════════════════════════
            # ── VALIDATION LAYER ────────────────────────────────────────────────
            # ════════════════════════════════════════════════════════════════════
            vld = validate_pipeline(
                scan_meta=scan_meta,
                predicted_class=predicted_class,
                raw_confidence=raw_confidence,
                model_available=model_ok,
                seg_quality_score=seg_quality_score,
                binary_mask_has_voxels=binary_mask_has_voxels,
                ood_detected=ood_detected,
                grade_label_raw=grade_label_raw
            )

            # Extract gated values from validator
            is_tumor            = vld['is_tumor']
            is_indeterminate    = vld['is_indeterminate']
            final_conf          = vld['final_confidence']
            grade_label         = vld['grade_label']
            gates               = vld['gates']
            insufficient_mask   = vld['insufficient_mask_data']
            status_block        = vld['status_block']
            alert               = vld['alert']
            reliability_reduced = vld['reliability_reduced']

            # ════════════════════════════════════════════════════════════════════
            # ── 2D vs 3D UNIFIED HANDLING ───────────────────────────────────────
            # 3D NIfTI: trust validator's gates fully — real accurate pipeline.
            # 2D JPG/PNG: don't let OOD gate block the UI entirely. Run every
            # stage, but honestly relabel it as reduced-accuracy / limited mode
            # instead of "INDETERMINATE". Numbers still shown, caveat attached.
            # ════════════════════════════════════════════════════════════════════
            if not is_nifti:
                # Downstream stages need a real mask; none exists without a trained segmentation head.
                seg_ok = binary_mask_has_voxels
                gates['segmentation_passed'] = seg_ok
                gates['run_measurements']    = seg_ok
                gates['run_radiomics']       = seg_ok
                gates['run_risk_scoring']    = seg_ok
                gates['run_grading']         = seg_ok
                reliability_reduced = True

                # ── For 2D scans: show the model's decision with a reliability caveat ──
                # (the validator's low-confidence INDETERMINATE state is not used for 2D).
                # Without a working model the only signal is the asymmetry heuristic, which
                # is not reliable, so that case stays INDETERMINATE.
                is_indeterminate = not model_ok
                is_tumor = model_ok and predicted_class == 1
                insufficient_mask = False

                if final_conf is None:
                    final_conf = raw_confidence
                if loc2d is not None:
                    final_conf = raw_confidence   # 2D model probability, ECE in runs/seg2d_eval
                else:
                    final_conf = min(final_conf, 0.65)  # cap: single 2D images give limited evidence

                # Grade only from a trained grading head (grade_label_raw stays "N/A" otherwise)
                if is_tumor and grade_label_raw not in (None, '', 'N/A'):
                    grade_label = grade_label_raw

                det_stage = 'failed' if is_indeterminate else 'completed'
                later_stage = 'completed' if seg_ok else 'not_run'
                status_block = {
                    'modality_verified': False,
                    'detection_stage': det_stage,
                    'segmentation_stage': later_stage,
                    'radiomics_stage': later_stage,
                    'grading_stage': 'completed' if (is_tumor and 'grade' in MODEL_TRAINED_HEADS) else 'not_run',
                    'risk_scoring_stage': later_stage,
                    'overall_reliability': 'reduced',
                    'reason_if_reduced': 'Single 2D image — unverified modality.',
                    'stage_summary': (f'Detection: {det_stage.replace("_", " ").title()} | Segmentation: {later_stage.replace("_", " ").title()} | '
                                      f'Radiomics: {later_stage.replace("_", " ").title()} | Risk Scoring: {later_stage.replace("_", " ").title()}'),
                    'mode': '2d_limited',
                }
                if is_indeterminate:
                    alert = {
                        'level': 'critical',
                        'title': 'INDETERMINATE — Detection Model Unavailable',
                        'message': ('The trained detection model could not be run, so no reliable '
                                    'tumor / no-tumor decision is available for this image.'),
                        'show_results': True
                    }
                else:
                    alert = {
                        'level': 'warning',
                        'title': '2D Limited-Analysis Mode',
                        'message': ('Single 2D image — tumor / no-tumor detection only. '
                                    + ('' if seg_ok else
                                       'This model has no trained segmentation, so tumor location, '
                                       'outline, volume, radiomics and risk score are not computed. ')
                                    + 'A negative result does not rule out a tumor. Research prototype, '
                                    'not for diagnosis.'),
                        'show_results': True
                    }

            # ── Step 5: Generate multi-color segmentation overlay ───────────────
            if is_tumor and binary_mask_has_voxels and gates.get('segmentation_passed') and vol_np is not None:
                overlay_b64 = generate_segmentation_overlay(vol_np, binary_mask, channel=0, sub_masks=sub_masks)


            if is_tumor and 'grade' not in MODEL_TRAINED_HEADS:
                grade_label = "Not available (grading model not trained)"

            # ── NIfTI with a detected lesion but no trained segmentation head ──
            if is_nifti and is_tumor and not seg_head_trained:
                alert = {
                    'level': 'warning',
                    'title': 'Detection Only — Segmentation Not Available',
                    'message': ('This model has no trained segmentation, so tumor location, outline, '
                                'volume, radiomics and risk score are not computed. The detection model '
                                'was trained on 2D images and has not been validated on 3D NIfTI scans.'),
                    'show_results': True
                }

            # ── Step 6: Measurements (gate: segmentation must have passed) ───────
            # Measure on the model grid (the mask's grid), in mm per grid voxel.
            spacing = grid_spacing or scan_meta
            px = spacing.get('pixel_spacing_x', 1.0)
            py = spacing.get('pixel_spacing_y', 1.0)
            st = spacing.get('slice_thickness', 5.0)
            volume_ml = None        # Rule 3: explicit None = "not computed"
            diameter_mm = None
            perpendicular_mm = None
            bounding_box_data = {}

            if gates.get('run_measurements') and PIPELINE_AVAILABLE and binary_mask is not None:
                _vol, _ = compute_volume_ml(binary_mask, px, py, st)
                _diam, _perp, bounding_box_data = compute_longest_diameter_rano(binary_mask, px, py, st)
                # Only accept if non-zero (zero = mask gave no data)
                volume_ml = _vol if _vol > 0 else None
                diameter_mm = _diam if _diam > 0 else None
                perpendicular_mm = _perp if _perp > 0 else None

                # A single 2D image has no slice thickness and no reliable pixel spacing,
                # so volume and diameters cannot be measured from it.
                if not is_nifti:
                    volume_ml = diameter_mm = perpendicular_mm = None

                print(f"[MEAS DEBUG] volume_ml={volume_ml}, diameter_mm={diameter_mm}, is_tumor={is_tumor}, insufficient_mask={insufficient_mask}")

            # ── Step 7: Growth Tracking ─────────────────────────────────────────
            growth_data = None
            if gates.get('run_measurements') and previous_volume_ml is not None and volume_ml is not None:
                if PIPELINE_AVAILABLE:
                    growth_data = compute_growth_change(volume_ml, previous_volume_ml)

            # ── Step 8: Radiomics ───────────────────────────────────────────────
            radiomics_data = None
            if gates.get('run_radiomics') and PIPELINE_AVAILABLE and vol_np is not None and binary_mask is not None:
                try:
                    radiomics_data = extract_all_radiomics(vol_np, binary_mask)
                except Exception as e:
                    print(f"Radiomics extraction failed: {e}")

            # ── Step 9: Risk Scoring ────────────────────────────────────────────
            risk_data = None
            if gates.get('run_risk_scoring') and PIPELINE_AVAILABLE:
                try:
                    risk_data = compute_risk_score(
                        volume_ml=volume_ml or 0,
                        diameter_mm=diameter_mm or 0,
                        enhancement_data=radiomics_data.get('enhancement') if radiomics_data else None,
                        edema_data=radiomics_data.get('edema') if radiomics_data else None,
                        location_data=radiomics_data.get('location') if radiomics_data else None,
                        shape_data=radiomics_data.get('shape') if radiomics_data else None,
                        grade_label=grade_label,
                        confidence=final_conf
                    )
                except Exception as e:
                    print(f"Risk scoring failed: {e}")

            # ── Step 9b: MC-Dropout Uncertainty & Mass Effect Midline Shift ───────
            uncertainty_score = None
            mass_effect_data = None
            if PIPELINE_AVAILABLE and is_tumor and not insufficient_mask and binary_mask_has_voxels:
                try:
                    from pipeline.mass_effect import compute_midline_shift
                    mass_effect_data = compute_midline_shift(vol_np, binary_mask, px, py)
                except Exception as e:
                    print(f"Midline shift calculation failed: {e}")

                if model_ok and seg_head_trained and input_tensor is not None:
                    try:
                        from pipeline.uncertainty import estimate_uncertainty
                        _, _, uncertainty_score = estimate_uncertainty(model, input_tensor, num_samples=15)
                    except Exception as e:
                        print(f"Uncertainty estimation failed: {e}")

            # ── Step 10: Build Measurements dict (null-safe) ────────────────────
            # Rule 3: Use None not 0.0 for uncomputed fields — JS handles rendering
            measurements = {
                'volume_ml': volume_ml,
                'diameter_mm': diameter_mm,
                'perpendicular_mm': perpendicular_mm,
                'rano_criteria': (
                    f'{diameter_mm:.1f} × {perpendicular_mm:.1f} mm'
                    if (diameter_mm is not None and perpendicular_mm is not None)
                    else None
                ),
                'voxel_count': mask_stats_data.get('voxel_count', 0) if gates.get('segmentation_passed') else None,
                'bounding_box': bounding_box_data if gates.get('segmentation_passed') else {},
            }

            # ── Step 11: Structured Report Generation ──────────────────────────
            final_img_b64 = display_img_b64 if display_img_b64 else save_slice_b64(input_tensor)

            detection_result_for_report = {
                'is_tumor': is_tumor,
                'is_indeterminate': is_indeterminate,
                'confidence': final_conf,
                'final_class': predicted_class,
                'threshold': threshold,
                'method': 'SE 3D U-Net detection head' if model_ok else 'Asymmetry heuristic (model unavailable)',
                'detection_state': vld['detection']['detection_state']
            }

            structured_report = None
            if PIPELINE_AVAILABLE:
                try:
                    structured_report = generate_structured_report(
                        patient_meta={'name': 'Anonymous', 'case_id': f'CASE-{uuid.uuid4().hex[:8].upper()}'},
                        scan_meta=scan_meta,
                        detection_result=detection_result_for_report,
                        mask_stats=mask_stats_data,
                        measurements=measurements,
                        growth_data=growth_data,
                        radiomics=radiomics_data,
                        risk_data=risk_data,
                        grade_label=grade_label,
                        confidence=final_conf,
                        overlay_b64=overlay_b64,
                        original_img_b64=final_img_b64,
                        insufficient_mask_data=insufficient_mask,
                        uncertainty_score=uncertainty_score,
                        mass_effect=mass_effect_data
                    )
                except Exception as e:
                    print(f"Report generation failed: {e}")

            # ── Build Response ──────────────────────────────────────────────────
            # Rule 5: Three-state label
            if is_indeterminate:
                result_label = "INDETERMINATE: MANUAL REVIEW REQUIRED"
            elif is_tumor:
                result_label = "POSITIVE: SPACE-OCCUPYING LESION (SOL) IDENTIFIED"
            else:
                result_label = "NEGATIVE: NO RADIOLOGICAL ANOMALY DETECTED"

            # Location (only if radiomics ran successfully)
            tumor_loc = None
            if radiomics_data and radiomics_data.get('location') and gates.get('run_radiomics'):
                tumor_loc = radiomics_data['location'].get('full_location')

            # Rule-based tumor type from radiomics, only when no trained type head exists.
            # These rules have not been validated against pathology.
            if is_tumor and radiomics_data and 'type' not in MODEL_TRAINED_HEADS:
                enh_d = radiomics_data.get('enhancement', {})
                loc_d = radiomics_data.get('location', {})
                shape_d = radiomics_data.get('shape', {})
                if enh_d.get('is_ring_enhancement'):
                    tumor_type_label_raw = "Glioblastoma Multiforme (HGG) (rule-based estimate, not validated)"
                elif loc_d.get('depth_category') == 'Cortical' and shape_d.get('sphericity', 0) > 0.55:
                    tumor_type_label_raw = "Meningioma (rule-based estimate, not validated)"
                elif loc_d.get('depth_category') == 'Deep':
                    tumor_type_label_raw = "Glioma (Deep / Thalamic) (rule-based estimate, not validated)"
                elif shape_d.get('irregularity', 0) > 0.35:
                    tumor_type_label_raw = "High-Grade Glioma (HGG) (rule-based estimate, not validated)"
                else:
                    tumor_type_label_raw = "Low-Grade Glioma (LGG) (rule-based estimate, not validated)"

            # Extract sub-region volumes (ET, TC, WT)
            sub_region_volumes = {'ET': None, 'TC': None, 'WT': None}
            if PIPELINE_AVAILABLE and sub_masks and is_tumor and not insufficient_mask and is_nifti:
                try:
                    px_v, py_v, st_v = px, py, st
                    if 'ET' in sub_masks and sub_masks['ET'].any():
                        sub_region_volumes['ET'] = round(float(compute_volume_ml(sub_masks['ET'], px_v, py_v, st_v)[0]), 2)
                    if 'TC' in sub_masks and sub_masks['TC'].any():
                        sub_region_volumes['TC'] = round(float(compute_volume_ml(sub_masks['TC'], px_v, py_v, st_v)[0]), 2)
                    if 'WT' in sub_masks and sub_masks['WT'].any():
                        sub_region_volumes['WT'] = round(float(compute_volume_ml(sub_masks['WT'], px_v, py_v, st_v)[0]), 2)
                except Exception as e:
                    print(f"Sub-region volume computation failed: {e}")

            response = {
                # Which model outputs come from trained heads; everything else in this
                # response is rule-based and unvalidated (research prototype, not for diagnosis).
                'model_status': {
                    'trained_heads': sorted(MODEL_TRAINED_HEADS),
                    'segmentation_available': seg_head_trained,
                    'input_type': '3D NIfTI' if is_nifti else '2D image',
                    # The detection head was trained and tested on 2D axial MRI images only.
                    'input_validated': (not is_nifti) and model_ok,
                    'disclaimer': 'Research prototype. Not validated for clinical use.',
                },
                # Core detection result
                'label': result_label,
                'confidence': round(final_conf, 4),
                'grade': grade_label,           # Rule 3: sanitized, never stale
                'tumor_type': tumor_type_label_raw if is_tumor else "Healthy / No Anomaly",
                'image_b64': final_img_b64,
                'overlay_b64': overlay_b64,

                # SE-Attention Sequence Weights & BraTS Sub-regions
                'se_channel_attention': se_channel_attention,
                'sub_region_volumes': sub_region_volumes,

                # Rule 2/3/5: Pipeline state flags
                'insufficient_mask_data': insufficient_mask,
                'is_indeterminate': is_indeterminate,
                'ood_detected': ood_detected,
                'reliability_reduced': reliability_reduced,
                'seg_quality_score': round(seg_quality_score, 3),

                # Uncertainty and Mass Effect
                'uncertainty_score': uncertainty_score,
                'midline_shift_mm': mass_effect_data.get('midline_shift_mm') if mass_effect_data else None,
                'mass_effect': mass_effect_data,

                # Rule 8: Top-level fail-loud alert
                'alert': alert,

                # Rule 7: Machine-readable status block
                'pipeline_status': status_block,

                # Scan metadata
                'scan_meta': {
                    'pixel_spacing_x': scan_meta.get('pixel_spacing_x', 1.0),
                    'pixel_spacing_y': scan_meta.get('pixel_spacing_y', 1.0),
                    'slice_thickness': scan_meta.get('slice_thickness', 5.0),
                    'orientation': scan_meta.get('orientation', 'AXIAL'),
                    'source': scan_meta.get('source', 'unknown'),
                    'warning': scan_meta.get('warning'),
                    'image_shape': scan_meta.get('image_shape', {}),
                    'modality_label': vld['modality'].get('modality_label', 'Unknown'),
                },

                # Rule 3: None means "not computed" — JS renders "—" or "N/A"
                'measurements': measurements,

                # Growth tracking
                'growth': growth_data,

                # Radiomics (empty dict if not computed)
                'radiomics': {
                    'location': radiomics_data.get('location', {}) if radiomics_data else {},
                    'enhancement': radiomics_data.get('enhancement', {}) if radiomics_data else {},
                    'edema': radiomics_data.get('edema', {}) if radiomics_data else {},
                    'shape': radiomics_data.get('shape', {}) if radiomics_data else {},
                    'texture': radiomics_data.get('texture', {}) if radiomics_data else {},
                    'who_cns5_indicators': radiomics_data.get('who_cns5_indicators', {}) if radiomics_data else {},
                    'vasari': radiomics_data.get('vasari', {}) if (radiomics_data and 'vasari' in radiomics_data) else {},
                } if (radiomics_data and gates.get('run_radiomics')) else {},

                # Risk scoring (null-safe)
                'risk': {
                    'total_score': risk_data.get('total_score', 0) if risk_data else 0,
                    'risk_level': risk_data.get('risk_level', 'N/A') if risk_data else 'N/A',
                    'risk_color': risk_data.get('risk_color', '#888') if risk_data else '#888',
                    'biological_risk': risk_data.get('biological_risk', {}) if risk_data else {},
                    'surgical_risk': risk_data.get('surgical_risk', {}) if risk_data else {},
                    'factor_breakdown': risk_data.get('factor_breakdown', []) if risk_data else [],
                    'clinical_note': risk_data.get('clinical_note', '') if risk_data else '',
                    'confidence_adjusted_score': risk_data.get('confidence_adjusted_score', 0) if risk_data else 0,
                    'clinical_summary': risk_data.get('clinical_summary', '') if risk_data else '',
                } if (risk_data and not insufficient_mask) else {'total_score': 0, 'risk_level': 'N/A', 'risk_color': '#888', 'factor_breakdown': [], 'clinical_note': '', 'confidence_adjusted_score': 0, 'clinical_summary': ''},

                # Legacy / backward compat fields
                'details': {
                    'tumor_size': f"{volume_ml:.2f} mL" if (is_tumor and not insufficient_mask and volume_ml is not None) else "—",
                    'tumor_loc': tumor_loc if not insufficient_mask else "—",
                    'impression': structured_report.get('clinical_impression', '') if structured_report else '',
                    'peak_asymmetry': float(peak_asym),
                },

                # Full structured report
                'structured_report': structured_report,
            }

            # ── 2D slice with a segmented tumor: fill size / side / type / risk ──
            if not is_nifti and loc2d is not None:
                response['model_status']['segmentation_available'] = True
                response['model_status']['seg2d_model'] = {
                    'path': 'models/seg2d_best.pth', 'epoch': SEG2D_META.get('epoch'),
                    'val_dice': SEG2D_META.get('val_dice')}
                response['tumor_type_probabilities'] = loc2d['type_probs']
                response['detector_probability_tumor'] = det_prob_tumor   # 3D detection model, for reference
                if is_tumor and loc2d['tumor_found']:
                    r2 = build_2d_report(loc2d, round(raw_confidence, 4))
                    response.update({k: r2[k] for k in ('tumor_type', 'grade', 'overlay_b64', 'measurements',
                                                        'radiomics', 'risk', 'tumor_analysis_2d')})
                    response['details']['tumor_size'] = (
                        f"{r2['measurements']['diameter_mm']:.0f} x {r2['measurements']['perpendicular_mm']:.0f} mm (approx.)"
                        if r2['measurements']['diameter_mm'] is not None else '—')
                    response['details']['tumor_loc'] = r2['location_text']
                    response['details']['impression'] = r2['impression']
                    if structured_report is not None:
                        structured_report['clinical_impression'] = r2['impression']
                    mm_px = loc2d.get('mm_per_px')
                    if mm_px:
                        px_orig = round(mm_px / loc2d['info']['orig_px_per_input_px'], 3)
                        response['scan_meta'].update({
                            'pixel_spacing_x': px_orig, 'pixel_spacing_y': px_orig, 'slice_thickness': None,
                            'orientation': {'axial_or_coronal': 'Axial / coronal', 'sagittal': 'Sagittal'}.get(loc2d['view'], 'Uncertain'),
                            'warning': ('2D image: no DICOM metadata. Pixel spacing ' +
                                        ('provided by the user.' if loc2d['scale_source'] == 'provided' else
                                         f"estimated from the head width ({px_orig} mm/pixel, about ±{loc2d['scale_rel_error']:.0%}).")),
                        })
                    response['pipeline_status'] = dict(response['pipeline_status'], **{
                        'segmentation_stage': 'completed', 'radiomics_stage': 'completed',
                        'risk_scoring_stage': 'completed',
                        'stage_summary': 'Detection: Completed | Segmentation: Completed (2D) | '
                                         'Measurements: Estimated | Risk Scoring: Rule-based'})
                    low = r2['tumor_analysis_2d']['outline_low_confidence']
                    response['alert'] = {
                        'level': 'critical' if low else 'warning',
                        'title': (f"Tumor detected — outline uncertain ({r2['tumor_analysis_2d']['side']}?)" if low
                                  else f"Tumor outlined — {r2['tumor_analysis_2d']['side']}"),
                        'message': (('Low outline confidence: the marked region may not be the tumor. ' if low else '') +
                                    'Size, side and risk come from one 2D slice. Without DICOM pixel spacing the mm '
                                    'scale is estimated from the head width, and volume assumes an ellipsoid. '
                                    'Research prototype, not for diagnosis.'),
                        'show_results': True}
                elif not is_tumor and not is_indeterminate:
                    response['alert'] = {
                        'level': 'warning', 'title': 'No tumor detected on this slice',
                        'message': ('The model found no tumor on this single 2D slice. A tumor on other slices '
                                    'is not excluded. Research prototype, not for diagnosis.'),
                        'show_results': True}
                elif is_tumor:
                    response['alert'] = {
                        'level': 'warning', 'title': 'Tumor detected — outline not found',
                        'message': ('The detection models classify this image as containing a tumor, but the '
                                    'segmentation model did not outline a region, so size and side are not given. '
                                    'Research prototype, not for diagnosis.'),
                        'show_results': True}
            return jsonify(response)

        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ─── Storage Routes ───────────────────────────────────────────────────────────

STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'storage')
os.makedirs(STORAGE_DIR, exist_ok=True)


@app.route('/save_report', methods=['POST'])
def save_report():
    try:
        data = request.json
        case_id = data.get('case_id', 'UNKNOWN')
        patient_name = data.get('patient_name', 'Anonymous')

        report_data = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.now().isoformat(),
            'case_id': case_id,
            'patient_name': patient_name,
            'diagnosis': data.get('diagnosis'),
            'confidence': data.get('confidence'),
            'tumor_size': data.get('tumor_size'),
            'tumor_loc': data.get('tumor_loc'),
            'impression': data.get('impression'),
            'risk_level': data.get('risk_level', 'N/A'),
            'risk_score': data.get('risk_score', 0),
            'volume_ml': data.get('volume_ml', 0),
            'diameter_mm': data.get('diameter_mm', 0),
            'grade': data.get('grade', 'N/A'),
            'scan_meta': data.get('scan_meta', {}),
            'radiomics': data.get('radiomics', {}),
            'risk_breakdown': data.get('risk_breakdown', []),
            'growth': data.get('growth'),
            'image_b64': data.get('image_b64', ''),
            'overlay_b64': data.get('overlay_b64', ''),
        }

        filename = f"{case_id}_{int(time.time())}.json"
        filepath = os.path.join(STORAGE_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=4)

        return jsonify({'status': 'success', 'message': 'Report saved successfully', 'report_id': report_data['id']})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_reports', methods=['GET'])
def get_reports():
    reports = []
    if os.path.exists(STORAGE_DIR):
        for filename in os.listdir(STORAGE_DIR):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(STORAGE_DIR, filename), 'r') as f:
                        reports.append(json.load(f))
                except:
                    continue
    reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return jsonify(reports)


@app.route('/delete_report/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    try:
        for filename in os.listdir(STORAGE_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(STORAGE_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        report = json.load(f)
                    if report.get('id') == report_id:
                        os.remove(filepath)
                        return jsonify({'status': 'success', 'message': 'Report deleted'})
                except:
                    continue
        return jsonify({'status': 'error', 'message': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── Settings Routes ──────────────────────────────────────────────────────────

@app.route('/get_settings', methods=['GET'])
def get_settings():
    return jsonify(SYSTEM_SETTINGS)


@app.route('/save_settings', methods=['POST'])
def save_settings_route():
    global SYSTEM_SETTINGS
    try:
        data = request.json
        for key, value in data.items():
            if key in SYSTEM_SETTINGS:
                SYSTEM_SETTINGS[key] = value
        save_settings_to_disk()
        return jsonify({'status': 'success', 'message': 'Settings saved', 'settings': SYSTEM_SETTINGS})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── Training Routes ──────────────────────────────────────────────────────────

class TrainArgs:
    def __init__(self, epochs, batch_size, lr, data_dir, use_synthetic, tumor_path=None, nontumor_path=None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_dir = data_dir
        self.use_synthetic_data = use_synthetic
        self.tumor_path = tumor_path
        self.nontumor_path = nontumor_path
        self.embed_dim = 64
        self.layers = 4
        self.heads = 4


def training_callback(msg, progress=None, accuracy=None):
    global training_state
    if msg:
        training_state['logs'].append(msg)
    if progress is not None:
        training_state['progress'] = progress
    if accuracy is not None:
        training_state['accuracy'] = accuracy
    if len(training_state['logs']) > 50:
        training_state['logs'].pop(0)


def run_training_task(epochs, batch_size, lr, data_source, tumor_path=None, nontumor_path=None):
    global training_state
    training_state['is_running'] = True
    training_state['progress'] = 0
    training_state['logs'] = []

    use_synthetic = (data_source == 'synthetic')
    if not use_synthetic and tumor_path:
        data_dir = os.path.dirname(tumor_path)   # 'data/train/tumor' → 'data/train'
    else:
        data_dir = './data'
    args = TrainArgs(epochs, batch_size, lr, data_dir, use_synthetic, tumor_path, nontumor_path)

    if not TORCH_AVAILABLE:
        training_state['logs'].append("PyTorch is not installed, so training cannot run. Install it with: pip install -r requirements.txt")
        training_state['is_running'] = False
        training_state['progress'] = 0
        return

    # Synthetic (random-noise) data is a smoke test only: write to a separate file and
    # never replace the deployed best_model.pth.
    args.output_path = "smoke_test_model.pth" if use_synthetic else "best_model.pth"

    try:
        training_state['logs'].append("Initializing PyTorch Training Pipeline...")
        result = train(args, progress_callback=training_callback) or {}
        training_state['logs'].append("Training Sequence Completed Successfully.")
        if use_synthetic:
            training_state['logs'].append("Smoke test finished on random data; deployed model left unchanged.")
            return
        # train() writes to a candidate file when best_model.pth exists and replaces it only if
        # the new model is more accurate on validation; otherwise keep serving the current model.
        if result.get('promoted') is False:
            training_state['logs'].append("Deployed model kept (new model was not better or did not finish).")
            return

        global model, MODEL_TRAINED_HEADS, MODEL_TEMPERATURE
        try:
            new_model = BrainTumorClassifier(in_channels=4, num_classes=2)
            checkpoint = torch.load("best_model.pth", map_location=device)

            # ── Handle both nested ({'model_state_dict': ...}) and flat checkpoint formats ──
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                ckpt_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                ckpt_dict = checkpoint
            else:
                ckpt_dict = {}

            model_dict = new_model.state_dict()
            matched = {k: v for k, v in ckpt_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            new_model.load_state_dict(ckpt_dict, strict=False)
            new_model.to(device)
            new_model.eval()
            model = new_model
            MODEL_TRAINED_HEADS, MODEL_TEMPERATURE = read_checkpoint_meta(checkpoint)
            training_state['logs'].append(f"Model reloaded: {len(matched)}/{len(model_dict)} layers matched.")
        except Exception as reload_err:
            training_state['logs'].append(f"Warning: Could not reload model: {reload_err}")
            
    except Exception as e:
        training_state['logs'].append(f"Error during training: {str(e)}")
        print(f"Training Error: {e}")
    finally:
        training_state['is_running'] = False
        training_state['progress'] = 100


@app.route('/start_training', methods=['POST'])
def start_training():
    global training_thread
    if training_state['is_running']:
        return jsonify({'status': 'error', 'message': 'Training already running'})

    data = request.json
    epochs = int(data.get('epochs', 10))
    batch_size = int(data.get('batch_size', 4))
    lr = float(data.get('lr', 0.0003))
    data_source = data.get('data_source', 'synthetic')
    tumor_path = data.get('tumor_path', '')
    nontumor_path = data.get('nontumor_path', '')

    training_thread = threading.Thread(
        target=run_training_task,
        args=(epochs, batch_size, lr, data_source, tumor_path, nontumor_path)
    )
    training_thread.start()

    return jsonify({'status': 'success', 'message': 'Training started'})


@app.route('/training_status')
def training_status():
    return jsonify(training_state)


@app.route('/stop_training', methods=['POST'])
def stop_training():
    training_state['is_running'] = False
    return jsonify({'status': 'success', 'message': 'Stopping training...'})


if __name__ == '__main__':
    print("Starting Flask Interface on http://localhost:5000")
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000, threaded=True)
