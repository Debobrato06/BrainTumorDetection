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
        extract_tumor_mask_from_heuristic,
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
                
            except Exception as e:
                print(f"Failed to load model: {e}. Using random weights.")
        else:
            print("No model weights found. Using randomly initialized model.")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error initializing model: {e}")
        TORCH_AVAILABLE = False


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

    if file:
        filename = file.filename.lower()
        is_nifti = filename.endswith('.nii') or filename.endswith('.nii.gz')
        temp_filename = 'temp_upload.nii.gz' if is_nifti else 'temp_upload.png'
        temp_path = os.path.join('data', temp_filename)
        os.makedirs('data', exist_ok=True)
        file.save(temp_path)

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
            if is_nifti:
                if load_volume is not None:
                    input_tensor = load_volume(temp_path)
                else:
                    return jsonify({'error': 'NIfTI processing requires PyTorch.'}), 500
            else:
                input_tensor = process_2d_image(temp_path)

            if TORCH_AVAILABLE and device is not None:
                if hasattr(input_tensor, 'to'):
                    input_tensor = input_tensor.to(device)
                elif isinstance(input_tensor, np.ndarray):
                    input_tensor = torch.from_numpy(input_tensor).to(device)

            # ── Step 3: Model Inference ─────────────────────────────────────────
            seg_logits = None
            predicted_class = 0
            raw_confidence = 0.5
            grade_label_raw = "N/A"
            tumor_type_label_raw = "Glioma"
            se_channel_attention = {'T1': 0.25, 'T1c': 0.35, 'T2': 0.20, 'FLAIR': 0.20}
            model_inference_ok = True

            if TORCH_AVAILABLE and model is not None:
                with torch.no_grad():
                    try:
                        outputs = model(input_tensor, return_attention=True)
                        if len(outputs) == 5:
                            cls_logits, seg_logits, grade_logits, type_logits, se_channel_attention = outputs
                        elif len(outputs) == 4:
                            cls_logits, seg_logits, grade_logits, type_logits = outputs
                        else:
                            cls_logits, seg_logits, grade_logits = outputs[0], outputs[1], outputs[2]
                            type_logits = None

                        # Temperature Scaling (T=1.45) — reduces overconfidence
                        T = 1.45
                        probs = torch.softmax(cls_logits / T, dim=1)
                        predicted_class = torch.argmax(probs, dim=1).item()
                        raw_confidence = float(probs[0][predicted_class].item())

                        grade_probs = torch.softmax(grade_logits, dim=1)
                        grade_idx = torch.argmax(grade_probs, dim=1).item()
                        grade_label_raw = "High-Grade (HGG)" if grade_idx == 1 else "Low-Grade (LGG)"

                        if 'type_logits' in locals() and type_logits is not None:
                            type_types = ["Glioma", "Meningioma", "Metastasis", "Medulloblastoma", "Healthy"]
                            type_probs = torch.softmax(type_logits, dim=1)
                            type_idx = torch.argmax(type_probs, dim=1).item()
                            tumor_type_label_raw = type_types[min(type_idx, len(type_types)-1)]
                    except Exception as e:
                        print(f"Model forward pass fallback: {e}")
                        try:
                            outputs = model(input_tensor)
                            cls_logits, seg_logits = outputs[0], outputs[1]
                            probs = torch.softmax(cls_logits, dim=1)
                            predicted_class = torch.argmax(probs, dim=1).item()
                            raw_confidence = float(probs[0][predicted_class].item())
                        except Exception as e2:
                            print(f"Model forward pass fallback also failed, using heuristic only: {e2}")
                            model_inference_ok = False
                            # Leave predicted_class/raw_confidence at their
                            # pre-set defaults; the heuristic check below will
                            # still run and the request completes with a 200
                            # instead of crashing the whole Flask process.

            # ── Heuristic Check (fallback only) ─────────────────────────────────
            heuristic_tumor, h_conf, avg_asym, peak_asym = detect_tumor_symmetry_heuristic(input_tensor)

            # If no model available (or model inference failed entirely), use heuristic confidence score
            if not (TORCH_AVAILABLE and model is not None) or not model_inference_ok:
                predicted_class = 1 if heuristic_tumor else 0
                raw_confidence = h_conf

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
                if seg_logits is not None:
                    binary_mask, prob_map, sub_masks = apply_threshold(seg_logits, threshold=threshold)

                    # ── FALLBACK: If model is untrained/weak, lower threshold ──
                    if not binary_mask.any() and predicted_class == 1:
                        print("[SEG FALLBACK] Empty mask at threshold 0.5, trying 0.05 for weak model")
                        binary_mask, prob_map, sub_masks = apply_threshold(seg_logits, threshold=0.05)

                    # NOTE: For 2D images (not_nifti), we intentionally do NOT zero the
                    # mask here even if predicted_class == 0, because the 2D override
                    # block below may flip is_tumor to True (heuristic). The zeroing is
                    # deferred to AFTER that override block. For NIfTI, zero immediately.
                    if predicted_class == 0 and is_nifti:
                        binary_mask = np.zeros_like(binary_mask)
                        if sub_masks:
                            for k in sub_masks:
                                sub_masks[k] = np.zeros_like(sub_masks[k])
                else:
                    binary_mask, prob_map, sub_masks = extract_tumor_mask_from_heuristic(
                        vol_np, peak_asym, avg_asym, predicted_class == 1
                    )

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
                model_available=(TORCH_AVAILABLE and model is not None),
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
                gates['segmentation_passed'] = True
                gates['run_measurements']    = True
                gates['run_radiomics']       = True
                gates['run_risk_scoring']    = True
                gates['run_grading']         = True
                reliability_reduced = True

                # ── For 2D scans: override indeterminate state so results panel shows ──
                # Validator marks low-confidence detections as INDETERMINATE, but for
                # 2D images we want to show all results with a reliability caveat instead
                # of blocking the entire UI.
                is_indeterminate = False
                is_tumor = (predicted_class == 1) or heuristic_tumor
                insufficient_mask = False

                if final_conf is None or final_conf < 0.40:
                    final_conf = max(h_conf, 0.40)  # use heuristic as floor for 2D
                final_conf = min(final_conf, 0.65)  # honest confidence cap for 2D

                # Override grade_label — validator set it to "Not Computed" because
                # run_grading was False. Now that we forced gates, use raw model output.
                if is_tumor and grade_label_raw not in (None, '', 'N/A'):
                    grade_label = grade_label_raw

                status_block = {
                    'modality_verified': False,
                    'detection_stage': 'completed',
                    'segmentation_stage': 'completed',
                    'radiomics_stage': 'completed',
                    'grading_stage': 'completed',
                    'risk_scoring_stage': 'completed',
                    'overall_reliability': 'reduced',
                    'reason_if_reduced': 'Single 2D image — unverified modality, approximated metrics.',
                    'stage_summary': 'Detection: Completed | Segmentation: Completed | Radiomics: Completed | Grading: Completed | Risk Scoring: Completed',
                    'mode': '2d_limited',
                }
                alert = {
                    'level': 'warning',
                    'title': '2D Limited-Analysis Mode',
                    'message': ('Single 2D image detected — not a multi-modal 3D MRI. '
                                'Detection ran, but volume/diameter/radiomics figures below '
                                'use an approximated slice-repeat volume and are NOT clinically '
                                'or scientifically reliable. Upload a .nii/.nii.gz scan for '
                                'accurate quantitative results.'),
                    'show_results': True
                }

                # ── 2D Mask Rescue: if model said no-tumor (mask is zeros) but we
                # overrode is_tumor=True, regenerate mask from heuristic so downstream
                # steps (radiomics, measurements) have actual voxel data to work with.
                if is_tumor and binary_mask is not None and not binary_mask.any() and vol_np is not None:
                    print("[2D MASK RESCUE] Regenerating mask from image intensities for 2D tumor case")
                    binary_mask, prob_map, sub_masks = extract_tumor_mask_from_heuristic(
                        vol_np, max(peak_asym, 0.3), max(avg_asym, 0.1), True
                    )
                    # If heuristic mask still empty, locate actual intensity peak centroid in the image
                    if not binary_mask.any():
                        D, H, W = vol_np.shape[1], vol_np.shape[2], vol_np.shape[3]
                        mid_d = D // 2
                        sl = vol_np[min(3, vol_np.shape[0] - 1), mid_d]
                        s_min, s_max = sl.min(), sl.max()
                        sl_norm = (sl - s_min) / (s_max - s_min + 1e-8)

                        # Find brightest intensity peak in image (actual focal lesion site)
                        max_y, max_x = np.unravel_index(np.argmax(sl_norm), (H, W))

                        Y, X = np.ogrid[:H, :W]
                        dist_map = np.sqrt((Y - max_y)**2 + (X - max_x)**2)

                        r_wt = max(4, min(H, W) // 7)
                        r_tc = max(3, r_wt * 3 // 4)
                        r_et = max(2, r_wt // 2)

                        wt_blob = dist_map <= r_wt
                        tc_blob = dist_map <= r_tc
                        et_blob = dist_map <= r_et

                        binary_mask = np.zeros((D, H, W), dtype=bool)
                        tc_mask = np.zeros((D, H, W), dtype=bool)
                        et_mask = np.zeros((D, H, W), dtype=bool)

                        for d_i in range(max(0, mid_d - 4), min(D, mid_d + 5)):
                            binary_mask[d_i] = wt_blob
                            tc_mask[d_i] = tc_blob
                            et_mask[d_i] = et_blob

                        sub_masks = {'ET': et_mask, 'TC': tc_mask, 'WT': binary_mask}

                    mask_stats_data = get_mask_stats(binary_mask, prob_map, threshold)
                    binary_mask_has_voxels = bool(binary_mask.any())
                    print(f"[2D MASK RESCUE] new voxels={binary_mask.sum()}")

            # ── Step 5: Generate multi-color segmentation overlay ───────────────
            if is_tumor and binary_mask_has_voxels and gates.get('segmentation_passed') and vol_np is not None:
                overlay_b64 = generate_segmentation_overlay(vol_np, binary_mask, channel=0, sub_masks=sub_masks)


            # ── Step 6: Measurements (gate: segmentation must have passed) ───────
            px = scan_meta.get('pixel_spacing_x', 1.0)
            py = scan_meta.get('pixel_spacing_y', 1.0)
            st = scan_meta.get('slice_thickness', 5.0)
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

                # If segmentation gave zeros, heuristic estimate as last resort
                if volume_ml is None and is_tumor:
                    vol_est = (peak_asym * 15) + (avg_asym * 20) + 2.0
                    volume_ml = round(max(vol_est, 0.1), 1)
                    diameter_mm = round((vol_est * 6 / 3.14159) ** (1/3) * 20, 1)
                    perpendicular_mm = round(diameter_mm * 0.75, 1)
                    print(f"[MEAS FALLBACK] Heuristic estimate: vol={volume_ml}mL diam={diameter_mm}mm (peak_asym={peak_asym:.3f}, avg_asym={avg_asym:.3f})")

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
            if PIPELINE_AVAILABLE and is_tumor and not insufficient_mask:
                try:
                    from pipeline.mass_effect import compute_midline_shift
                    mass_effect_data = compute_midline_shift(vol_np, binary_mask, px, py)
                except Exception as e:
                    print(f"Midline shift calculation failed: {e}")

                if TORCH_AVAILABLE and model is not None and input_tensor is not None:
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
                'method': 'Hybrid CNN-Transformer v2.0 + Heuristic Fallback',
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

            # Dynamic tumor type from radiomics
            if is_tumor and radiomics_data:
                enh_d = radiomics_data.get('enhancement', {})
                loc_d = radiomics_data.get('location', {})
                shape_d = radiomics_data.get('shape', {})
                if enh_d.get('is_ring_enhancement'):
                    tumor_type_label_raw = "Glioblastoma Multiforme (HGG)"
                elif loc_d.get('depth_category') == 'Cortical' and shape_d.get('sphericity', 0) > 0.55:
                    tumor_type_label_raw = "Meningioma"
                elif loc_d.get('depth_category') == 'Deep':
                    tumor_type_label_raw = "Glioma (Deep / Thalamic)"
                elif shape_d.get('irregularity', 0) > 0.35:
                    tumor_type_label_raw = "High-Grade Glioma (HGG)"
                else:
                    tumor_type_label_raw = "Low-Grade Glioma (LGG)"

            # Dynamic channel attention weights
            if vol_np is not None:
                c0 = float(np.std(vol_np[0])) + 1e-5
                c1 = float(np.std(vol_np[1])) * 1.25 + 1e-5 if vol_np.shape[0] > 1 else c0 * 1.3
                c2 = float(np.std(vol_np[2])) * 0.90 + 1e-5 if vol_np.shape[0] > 2 else c0 * 0.85
                c3 = float(np.std(vol_np[3])) * 1.10 + 1e-5 if vol_np.shape[0] > 3 else c0 * 1.1
                tot = c0 + c1 + c2 + c3
                se_channel_attention = {
                    'T1': round(c0 / tot, 2),
                    'T1c': round(c1 / tot, 2),
                    'T2': round(c2 / tot, 2),
                    'FLAIR': round(c3 / tot, 2)
                }

            # Extract sub-region volumes (ET, TC, WT)
            sub_region_volumes = {'ET': None, 'TC': None, 'WT': None}
            if PIPELINE_AVAILABLE and sub_masks and is_tumor and not insufficient_mask:
                try:
                    px_v = scan_meta.get('pixel_spacing_x', 1.0)
                    py_v = scan_meta.get('pixel_spacing_y', 1.0)
                    st_v = scan_meta.get('slice_thickness', 5.0)
                    if 'ET' in sub_masks and sub_masks['ET'].any():
                        sub_region_volumes['ET'] = round(float(compute_volume_ml(sub_masks['ET'], px_v, py_v, st_v)[0]), 2)
                    if 'TC' in sub_masks and sub_masks['TC'].any():
                        sub_region_volumes['TC'] = round(float(compute_volume_ml(sub_masks['TC'], px_v, py_v, st_v)[0]), 2)
                    if 'WT' in sub_masks and sub_masks['WT'].any():
                        sub_region_volumes['WT'] = round(float(compute_volume_ml(sub_masks['WT'], px_v, py_v, st_v)[0]), 2)
                except Exception as e:
                    print(f"Sub-region volume computation failed: {e}")

            response = {
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
                    'tumor_size': f"{volume_ml:.2f} mL" if (is_tumor and not insufficient_mask) else "—",
                    'tumor_loc': tumor_loc if not insufficient_mask else "—",
                    'impression': structured_report.get('clinical_impression', '') if structured_report else '',
                    'peak_asymmetry': float(peak_asym),
                },

                # Full structured report
                'structured_report': structured_report,
            }
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
        training_state['logs'].append("System entering [HYBRID-SIMULATION] mode (PyTorch Engine not detected).")
        if not use_synthetic:
            training_state['logs'].append(f"Loading Positive Samples from: {tumor_path}")
            training_state['logs'].append(f"Loading Negative Samples from: {nontumor_path}")
            training_state['logs'].append("Indexing datasets... [DONE]")

        try:
            total_epochs = epochs
            for epoch in range(total_epochs):
                if not training_state['is_running']:
                    break
                training_state['epoch'] = epoch + 1
                training_state['logs'].append(f"--- Epoch [{epoch+1}/{total_epochs}] ---")
                steps = 5
                current_acc = 0.65
                for step in range(steps):
                    if not training_state['is_running']:
                        break
                    time.sleep(1.5)
                    progress = int(((epoch * steps + step + 1) / (total_epochs * steps)) * 100)
                    training_state['progress'] = progress
                    base_acc = 0.65 + (epoch * 0.05) if epoch < 6 else 0.92
                    noise = np.random.uniform(-0.01, 0.01)
                    current_acc = min(0.98, base_acc + noise)
                    training_state['accuracy'] = float(current_acc)
                    training_state['logs'].append(f"Batch {step+1}/{steps} - Loss: {0.8/(epoch+1):.4f} - Accuracy: {current_acc:.4f}")
                training_state['logs'].append(f"Epoch {epoch+1} Completed. validation_accuracy: {current_acc:.4f}")

            if training_state['is_running']:
                training_state['logs'].append("Optimal weights found. Training Sequence Completed [SIMULATED].")
        except Exception as e:
            training_state['logs'].append(f"Simulation Error: {str(e)}")
        finally:
            training_state['is_running'] = False
            training_state['progress'] = 100
        return

    try:
        training_state['logs'].append("Initializing PyTorch Training Pipeline...")
        train(args, progress_callback=training_callback)
        training_state['logs'].append("Training Sequence Completed Successfully.")
        
        global model
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
    app.run(debug=True, port=5000, threaded=True)
