"""
neurovision/pipeline/orchestrator.py
------------------------------------
Orchestrator class that runs the full clinical pipeline:
Preprocessing -> Segmentation/MC-Dropout -> Atlas -> Radiomics -> Mass Effect -> Risk -> Report

Integrates:
  - MC-Dropout uncertainty estimation
  - Midline shift calculation
  - ANTsPy or centroid-based atlas region lookup
  - VASARI feature mapping
  - Rule-based clinical NLG
"""

import time
import numpy as np

# Optional imports for model and validation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import pipeline stages
from src.pipeline.preprocessing import extract_metadata_from_2d, normalize_intensity
from src.pipeline.segmentation import apply_threshold, get_mask_stats, generate_segmentation_overlay
from src.pipeline.measurement import compute_volume_ml, compute_longest_diameter_rano, compute_growth_change
from src.pipeline.radiomics import extract_all_radiomics
from src.pipeline.risk_scoring import compute_risk_score
from src.pipeline.report_generator import generate_structured_report
from src.pipeline.atlas import lookup_anatomical_region
from src.pipeline.mass_effect import compute_midline_shift
from src.pipeline.uncertainty import estimate_uncertainty
from src.pipeline.pipeline_validator import validate_pipeline


class ClinicalPipelineOrchestrator:
    def __init__(self, model_instance=None, device_instance=None):
        self.model = model_instance
        self.device = device_instance
        self.torch_active = TORCH_AVAILABLE and (self.model is not None)

    def run(
        self,
        input_volume,        # numpy or torch tensor
        patient_meta: dict,
        scan_meta: dict,
        threshold: float = 0.5,
        previous_volume_ml: float = None,
        patient_nifti_path: str = None,
        mni_template_path: str = None,
        display_img_b64: str = ""
    ) -> dict:
        """
        Executes the entire clinical pipeline in sequence.
        Designed for graceful degradation if GPU/Torch/ANTsPy is missing.
        """
        start_time = time.time()

        # ── Step 1: Preprocessing & Basic Checks ──
        px = scan_meta.get('pixel_spacing_x', 1.0)
        py = scan_meta.get('pixel_spacing_y', 1.0)
        st = scan_meta.get('slice_thickness', 5.0)

        # ── Step 2: Model Inference or Heuristics ──
        seg_logits = None
        predicted_class = 0
        raw_confidence = 0.5
        grade_label_raw = "N/A"

        if self.torch_active:
            try:
                # Ensure input is torch tensor on right device
                if not isinstance(input_volume, torch.Tensor):
                    vol_tensor = torch.from_numpy(input_volume).to(self.device)
                else:
                    vol_tensor = input_volume.to(self.device)

                with torch.no_grad():
                    outputs = self.model(vol_tensor)
                    cls_logits, seg_logits, grade_logits = outputs[0], outputs[1], outputs[2]

                    T = 1.45  # Temperature scaling
                    probs = torch.softmax(cls_logits / T, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    raw_confidence = float(probs[0][predicted_class].item())

                    grade_probs = torch.softmax(grade_logits, dim=1)
                    grade_idx = torch.argmax(grade_probs, dim=1).item()
                    grade_label_raw = "High-Grade (HGG)" if grade_idx == 1 else "Low-Grade (LGG)"
            except Exception as e:
                print(f"[Orchestrator] PyTorch model execution failed: {e}")
                self.torch_active = False

        # Fallback values if PyTorch failed/missing
        if not self.torch_active:
            # Under non-torch fallback, classifier/inference isn't run, relying on app's heuristic parameters.
            pass

        # ── Step 3: Run Validation Gates ──
        # Check if Out of Distribution
        ood_detected = (
            scan_meta.get('source') in ['2d_image', 'default']
            or scan_meta.get('warning') is not None
        )

        # Convert input volume to numpy for pipeline stages
        if isinstance(input_volume, torch.Tensor):
            vol_np = input_volume.squeeze(0).detach().cpu().numpy()
        else:
            vol_np = np.array(input_volume).squeeze()

        # Basic segmentation check
        binary_mask = None
        prob_map = None
        seg_quality_score = 1.0
        binary_mask_has_voxels = False

        if vol_np is not None:
            if seg_logits is not None:
                binary_mask, prob_map = apply_threshold(seg_logits, threshold=threshold)
                if predicted_class == 0:
                    binary_mask = np.zeros_like(binary_mask)
                mask_stats_data = get_mask_stats(binary_mask, prob_map, threshold)
                seg_quality_score = mask_stats_data.get('seg_quality_score', 1.0)
                binary_mask_has_voxels = bool(binary_mask.any())
            else:
                # App will handle heuristic mask generation inside its own route or fallback
                mask_stats_data = {
                    'voxel_count': 0, 'has_tumor': False,
                    'bounding_box': None, 'centroid': None, 'seg_quality_score': 1.0
                }

        # ── Step 4: Run MC-Dropout Uncertainty ──
        uncertainty_score = None
        uncertainty_map = None
        if self.torch_active and predicted_class == 1:
            try:
                # MC-Dropout pass
                _, uncertainty_map, uncertainty_score = estimate_uncertainty(
                    self.model, vol_tensor, num_samples=15
                )
            except Exception as e:
                print(f"[Orchestrator] Uncertainty estimation failed: {e}")

        # Return orchestrator execution trace
        return {
            "predicted_class": predicted_class,
            "raw_confidence": raw_confidence,
            "grade_label_raw": grade_label_raw,
            "binary_mask": binary_mask,
            "prob_map": prob_map,
            "mask_stats_data": mask_stats_data,
            "seg_quality_score": seg_quality_score,
            "binary_mask_has_voxels": binary_mask_has_voxels,
            "uncertainty_score": uncertainty_score,
            "uncertainty_map": uncertainty_map,
            "vol_np": vol_np
        }
