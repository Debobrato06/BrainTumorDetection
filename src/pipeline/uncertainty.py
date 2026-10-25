"""
Stage 8: MC-Dropout Uncertainty Estimation
------------------------------------------
Performs Monte Carlo Dropout during model inference.
Runs N forward passes with dropout active, then computes pixel-wise variance.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def estimate_uncertainty(model, volume_tensor, num_samples=10) -> tuple:
    """
    Compute mean prediction map and uncertainty variance map using MC-Dropout.

    Args:
        model: PyTorch model instance (must have dropout layers)
        volume_tensor: Torch tensor (B, C, D, H, W)
        num_samples: Number of forward passes

    Returns:
        mean_probability: numpy array (D, H, W)
        uncertainty_map: numpy array (D, H, W) — pixel-wise variance
        average_uncertainty_score: float — mean variance in the positive region
        (all three are None when uncertainty cannot be computed)
    """
    if not TORCH_AVAILABLE:
        return None, None, None

    try:
        # Keep BatchNorm in eval mode (running statistics) and activate only dropout layers;
        # model.train() would also switch BatchNorm to batch statistics and distort outputs.
        was_training = model.training
        model.eval()
        n_dropout = 0
        for m in model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                m.train()
                n_dropout += 1
        if n_dropout == 0:
            print("[Uncertainty] Model has no dropout layers; MC-Dropout uncertainty is undefined")
            model.train(was_training)
            return None, None, None
        preds = []

        with torch.no_grad():
            for _ in range(num_samples):
                # We expect the model forward to return (cls_logits, seg_logits, grade_logits)
                out = model(volume_tensor)
                # seg_logits is the second element
                if isinstance(out, tuple) and len(out) >= 2:
                    seg_logits = out[1]
                else:
                    seg_logits = out.get('mask') if isinstance(out, dict) else out

                probs = torch.sigmoid(seg_logits).squeeze()
                preds.append(probs)

        model.train(was_training)

        # Calculate metrics
        stacked = torch.stack(preds)
        mean_prob = stacked.mean(dim=0).cpu().numpy()
        var_map = stacked.var(dim=0).cpu().numpy()
        if not np.any(var_map > 0):
            # Dropout exists only outside the segmentation path (e.g. classification heads),
            # so the masks never vary: there is no segmentation uncertainty to report.
            print("[Uncertainty] Segmentation output has no dropout-induced variance; not computed")
            return None, None, None

        # Uncertainty score is average variance in tumor boundaries
        tumor_mask = mean_prob > 0.5
        if tumor_mask.any():
            avg_uncertainty = float(var_map[tumor_mask].mean())
        else:
            avg_uncertainty = float(var_map.mean())

        return mean_prob, var_map, round(avg_uncertainty, 6)

    except Exception as e:
        print(f"[Uncertainty] MC-Dropout computation failed: {e}")
        # Report "not computed" rather than a fake zero uncertainty
        return None, None, None
