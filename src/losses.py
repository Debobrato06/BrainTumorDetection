import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky loss on sigmoid outputs, averaged over channels (WT/TC/ET).

        TI = TP / (TP + alpha * FP + beta * FN)

    alpha weights false positives and beta weights false negatives, so
    beta > alpha (default alpha=0.3, beta=0.7) penalizes missed tumor voxels more.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits, targets: (B, C, D, H, W); targets binary per channel
        probs = torch.sigmoid(logits)
        dims = (0,) + tuple(range(2, probs.ndim))  # sum over batch and space, keep channels
        tp = (probs * targets).sum(dims)
        fp = (probs * (1 - targets)).sum(dims)
        fn = ((1 - probs) * targets).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class MultiTaskLoss(nn.Module):
    """
    L = w_seg * Tversky(seg) + w_cls * CE(cls) + w_grade * CE(grade)

    Grade labels equal to -1 (unknown) are ignored; if a whole batch has no grade
    label the grade term is 0. Pass a missing target as None to skip that task.
    """

    def __init__(self, w_seg=0.6, w_cls=0.2, w_grade=0.2, tversky_alpha=0.3, tversky_beta=0.7,
                 cls_weight=None):
        super().__init__()
        self.w_seg, self.w_cls, self.w_grade = w_seg, w_cls, w_grade
        self.seg_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.register_buffer("cls_weight", None if cls_weight is None else torch.as_tensor(cls_weight, dtype=torch.float))

    def forward(self, outputs, seg=None, label=None, grade=None):
        cls_logits, seg_logits, grade_logits = outputs[:3]
        terms = {}
        zero = seg_logits.sum() * 0.0
        terms["seg"] = self.seg_loss(seg_logits, seg) if seg is not None else zero
        terms["cls"] = F.cross_entropy(cls_logits, label, weight=self.cls_weight) if label is not None else zero
        if grade is not None and (grade >= 0).any():
            terms["grade"] = F.cross_entropy(grade_logits, grade, ignore_index=-1)
        else:
            terms["grade"] = zero
        total = self.w_seg * terms["seg"] + self.w_cls * terms["cls"] + self.w_grade * terms["grade"]
        return total, terms
