"""
Builds the proposed model or a baseline from a config dict.

Every model returns (cls_logits, seg_logits, grade_logits). Segmentation-only
baselines return None for cls/grade, so they are trained and evaluated with the
same loop, crop size, splits and loss as the proposed model.
nnU-Net v2 is run with its own CLI on the same splits (scripts/export_nnunet_splits.py).
"""
import torch.nn as nn

from hybrid_model import HybridTumorModel


class SegOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, **_):
        return None, self.net(x), None


def build_model(model_cfg, img_size, in_channels=4, out_channels=3):
    name = model_cfg["name"].lower()
    kw = dict(model_cfg.get("kwargs", {}))
    img_size = tuple(img_size)

    if name == "hybrid":
        return HybridTumorModel(in_channels=in_channels, img_size=img_size, num_seg_classes=out_channels, **kw)

    # Baselines from MONAI (imported lazily so the app does not need MONAI)
    from monai.networks import nets

    if name == "unet3d":
        net = nets.UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                        channels=kw.get("channels", (32, 64, 128, 256, 320)),
                        strides=kw.get("strides", (2, 2, 2, 2)), num_res_units=kw.get("num_res_units", 2),
                        norm="instance")
    elif name == "segresnet":
        net = nets.SegResNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                             init_filters=kw.get("init_filters", 16), blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                             dropout_prob=kw.get("dropout_prob", 0.2))
    elif name == "unetr":
        net = nets.UNETR(in_channels=in_channels, out_channels=out_channels, img_size=img_size,
                         feature_size=kw.get("feature_size", 16), hidden_size=768, mlp_dim=3072, num_heads=12,
                         norm_name="instance")
    elif name == "swinunetr":
        net = nets.SwinUNETR(in_channels=in_channels, out_channels=out_channels,
                             feature_size=kw.get("feature_size", 48), use_checkpoint=kw.get("use_checkpoint", True),
                             **({"img_size": img_size} if "img_size" in nets.SwinUNETR.__init__.__code__.co_varnames else {}))
    else:
        raise ValueError(f"Unknown model name: {name}")
    return SegOnly(net)
