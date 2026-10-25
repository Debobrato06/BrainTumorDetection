"""2D tumor segmentation + tumor-type model for single MRI slices (JPG/PNG).

TumorSegNet2D: ImageNet-pretrained ResNet34 encoder with a U-Net decoder.
  - seg head: 1-channel tumor mask logits at input resolution
  - cls head: 4 classes (CLASSES) from the encoder's global feature

Trained by src/train_seg2d.py on the figshare dataset (expert tumor masks, T1-CE) plus the
Kaggle Brain Tumor MRI Dataset (class labels only). Both use the same input: grayscale, padded
to a square, resized to INPUT_SIZE, scaled to [0, 1] (prepare_slice).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 256
CLASSES = ['No tumor', 'Glioma', 'Meningioma', 'Pituitary tumor']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def prepare_slice(img, size=INPUT_SIZE):
    """PIL image -> (uint8 array (size, size), info). The image is converted to grayscale and
    zero-padded to a square (centered) before resizing, so one output pixel covers the same
    distance horizontally and vertically. info maps output pixels back to the original image."""
    import cv2
    g = np.asarray(img.convert('L'), dtype=np.uint8)
    h, w = g.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    sq = np.zeros((side, side), dtype=np.uint8)
    sq[top:top + h, left:left + w] = g
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    out = cv2.resize(sq, (size, size), interpolation=interp)
    return out, {'orig_h': h, 'orig_w': w, 'pad_top': top, 'pad_left': left,
                 'orig_px_per_input_px': side / size}


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.block = _conv_block(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class TumorSegNet2D(nn.Module):
    def __init__(self, pretrained=False, num_classes=len(CLASSES)):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # /2, 64
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.up4 = _Up(512, 256, 256)                            # /16
        self.up3 = _Up(256, 128, 128)                            # /8
        self.up2 = _Up(128, 64, 64)                              # /4
        self.up1 = _Up(64, 64, 32)                               # /2
        # BatchNorm before the ReLU: without it the 16 units died early in training and the
        # tumor probability could not exceed 0.48 anywhere
        self.seg_out = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16),
                                     nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        self.cls_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        """x: (B, 1, H, W) in [0, 1] -> (seg_logits (B, 1, H, W), cls_logits (B, num_classes))"""
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        cls_logits = self.cls_head(F.adaptive_avg_pool2d(s4, 1).flatten(1))
        d = self.up1(self.up2(self.up3(self.up4(s4, s3), s2), s1), s0)
        d = F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.seg_out(d), cls_logits


def load_seg2d(path, device='cpu'):
    """Load a checkpoint written by train_seg2d.py -> (model in eval mode, checkpoint metadata)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TumorSegNet2D(pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ('model_state_dict', 'optimizer_state_dict')}
    return model, meta


@torch.no_grad()
def predict_slice(model, slice_u8, device='cpu', tta=True):
    """uint8 (S, S) -> (tumor probability map (S, S) float32, class probabilities (4,)).
    With tta, the prediction is averaged with that of the horizontally flipped image."""
    x = torch.from_numpy(slice_u8).float().div(255.0)[None, None].to(device)
    if tta:
        x = torch.cat([x, torch.flip(x, dims=[3])])
    seg, cls = model(x)
    prob = torch.sigmoid(seg)
    cls_p = torch.softmax(cls, dim=1)
    if tta:
        prob = (prob[0:1] + torch.flip(prob[1:2], dims=[3])) / 2
        cls_p = cls_p.mean(dim=0, keepdim=True)
    return prob[0, 0].cpu().numpy(), cls_p[0].cpu().numpy()
