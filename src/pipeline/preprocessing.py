"""
Stage 1: Image Input & Preprocessing
- Extracts scan metadata from NIfTI affine (pixel_spacing, slice_thickness)
- Intensity normalization (percentile-based, simulating N4 bias correction output)
- Provides default clinical values for 2D PNG inputs with a warning flag
"""

import numpy as np
import os

# Default clinical values for 2D PNG/JPG uploads (no DICOM metadata available)
DEFAULT_PIXEL_SPACING_MM = 1.0   # 1 mm/pixel — standard 1x1mm clinical MRI
DEFAULT_SLICE_THICKNESS_MM = 5.0  # 5 mm — common T1/T2 slice thickness


def extract_metadata_from_nifti(nib_img):
    """
    Extract pixel spacing and slice thickness from NIfTI affine matrix.
    
    Returns a dict with:
        - pixel_spacing_x (mm)
        - pixel_spacing_y (mm)
        - slice_thickness (mm)
        - voxel_volume_mm3
        - image_shape
        - orientation (approx)
        - source: 'nifti'
        - warning: None
    """
    try:
        affine = nib_img.affine
        # Voxel sizes from the affine (column norms = voxel sizes in mm)
        voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        pixel_spacing_x = float(abs(voxel_sizes[0]))
        pixel_spacing_y = float(abs(voxel_sizes[1]))
        slice_thickness = float(abs(voxel_sizes[2]))

        # Clamp to reasonable clinical range (0.1mm – 10mm)
        pixel_spacing_x = max(0.1, min(10.0, pixel_spacing_x))
        pixel_spacing_y = max(0.1, min(10.0, pixel_spacing_y))
        slice_thickness = max(0.5, min(10.0, slice_thickness))

        voxel_volume_mm3 = pixel_spacing_x * pixel_spacing_y * slice_thickness

        shape = nib_img.shape
        image_shape = {
            'depth': int(shape[2]) if len(shape) > 2 else 1,
            'height': int(shape[1]) if len(shape) > 1 else int(shape[0]),
            'width': int(shape[0]),
            'channels': int(shape[3]) if len(shape) > 3 else 1
        }

        # Approximate orientation from affine diagonal signs
        orientation = _infer_orientation(affine)

        return {
            'pixel_spacing_x': round(pixel_spacing_x, 3),
            'pixel_spacing_y': round(pixel_spacing_y, 3),
            'slice_thickness': round(slice_thickness, 3),
            'voxel_volume_mm3': round(voxel_volume_mm3, 4),
            'image_shape': image_shape,
            'orientation': orientation,
            'source': 'nifti',
            'warning': None
        }

    except Exception as e:
        # Fallback to defaults if affine parsing fails
        return _default_metadata(warning=f"NIfTI metadata parse failed ({e}). Using clinical defaults.")


def extract_metadata_from_2d(image_path):
    """
    For 2D PNG/JPG: no DICOM metadata available.
    Returns defaults with a visible warning.
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size

        # Check EXIF for any pixel spacing hints (rare but possible in medical JPEG)
        dpi = img.info.get('dpi', (None, None))
        if dpi and dpi[0] and dpi[0] > 0:
            # Convert DPI to mm/pixel: 1 inch = 25.4 mm
            px = round(25.4 / dpi[0], 3)
            py = round(25.4 / dpi[1], 3)
        else:
            px = DEFAULT_PIXEL_SPACING_MM
            py = DEFAULT_PIXEL_SPACING_MM

        voxel_volume_mm3 = px * py * DEFAULT_SLICE_THICKNESS_MM
        return {
            'pixel_spacing_x': px,
            'pixel_spacing_y': py,
            'slice_thickness': DEFAULT_SLICE_THICKNESS_MM,
            'voxel_volume_mm3': round(voxel_volume_mm3, 4),
            'image_shape': {'depth': 1, 'height': h, 'width': w, 'channels': 3},
            'orientation': 'AXIAL (assumed)',
            'source': '2d_image',
            'warning': (
                f"2D image detected — no DICOM metadata. "
                f"Using clinical defaults: pixel_spacing={px}mm, slice_thickness={DEFAULT_SLICE_THICKNESS_MM}mm. "
                f"Volume estimates should be interpreted with caution."
            )
        }
    except Exception as e:
        return _default_metadata(warning=f"Image metadata parse failed ({e}). Using clinical defaults.")


def normalize_intensity(volume_np):
    """
    Percentile-based intensity normalization (simulates N4 bias field correction output).
    Clips to [1st, 99th] percentile and scales to [0, 1].
    
    Args:
        volume_np: numpy array, any shape (C, D, H, W) or (D, H, W)
    Returns:
        normalized numpy array, same shape, float32
    """
    volume_np = volume_np.astype(np.float32)
    p1 = np.percentile(volume_np, 1)
    p99 = np.percentile(volume_np, 99)
    if p99 > p1:
        volume_np = (volume_np - p1) / (p99 - p1 + 1e-8)
    volume_np = np.clip(volume_np, 0.0, 1.0)
    return volume_np


def get_scan_info_string(metadata):
    """Human-readable scan info string for the UI."""
    shape = metadata.get('image_shape', {})
    return (
        f"{shape.get('width', '?')}×{shape.get('height', '?')}×{shape.get('depth', '?')} vox | "
        f"px={metadata['pixel_spacing_x']}mm | "
        f"st={metadata['slice_thickness']}mm | "
        f"{metadata['orientation']}"
    )


def _infer_orientation(affine):
    """Rough orientation inference from affine matrix diagonal."""
    try:
        d = np.diag(affine[:3, :3])
        labels = ['R/L', 'A/P', 'S/I']
        dominant_axis = int(np.argmax(np.abs(d)))
        orientations = {0: 'SAGITTAL', 1: 'CORONAL', 2: 'AXIAL'}
        return orientations.get(dominant_axis, 'AXIAL')
    except Exception:
        return 'AXIAL'


def _default_metadata(warning=None):
    voxel_volume = DEFAULT_PIXEL_SPACING_MM * DEFAULT_PIXEL_SPACING_MM * DEFAULT_SLICE_THICKNESS_MM
    return {
        'pixel_spacing_x': DEFAULT_PIXEL_SPACING_MM,
        'pixel_spacing_y': DEFAULT_PIXEL_SPACING_MM,
        'slice_thickness': DEFAULT_SLICE_THICKNESS_MM,
        'voxel_volume_mm3': round(voxel_volume, 4),
        'image_shape': {'depth': 64, 'height': 64, 'width': 64, 'channels': 4},
        'orientation': 'AXIAL (default)',
        'source': 'default',
        'warning': warning or "Using clinical default metadata values."
    }
