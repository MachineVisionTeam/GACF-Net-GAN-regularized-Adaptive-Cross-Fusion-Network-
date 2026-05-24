"""Stronger augmentations for FAN-Seg Phase 1 v2: HED color jitter + elastic deformation.

Both operate on numpy arrays at the loader level. Compatible with HoVer-Net's HV map
generation (we apply geometric transforms to image+label jointly, then re-derive HV).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# RGB ↔ HED color matrices (Ruifrok & Johnston 2001), same as HoVer-NeXt
RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
_EPS = 1e-6


def rgb_to_hed(rgb):
    """rgb: (H, W, 3) in [0, 1] → hed (H, W, 3)."""
    rgb = np.maximum(rgb, _EPS)
    log_rgb = np.log(rgb) / np.log(_EPS)
    return log_rgb @ HED_FROM_RGB


def hed_to_rgb(hed):
    """hed: (H, W, 3) → rgb (H, W, 3) in [0, 1]."""
    log_e = -np.log(_EPS)
    rgb = np.exp(-(hed * log_e) @ RGB_FROM_HED)
    return np.clip(rgb, 0.0, 1.0)


def hed_jitter(rgb, sigma=0.05, bias=0.05, rng=None):
    """Apply random per-channel multiplicative + additive jitter in HED space.

    rgb: (H, W, 3) in [0, 1]. Returns the same shape, also in [0, 1].
    sigma:  ±5% multiplicative jitter per HED channel
    bias:   ±5% additive jitter per HED channel
    """
    rng = rng or np.random
    hed = rgb_to_hed(rgb)
    alpha = rng.uniform(1.0 - sigma, 1.0 + sigma, size=(1, 1, 3)).astype(np.float32)
    beta  = rng.uniform(-bias, bias, size=(1, 1, 3)).astype(np.float32)
    hed = hed * alpha + beta
    return hed_to_rgb(hed)


def elastic_transform(image, inst_map, type_map, alpha=120.0, sigma=8.0, rng=None):
    """Elastic deformation (Simard et al. 2003). Joint warping of image + label.

    image:    (H, W, 3) float32
    inst_map: (H, W) int32 (instance IDs)
    type_map: (H, W) int32 (class labels)
    Returns: (image_warped, inst_warped, type_warped) with same shapes/dtypes.

    Uses nearest-neighbour interpolation for label maps to preserve integer IDs.
    """
    rng = rng or np.random
    H, W = image.shape[:2]
    dx = gaussian_filter(rng.uniform(-1, 1, (H, W)), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (H, W)), sigma) * alpha
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([y + dy, x + dx], axis=0)

    # Bilinear interpolation for image
    img_warp = np.stack(
        [map_coordinates(image[..., c], coords, order=1, mode="reflect") for c in range(3)],
        axis=-1,
    ).astype(image.dtype)
    # Nearest for label maps
    inst_warp = map_coordinates(inst_map.astype(np.int32), coords, order=0, mode="reflect").astype(inst_map.dtype)
    type_warp = map_coordinates(type_map.astype(np.int32), coords, order=0, mode="reflect").astype(type_map.dtype)
    return img_warp, inst_warp, type_warp
