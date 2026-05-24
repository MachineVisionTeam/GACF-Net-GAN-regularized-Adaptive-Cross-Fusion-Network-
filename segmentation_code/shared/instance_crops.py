"""Helper to extract per-instance 64×64 RGB+mask crops + UNI token at centroid.

Used by Phase 2 training and TTA. Vectorized where possible.

Output per instance:
  - crop:  (4, 64, 64) float32 — RGB + binary mask, concatenated
  - token: (1024,) float32   — UNI patch token at the centroid (16-grid → bilinear)
  - centroid_y, centroid_x
"""
import torch
import torch.nn.functional as F
import numpy as np


CROP_SIZE = 64
TOKEN_GRID = 16   # UNI ViT-L/16 produces 16x16 token grid for 256x256 input


def pad_and_crop(image_chw, mask_2d, cy, cx, size=CROP_SIZE):
    """Pad-and-crop a region of given size centered at (cy, cx).

    Args:
        image_chw: (3, H, W) float — RGB image
        mask_2d:   (H, W) bool/{0,1}— binary mask (single instance)
        cy, cx:    int centroid

    Returns:
        crop:  (4, size, size) float32 tensor [RGB, mask]
    """
    H, W = mask_2d.shape
    half = size // 2
    # Reflection-pad to handle near-edge centroids
    pad = half + 1
    img_padded = F.pad(image_chw.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    msk_padded = F.pad(mask_2d.unsqueeze(0).unsqueeze(0).float(),
                       (pad, pad, pad, pad), mode="constant", value=0).squeeze()
    cy_p = int(cy) + pad
    cx_p = int(cx) + pad
    img_c = img_padded[:, cy_p - half:cy_p + half, cx_p - half:cx_p + half]
    msk_c = msk_padded[cy_p - half:cy_p + half, cx_p - half:cx_p + half]
    return torch.cat([img_c, msk_c.unsqueeze(0)], dim=0)   # (4, size, size)


def token_at_centroid(token_grid_chw, cy, cx, image_h=256, image_w=256):
    """Bilinear sample the UNI token at pixel (cy, cx) from the 16×16 token grid.

    Args:
        token_grid_chw: (1024, 16, 16) float — single image's token grid
        cy, cx:         int pixel coordinates in 256×256 space

    Returns:
        token: (1024,) float32 tensor
    """
    # Map pixel coords to grid coords [-1, 1]
    gy = (cy / image_h) * 2 - 1     # cy in [0, H) → gy in [-1, 1)
    gx = (cx / image_w) * 2 - 1
    grid = torch.tensor([[[[gx, gy]]]], dtype=token_grid_chw.dtype,
                         device=token_grid_chw.device)   # (1, 1, 1, 2)
    sampled = F.grid_sample(token_grid_chw.unsqueeze(0), grid,
                             mode="bilinear", align_corners=False)
    return sampled.view(-1)   # (1024,)


def build_instance_crops_batch(image_b3hw, inst_map_b1hw, token_grid_b1024_16_16,
                                  max_per_image=64, size=CROP_SIZE):
    """For a batch of images + instance maps, build a flat list of per-instance crops + tokens.

    Args:
        image_b3hw:  (B, 3, H, W) float in [0,1]
        inst_map_bhw: (B, H, W) int — instance IDs, 0=bg
        token_grid:  (B, 1024, 16, 16) float — UNI tokens per image
        max_per_image: cap on instances per image to bound compute (random subsample if exceeded)

    Returns:
        crops:    (N, 4, size, size) tensor
        tokens:   (N, 1024) tensor
        batch_id: (N,) which image in the batch each instance came from
        nuc_ids:  (N,) instance ID within its image
    """
    device = image_b3hw.device
    B = image_b3hw.shape[0]
    crops_list, tokens_list, batch_id_list, nuc_id_list = [], [], [], []

    for b in range(B):
        inst_map = inst_map_b1hw[b]                  # (H, W)
        unique_ids = torch.unique(inst_map)
        unique_ids = unique_ids[unique_ids > 0].tolist()
        if max_per_image and len(unique_ids) > max_per_image:
            chosen = np.random.choice(unique_ids, max_per_image, replace=False).tolist()
            unique_ids = chosen

        if len(unique_ids) == 0:
            continue

        for nid in unique_ids:
            mask = (inst_map == nid)
            if mask.sum() < 5:    # skip tiny artifacts
                continue
            ys, xs = torch.where(mask)
            cy = ys.float().mean().item()
            cx = xs.float().mean().item()

            crop = pad_and_crop(image_b3hw[b], mask, cy, cx, size=size)
            tok  = token_at_centroid(token_grid_b1024_16_16[b], cy, cx)
            crops_list.append(crop)
            tokens_list.append(tok)
            batch_id_list.append(b)
            nuc_id_list.append(int(nid))

    if not crops_list:
        return None, None, None, None
    crops = torch.stack(crops_list).to(device)
    tokens = torch.stack(tokens_list).to(device)
    batch_id = torch.tensor(batch_id_list, dtype=torch.long, device=device)
    nuc_ids  = torch.tensor(nuc_id_list, dtype=torch.long, device=device)
    return crops, tokens, batch_id, nuc_ids
