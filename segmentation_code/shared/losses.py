"""Loss functions for FAN-Seg Phase 0: HV (foreground MSE) + Sem3 (CE + Dice)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def hv_loss(hv_pred, hv_gt, fg_mask):
    """MSE on foreground pixels only.

    hv_pred, hv_gt: (B, 2, H, W) in [-1, 1]
    fg_mask:        (B, H, W)    binary {0, 1}
    """
    fg = fg_mask.unsqueeze(1)                         # (B, 1, H, W)
    diff2 = (hv_pred - hv_gt).pow(2) * fg
    n = fg.sum().clamp(min=1.0) * 2                   # 2 channels
    return diff2.sum() / n


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def _hv_gradients(hv):
    """Return (gx_h, gy_h, gx_v, gy_v) — Sobel gradients of horizontal & vertical maps."""
    sx = _SOBEL_X.to(hv.device, hv.dtype)
    sy = _SOBEL_Y.to(hv.device, hv.dtype)
    h = hv[:, 0:1]; v = hv[:, 1:2]
    return (
        torch.nn.functional.conv2d(h, sx, padding=1),
        torch.nn.functional.conv2d(h, sy, padding=1),
        torch.nn.functional.conv2d(v, sx, padding=1),
        torch.nn.functional.conv2d(v, sy, padding=1),
    )


def msge_loss(hv_pred, hv_gt, fg_mask):
    """Mean squared gradient error on HV maps (HoVer-Net 2019 formulation).
    Computed on foreground pixels only."""
    fg = fg_mask.unsqueeze(1)
    g_pred = _hv_gradients(hv_pred)
    g_gt   = _hv_gradients(hv_gt)
    diff2 = sum((gp - gg).pow(2) for gp, gg in zip(g_pred, g_gt))   # sum of 4 terms
    diff2 = diff2 * fg
    n = fg.sum().clamp(min=1.0) * 4                                  # 4 gradient terms
    return diff2.sum() / n


def dice_loss_3class(sem3_logits, sem_gt, eps=1e-6):
    """Multi-class Dice averaged across classes."""
    p = F.softmax(sem3_logits, dim=1)                 # (B, 3, H, W)
    onehot = F.one_hot(sem_gt, num_classes=3).permute(0, 3, 1, 2).float()
    inter = (p * onehot).sum(dim=(0, 2, 3))
    union = p.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def sem3_loss(sem3_logits, sem_gt, weight=None):
    """0.5 * CE + 0.5 * Dice on the 3-class semantic head.

    weight: optional (3,) tensor of per-class CE weights. Useful when boundary
            pixels are rare relative to bg/fg (typical for small nuclei).
    """
    ce = F.cross_entropy(sem3_logits, sem_gt, weight=weight)
    dice = dice_loss_3class(sem3_logits, sem_gt)
    return 0.5 * ce + 0.5 * dice, ce.detach(), dice.detach()


# =============================================================================
# Phase 2 GAN losses — hinge form, per-instance, morphology-balanced
# =============================================================================
def hinge_d_loss(real_logits, fake_logits, weights=None):
    """Hinge discriminator loss.
    L_D = mean( w * [ relu(1 - D(real)) + relu(1 + D(fake)) ] )
    """
    real_term = torch.relu(1.0 - real_logits)
    fake_term = torch.relu(1.0 + fake_logits)
    per_item  = real_term + fake_term
    if weights is not None:
        per_item = per_item * weights
        return per_item.sum() / weights.sum().clamp(min=1.0)
    return per_item.mean()


def hinge_g_loss(fake_logits, weights=None):
    """Hinge generator loss (non-saturating).
    L_G = -mean( w * D(fake) )
    """
    if weights is not None:
        per_item = -fake_logits * weights
        return per_item.sum() / weights.sum().clamp(min=1.0)
    return (-fake_logits).mean()


def cluster_inverse_freq_weights(cluster_ids, freqs, gamma=0.5):
    """w_k = (1 / f_{c_k})^gamma. cluster_ids: (N,) long; freqs: list[K]."""
    freqs_t = torch.tensor(freqs, dtype=torch.float32, device=cluster_ids.device)
    f_per_item = freqs_t[cluster_ids].clamp(min=1e-6)
    return (1.0 / f_per_item).pow(gamma)


# =============================================================================
# Type head: focal class-balanced cross-entropy for per-pixel nucleus classification
# =============================================================================
def focal_class_balanced_ce(logits, target, class_weights, gamma=2.0, ignore_index=-100):
    """Focal cross-entropy with per-class weighting for the type head.

    Args:
        logits        : (B, C, H, W) per-pixel class logits
        target        : (B, H, W)    int target class IDs (0 = bg, 1..K = nucleus classes)
        class_weights : (C,) tensor — inverse-frequency weights (or any) per class
        gamma         : focal modulation strength (Lin et al. 2017); 0 = no focal
        ignore_index  : pixel value to ignore in loss (default -100, no ignore)

    Returns: scalar loss.
    """
    import torch.nn.functional as F
    log_p = F.log_softmax(logits, dim=1)                              # (B, C, H, W)
    p     = log_p.exp()
    target_idx = target.unsqueeze(1).long()                           # (B, 1, H, W)
    log_p_t = log_p.gather(1, target_idx).squeeze(1)                  # (B, H, W)
    p_t     = p.gather(1, target_idx).squeeze(1)                      # (B, H, W)
    focal = (1.0 - p_t).clamp(min=0.0) ** gamma                       # (B, H, W)
    w = class_weights.to(logits.device)[target.long()]                # (B, H, W)
    loss = -(w * focal * log_p_t)                                     # (B, H, W)
    if ignore_index >= 0:
        valid = (target != ignore_index).float()
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)
    return loss.mean()


def compute_inverse_freq_class_weights(class_pixel_counts, smooth=1.0, normalize=True):
    """Inverse-frequency weights from per-class pixel counts (dict {class_id: count}).
    Returns tensor (max_class_id + 1,) where index 0 is background (usually class 0).

    smooth: add to denominator to prevent infinite weights for very rare classes.
    normalize: rescale weights so the mean equals 1.0.
    """
    max_c = max(class_pixel_counts.keys())
    counts = torch.tensor([class_pixel_counts.get(c, 0) for c in range(max_c + 1)], dtype=torch.float32)
    w = 1.0 / (counts + smooth)
    if normalize:
        # rescale so mean weight = 1 (keeps overall loss magnitude similar to vanilla CE)
        w = w * (len(w) / w.sum())
    return w
