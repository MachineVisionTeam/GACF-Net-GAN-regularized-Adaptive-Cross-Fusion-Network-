"""DPT-style decoder for FAN-Seg Phase 1.

Takes 4 multi-scale ViT feature maps (each B,1024,16,16) and produces full-resolution
HV + Sem3 outputs (B, 2/3, 256, 256). Architecture follows Ranftl et al. DPT 2021,
adapted to UNI ViT-L's 16×16 token grid at 256×256 input.

Reassemble scales (relative to 16x16 base):
  layer 6  → 64×64  (upsample ×4 — high-res, early features)
  layer 12 → 32×32  (upsample ×2)
  layer 18 → 16×16  (no scale change)
  layer 24 →  8×8   (downsample ×2 — low-res, abstract features)

Then progressive RefineNet-style fusion 8→16→32→64 → final upsample to 256.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.act = nn.ReLU(True)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        return x + out


class FusionBlock(nn.Module):
    """Combine a low-res path with a higher-res skip, then upsample by 2."""
    def __init__(self, ch):
        super().__init__()
        self.rcu1 = ResidualConvUnit(ch)
        self.rcu2 = ResidualConvUnit(ch)
        self.out  = nn.Conv2d(ch, ch, 1, bias=True)

    def forward(self, low, skip=None):
        low = F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            low = low + self.rcu1(skip)
        low = self.rcu2(low)
        return self.out(low)


class Reassemble(nn.Module):
    """Project ViT spatial tokens to a chosen resolution + channel count."""
    def __init__(self, in_dim=1024, out_ch=256, scale=1):
        """scale > 1 → upsample (transposed conv);  scale < 1 → downsample (strided conv);  scale = 1 → 1×1."""
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_ch, 1)
        if scale == 4:
            self.resample = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=4)
        elif scale == 2:
            self.resample = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)
        elif scale == 1:
            self.resample = nn.Identity()
        elif scale == 0.5:
            self.resample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        else:
            raise ValueError(f"unsupported scale {scale}")

    def forward(self, x):
        return self.resample(self.proj(x))


class DPTDecoder(nn.Module):
    """4-scale DPT decoder. Heads:
        HV head   — 2-ch  HoVer regression  (always)
        Sem3 head — 3-ch  bg/interior/boundary classification (always)
        Type head — (num_type_classes)-ch per-pixel nucleus class (OPTIONAL).
                    Enabled by passing num_type_classes=K (K = bg(1) + nucleus classes).
                    For Lizard pass num_type_classes=7 (bg + 6).

    Output: (hv, sem3) if no type head; (hv, sem3, type_logits) if type head enabled.
    """

    def __init__(self, in_dim=1024, ch=256, num_type_classes=None):
        super().__init__()
        self.num_type_classes = num_type_classes
        # 4 reassembles
        self.re_l1 = Reassemble(in_dim, ch, scale=4)
        self.re_l2 = Reassemble(in_dim, ch, scale=2)
        self.re_l3 = Reassemble(in_dim, ch, scale=1)
        self.re_l4 = Reassemble(in_dim, ch, scale=0.5)
        # Fusion
        self.fuse4 = FusionBlock(ch); self.fuse3 = FusionBlock(ch)
        self.fuse2 = FusionBlock(ch); self.fuse1 = FusionBlock(ch); self.fuse0 = FusionBlock(ch)

        self.head_shared = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(True),
        )
        self.hv_head   = nn.Conv2d(ch // 2, 2, 1)
        self.sem3_head = nn.Conv2d(ch // 2, 3, 1)
        if num_type_classes is not None and num_type_classes > 0:
            self.type_head = nn.Conv2d(ch // 2, num_type_classes, 1)
        else:
            self.type_head = None

    def forward(self, ms_feats):
        f1, f2, f3, f4 = ms_feats
        l1 = self.re_l1(f1); l2 = self.re_l2(f2); l3 = self.re_l3(f3); l4 = self.re_l4(f4)
        x = self.fuse4(l4, skip=l3); x = self.fuse3(x, skip=l2); x = self.fuse2(x, skip=l1)
        x = self.fuse1(x); x = self.fuse0(x)
        feat = self.head_shared(x)
        hv = torch.tanh(self.hv_head(feat))
        sem3 = self.sem3_head(feat)
        if self.type_head is not None:
            type_logits = self.type_head(feat)
            return hv, sem3, type_logits
        return hv, sem3
