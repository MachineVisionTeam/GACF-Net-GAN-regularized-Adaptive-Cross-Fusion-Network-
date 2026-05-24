"""Whole-image PatchGAN discriminator for FAN-Seg Phase 2 (Mahmood-inspired refactor).

Design:
  Input  : 8-channel image  (RGB 3 + HV 2 + Sem3 prob 3)  at full 256x256
  Output : (B, 1, 16, 16) PatchGAN logits — averaged in the loss

Why 8-ch CONTINUOUS:
  Phase 2 v1/v2 collapsed because D was fed BINARY GT vs SOFT pred. D solved
  the task by detecting the 0.0/1.0 vs 0.x value distribution rather than by
  judging shape. Mahmood TMI 2020 fixes this by training D on a continuous
  regression target. Here, both REAL (GT HV + GT Sem3 one-hot) and FAKE
  (predicted HV + predicted softmax(Sem3)) have the same continuous range —
  D must judge the joint distribution of HV and Sem3 conditioned on RGB.

Backbone: PatchGAN 256 -> 128 -> 64 -> 32 -> 16
  Conv4x4 stride 2, LeakyReLU(0.2), spectral-norm on every conv.
  No batch norm (incompatible with single-image cond + spectral norm spirit).
  Final 1x1 conv -> 1-ch patch logits.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class _SNConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, pad=1, bias=True):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=bias))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class FanSegDiscriminator(nn.Module):
    """Whole-image PatchGAN, continuous 8-channel input.

    Args:
        in_ch:    8 (RGB 3 + HV 2 + Sem3 3)
        channels: per-stage feature channels
    """
    def __init__(self, in_ch=8, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        # 256 -> 128 -> 64 -> 32 -> 16
        self.block1 = _SNConv(in_ch, c1)
        self.block2 = _SNConv(c1, c2)
        self.block3 = _SNConv(c2, c3)
        self.block4 = _SNConv(c3, c4, stride=1)   # keep 32 spatial here
        # Final 1x1 -> patch logits
        self.head = spectral_norm(nn.Conv2d(c4, 1, kernel_size=1))

    def forward(self, x):
        """x: (B, 8, 256, 256) — concat(RGB, HV, Sem3_prob) along ch.
        Returns patch logits (B, 1, 32, 32)."""
        h = self.block1(x)   # 128
        h = self.block2(h)   # 64
        h = self.block3(h)   # 32
        h = self.block4(h)   # 32 (stride 1)
        return self.head(h)  # (B, 1, 32, 32)


if __name__ == "__main__":
    D = FanSegDiscriminator()
    n = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"D trainable params: {n/1e6:.2f}M")
    x = torch.randn(2, 8, 256, 256)
    s = D(x)
    print(f"output shape: {s.shape}, range=[{s.min().item():.3f}, {s.max().item():.3f}]")
    s.mean().backward()
    print("backward OK")
