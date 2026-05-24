"""Lightweight decoder for Phase 0: 4 transposed-conv stages 16→256 + HV + Sem3 heads."""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(self.up(x)))


class TinyDecoder(nn.Module):
    """16x16x1024 → 256x256x{2,3} via 4 upsampling stages + two task heads."""

    def __init__(self, in_dim=1024, channels=(512, 256, 128, 64)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up1 = UpStage(in_dim, c1)   # 16  → 32
        self.up2 = UpStage(c1, c2)       # 32  → 64
        self.up3 = UpStage(c2, c3)       # 64  → 128
        self.up4 = UpStage(c3, c4)       # 128 → 256
        self.hv_head   = nn.Sequential(nn.Conv2d(c4, c4, 3, padding=1), nn.ReLU(True), nn.Conv2d(c4, 2, 1))
        self.sem3_head = nn.Sequential(nn.Conv2d(c4, c4, 3, padding=1), nn.ReLU(True), nn.Conv2d(c4, 3, 1))

    def forward(self, x):
        f = self.up4(self.up3(self.up2(self.up1(x))))
        hv = torch.tanh(self.hv_head(f))    # (B, 2, H, W) ∈ [-1, 1]
        sem3 = self.sem3_head(f)            # (B, 3, H, W) raw logits
        return hv, sem3
