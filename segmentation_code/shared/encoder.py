"""UNI encoder wrapper. Frozen by default. Returns (B, 1024, H/16, W/16) spatial tokens."""
import torch
import torch.nn as nn
import timm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class UNIEncoder(nn.Module):
    EMBED_DIM = 1024
    PATCH_SIZE = 16

    def __init__(self, freeze=True):
        super().__init__()
        self.vit = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    def train(self, mode=True):
        # keep vit in eval mode even if outer module is set to train
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x):
        # x: (B, 3, H, W) in [0, 1]
        x = (x - self.mean) / self.std
        feats = self.vit.forward_features(x)        # (B, 1+N, D)
        patch_tokens = feats[:, 1:, :]              # drop CLS → (B, N, D)
        B, N, D = patch_tokens.shape
        H = W = int(N**0.5)
        spatial = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        return spatial  # (B, 1024, H/16, W/16)

    def forward_multiscale(self, x, depths=(5, 11, 17, 23)):
        """Return spatial features (B, D, H/16, W/16) from `depths` ViT blocks (0-indexed)."""
        x = (x - self.mean) / self.std
        # timm get_intermediate_layers(n=list, reshape=True, norm=True) → list of (B, D, H, W)
        return self.vit.get_intermediate_layers(
            x, n=list(depths), reshape=True, return_prefix_tokens=False, norm=True
        )
