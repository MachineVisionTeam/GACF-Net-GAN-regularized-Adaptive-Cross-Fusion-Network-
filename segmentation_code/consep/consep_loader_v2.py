"""CoNSeP loader v2 with HED jitter + elastic deformation on top of flip/rotate.

Mirrors the existing loader's output dict but adds two stronger augmentations.
HV maps are computed AFTER all geometric warps (so they stay correct).
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/home/sbarua/CoNSeP/segmentation_replicate/anseg_cgan/code")
from utils.gt_generation import compute_hv_maps, compute_3class_semantic

from augmentations import hed_jitter, elastic_transform


class CoNSePDatasetV2(Dataset):
    """Same output schema as anseg_cgan's CoNSePDataset, with HED + elastic added."""

    def __init__(self, fold_dir, augment=False,
                 p_hed=0.7, hed_sigma=0.05, hed_bias=0.05,
                 p_elastic=0.5, elastic_alpha=120.0, elastic_sigma=8.0):
        self.images = np.load(os.path.join(fold_dir, "images.npy"))
        self.labels = np.load(os.path.join(fold_dir, "labels.npy"))
        self.augment = augment
        self.p_hed     = p_hed
        self.hed_sigma = hed_sigma
        self.hed_bias  = hed_bias
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        assert self.images.shape[:3] == self.labels.shape[:3]
        print(f"  [v2] Loaded {len(self.images)} patches from {fold_dir} (augment={augment})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        lbl = self.labels[idx]
        inst_map = lbl[..., 0].astype(np.int32)
        type_map = lbl[..., 1].astype(np.int32)

        if self.augment:
            # Geometric: flip + rotate (same as v1)
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                inst_map = inst_map[:, ::-1].copy()
                type_map = type_map[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :, :].copy()
                inst_map = inst_map[::-1, :].copy()
                type_map = type_map[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k, axes=(0, 1)).copy()
                inst_map = np.rot90(inst_map, k, axes=(0, 1)).copy()
                type_map = np.rot90(type_map, k, axes=(0, 1)).copy()

            # New: elastic deformation
            if np.random.rand() < self.p_elastic:
                img, inst_map, type_map = elastic_transform(
                    img, inst_map, type_map,
                    alpha=self.elastic_alpha, sigma=self.elastic_sigma,
                )

            # New: HED color jitter
            if np.random.rand() < self.p_hed:
                img = hed_jitter(img, sigma=self.hed_sigma, bias=self.hed_bias)

        hv      = compute_hv_maps(inst_map)
        sem     = compute_3class_semantic(inst_map)
        fg_mask = (inst_map > 0).astype(np.float32)

        return {
            "image":    torch.from_numpy(img).permute(2, 0, 1).float(),
            "hv":       torch.from_numpy(hv).permute(2, 0, 1).float(),
            "sem":      torch.from_numpy(sem).long(),
            "cls":      torch.from_numpy(type_map).long(),
            "fg_mask":  torch.from_numpy(fg_mask).float(),
            "inst_map": torch.from_numpy(inst_map).long(),
        }
