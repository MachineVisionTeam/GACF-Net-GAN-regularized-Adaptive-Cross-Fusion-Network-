"""Lizard (CoNIC pre-patched) loader for FAN-Seg.

Reads /mnt/storage1/Lizard/conic_patches/data/{images.npy, labels.npy} (the official
4,981-patch CoNIC dataset). Splits via the researcher's existing 5-fold patient-wise
assignment at /mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy.

Output dict matches the CoNSeP loader so train_phase1_v2.py works unchanged:
  image    (3, H, W) float32 in [0, 1]
  hv       (2, H, W) float32 in [-1, 1]
  sem      (H, W)    int64 {0=bg, 1=fg-interior, 2=boundary}  (HoVer-Net Sem3)
  cls      (H, W)    int64 {0=bg, 1..6=Lizard class}
  fg_mask  (H, W)    float32 (1 = nucleus pixel)
  inst_map (H, W)    int64
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/home/sbarua/CoNSeP/segmentation_replicate/anseg_cgan/code")
from utils.gt_generation import compute_hv_maps, compute_3class_semantic

from augmentations import hed_jitter, elastic_transform


CONIC_ROOT     = "/mnt/storage1/Lizard/conic_patches/data"
METADATA_PATH  = "/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy"


def _load_fold_index(split, mode):
    """Return list of patch indices belonging to (split, mode).

    split: 1..5
    mode:  'train' (all patches NOT in this fold) or 'test' (patches in this fold)
    """
    meta = np.load(METADATA_PATH, allow_pickle=True).item()
    fa = meta["fold_assignments"]                       # dict: patch_idx → fold_id
    if mode == "test":
        return sorted([k for k, v in fa.items() if v == split])
    elif mode == "train":
        return sorted([k for k, v in fa.items() if v != split])
    raise ValueError(f"mode must be train|test, got {mode}")


class LizardDataset(Dataset):
    """Lizard CoNIC patches loader. 6 nuclear classes."""

    NUM_CLASSES = 6   # 1=Neutrophil, 2=Epithelial, 3=Lymphocyte, 4=Plasma, 5=Eosinophil, 6=Connective
    CLASS_NAMES = ["Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Eosinophil", "Connective"]

    def __init__(self, split, mode, augment=False,
                 p_hed=0.7, hed_sigma=0.05, hed_bias=0.05,
                 p_elastic=0.5, elastic_alpha=120.0, elastic_sigma=8.0):
        assert mode in ("train", "test")
        self.split = split
        self.mode  = mode
        self.augment = augment
        self.p_hed     = p_hed
        self.hed_sigma = hed_sigma
        self.hed_bias  = hed_bias
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        # mmap the master arrays — index into them lazily
        self._images = np.load(os.path.join(CONIC_ROOT, "images.npy"), mmap_mode="r")
        self._labels = np.load(os.path.join(CONIC_ROOT, "labels.npy"), mmap_mode="r")
        self.indices = _load_fold_index(split, mode)
        print(f"  [Lizard] split={split} mode={mode}: {len(self.indices)} patches "
              f"from {self._images.shape[0]} total (augment={augment})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        patch_idx = self.indices[idx]
        img = self._images[patch_idx].astype(np.float32) / 255.0    # (H, W, 3)
        lbl = self._labels[patch_idx]                                 # (H, W, 2)
        inst_map = lbl[..., 0].astype(np.int32)
        type_map = lbl[..., 1].astype(np.int32)

        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy(); inst_map = inst_map[:, ::-1].copy(); type_map = type_map[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :, :].copy(); inst_map = inst_map[::-1, :].copy(); type_map = type_map[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k, axes=(0, 1)).copy()
                inst_map = np.rot90(inst_map, k, axes=(0, 1)).copy()
                type_map = np.rot90(type_map, k, axes=(0, 1)).copy()
            if np.random.rand() < self.p_elastic:
                img, inst_map, type_map = elastic_transform(
                    img, inst_map, type_map,
                    alpha=self.elastic_alpha, sigma=self.elastic_sigma,
                )
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
            "patch_idx": patch_idx,
        }
