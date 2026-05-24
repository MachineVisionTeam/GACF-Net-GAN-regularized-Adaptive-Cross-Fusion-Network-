"""PanNuke loader for BHF-Net.

Reads pre-patched PanNuke from
  /home/sbarua/Panuke/hover_net_x_panuke/data/{images,masks}/fold{1,2,3}/(images|labels).npy

PanNuke uses 3-fold CV. For test fold N: train = concat(other 2 folds), test = fold N.

Output format matches LizardDataset.
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, "/home/sbarua/CoNSeP/segmentation_replicate/anseg_cgan/code")
from utils.gt_generation import compute_hv_maps, compute_3class_semantic

sys.path.insert(0, "/home/sbarua/fanseg/code")
from augmentations import hed_jitter, elastic_transform


IMAGES_ROOT = "/home/sbarua/Panuke/hover_net_x_panuke/data/images"
LABELS_ROOT = "/home/sbarua/Panuke/hover_net_x_panuke/data/masks"


class PanNukeDataset(Dataset):
    """PanNuke patches loader. 5 nuclear classes (PanNuke standard)."""

    NUM_CLASSES = 5
    CLASS_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    NUM_FOLDS = 3   # PanNuke standard

    def __init__(self, split, mode, augment=False,
                 p_hed=0.7, hed_sigma=0.05, hed_bias=0.05,
                 p_elastic=0.5, elastic_alpha=120.0, elastic_sigma=8.0):
        assert mode in ("train", "test")
        assert split in (1, 2, 3)
        self.split = split
        self.mode  = mode
        self.augment = augment
        self.p_hed     = p_hed
        self.hed_sigma = hed_sigma
        self.hed_bias  = hed_bias
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        if mode == "test":
            folds = [split]
        else:
            folds = [f for f in [1, 2, 3] if f != split]

        # Memory-map per fold, build a (fold_idx, local_idx) → flat-idx mapping
        self._fold_imgs = []   # list of memory-mapped arrays
        self._fold_lbls = []
        self._index = []       # list of (fold_pos, local_idx)
        for f in folds:
            imgs = np.load(os.path.join(IMAGES_ROOT, f"fold{f}", "images.npy"), mmap_mode="r")
            lbls = np.load(os.path.join(LABELS_ROOT, f"fold{f}", "labels.npy"), mmap_mode="r")
            fold_pos = len(self._fold_imgs)
            self._fold_imgs.append(imgs)
            self._fold_lbls.append(lbls)
            for i in range(len(imgs)):
                self._index.append((fold_pos, i))
        print(f"  [PanNuke] split={split} mode={mode}: {len(self._index)} patches from folds {folds} "
              f"(augment={augment})")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fold_pos, local_idx = self._index[idx]
        # PanNuke images are float64 in [0, 255] — normalize to [0, 1] float32
        img = self._fold_imgs[fold_pos][local_idx].astype(np.float32) / 255.0   # (H, W, 3)
        lbl = self._fold_lbls[fold_pos][local_idx]                               # (H, W, 2) int64
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
            "patch_idx": idx,
        }


if __name__ == "__main__":
    # Smoke test all 3 splits
    for split in [1, 2, 3]:
        for mode in ("train", "test"):
            ds = PanNukeDataset(split=split, mode=mode, augment=False)
            sample = ds[0]
            print(f"    split={split} {mode}: img {tuple(sample['image'].shape)}  "
                  f"hv {tuple(sample['hv'].shape)}  sem unique {sorted(set(sample['sem'].flatten().tolist()))}  "
                  f"cls unique {sorted(set(sample['cls'].flatten().tolist()))}")
