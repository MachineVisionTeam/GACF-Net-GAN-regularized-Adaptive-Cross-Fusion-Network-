"""CoNSeP loader for BHF-Net (mirror of LizardDataset).

Reads pre-patched CoNSeP folds from
  /home/sbarua/CoNSeP/segmentation_replicate/hover_net_x_finetune/data/
already in (N, 256, 256, 3) uint8 images + (N, 256, 256, 2) int32 labels format.
Class remap to 4 (REMAP={1:1,2:2,3:3,4:3,5:4,6:1,7:3}) was applied at prep stage.

Output dict format matches LizardDataset so existing train/inference scripts work
unchanged when their `from lizard_loader import LizardDataset` is swapped for
`from consep_loader import CoNSePDataset`.
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset

# Same upstream utilities as Lizard
sys.path.insert(0, "/home/sbarua/CoNSeP/segmentation_replicate/anseg_cgan/code")
from utils.gt_generation import compute_hv_maps, compute_3class_semantic

sys.path.insert(0, "/home/sbarua/fanseg/code")
from augmentations import hed_jitter, elastic_transform


CONSEP_ROOT = "/home/sbarua/CoNSeP/segmentation_replicate/hover_net_x_finetune/data"


class CoNSePDataset(Dataset):
    """CoNSeP patches loader. 4 nuclear classes after standard remap."""

    NUM_CLASSES = 4
    CLASS_NAMES = ["Miscellaneous", "Inflammatory", "Epithelial", "Spindle-shaped"]

    def __init__(self, split, mode, augment=False,
                 p_hed=0.7, hed_sigma=0.05, hed_bias=0.05,
                 p_elastic=0.5, elastic_alpha=120.0, elastic_sigma=8.0):
        assert mode in ("train", "test")
        assert split in (1, 2, 3, 4, 5)
        self.split = split
        self.mode  = mode
        self.augment = augment
        self.p_hed     = p_hed
        self.hed_sigma = hed_sigma
        self.hed_bias  = hed_bias
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        fold_dir = os.path.join(CONSEP_ROOT, f"fold{split}_test" if mode == "test" else f"fold{split}")
        self._images = np.load(os.path.join(fold_dir, "images.npy"), mmap_mode="r")
        self._labels = np.load(os.path.join(fold_dir, "labels.npy"), mmap_mode="r")
        assert self._images.shape[0] == self._labels.shape[0], \
            f"image/label mismatch: {self._images.shape} vs {self._labels.shape}"
        print(f"  [CoNSeP] split={split} mode={mode}: {len(self._images)} patches "
              f"from {fold_dir} (augment={augment})")

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img = self._images[idx].astype(np.float32) / 255.0    # (H, W, 3)
        lbl = self._labels[idx]                                # (H, W, 2)
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
    # Smoke test all 5 splits
    for split in [1, 2, 3, 4, 5]:
        for mode in ("train", "test"):
            ds = CoNSePDataset(split=split, mode=mode, augment=False)
            sample = ds[0]
            print(f"    split={split} {mode}: img {tuple(sample['image'].shape)}  "
                  f"hv {tuple(sample['hv'].shape)}  sem unique {sorted(set(sample['sem'].flatten().tolist()))}  "
                  f"cls unique {sorted(set(sample['cls'].flatten().tolist()))}")
