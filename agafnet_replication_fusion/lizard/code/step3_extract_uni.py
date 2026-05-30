"""Step 3 — Extract UNI ViT-L tight + context features at AGAFNet instance locations.

Same fold->local_patch->pred_inst_id order as the label table (row-aligned).
Uses RAW Lizard images (not Reinhard) per UNI protocol.

  tight   = UNI token at the AGAFNet instance centroid (1024-d)
  context = mean of UNI tokens overlapping AGAFNet instance bbox + 1 margin (1024-d)

Output: /home/sbarua/fanseg/agafnet_mask_fusion/features/uni_agafnet.h5
        tight_features (N, 1024), context_features (N, 1024) aligned to nucleus table.
"""
import os, sys, time
import numpy as np
import h5py
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/sbarua/fanseg/code")
from encoder import UNIEncoder

AGAF_MASKS = '/home/sbarua/Lizard/agafnet_replication/fusion/features/instance_maps/fold{N}_inst_maps.npz'
RAW_IMGS   = '/mnt/storage1/Lizard/conic_patches/data/images.npy'
METADATA   = '/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy'
TABLE_H5   = '/home/sbarua/fanseg/agafnet_mask_fusion/features/agafnet_nucleus_table.h5'
OUT_H5     = '/home/sbarua/fanseg/agafnet_mask_fusion/features/uni_agafnet.h5'

PATCH_SIZE  = 256
TOKEN_PATCH = 16
TOKEN_GRID  = PATCH_SIZE // TOKEN_PATCH   # 16
GPU_BS      = 32
FOLDS = [1, 2, 3, 4, 5]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    meta = np.load(METADATA, allow_pickle=True).item()
    fa = meta['fold_assignments']
    raw = np.load(RAW_IMGS, mmap_mode='r')

    with h5py.File(TABLE_H5, 'r') as f:
        tab_fold  = f['fold_id'][:]
        tab_local = f['local_patch'][:]
        tab_pid   = f['pred_inst_id'][:]
    N = len(tab_fold)
    print(f"[init] table has {N:,} nuclei")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    encoder = UNIEncoder(freeze=True).to(device).eval()

    tight = np.zeros((N, 1024), dtype=np.float32)
    ctx   = np.zeros((N, 1024), dtype=np.float32)

    # Build per-patch work list: (fold, local, global, [(pid, cy, cx, ymin, xmin, ymax, xmax), ...])
    # following the SAME order as the table.
    print("[pass1] building per-patch metadata...")
    per_patch = []
    row = 0
    for fold in FOLDS:
        d = np.load(AGAF_MASKS.format(N=fold))
        fold_globals = sorted([k for k, v in fa.items() if v == fold])
        n_patches = len([k for k in d.files if k.startswith('inst_')])
        for local in range(n_patches):
            inst_pred = d[f'inst_{local}'].astype(np.int32)
            global_idx = fold_globals[local]
            nucs = []
            n_pred = int(inst_pred.max())
            for pid in range(1, n_pred + 1):
                m = inst_pred == pid
                if m.sum() < 5: continue
                ys, xs = np.where(m)
                nucs.append((pid, ys.mean(), xs.mean(),
                             ys.min(), xs.min(), ys.max(), xs.max(), row))
                row += 1
            if nucs:
                per_patch.append((fold, local, int(global_idx), nucs))
    assert row == N, f"metadata rows {row} != table {N}"
    print(f"  {len(per_patch)} patches, {N} nuclei")

    # Pass 2: UNI inference per patch-batch
    print("[pass2] UNI inference...")
    t0 = time.time()
    for ci in range(0, len(per_patch), GPU_BS):
        batch = per_patch[ci:ci+GPU_BS]
        imgs = np.stack([raw[p[2]] for p in batch]).astype(np.float32) / 255.0
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device, non_blocking=True)
        with torch.inference_mode():
            tokens = encoder(imgs_t)        # (B, 1024, 16, 16)
        tokens_np = tokens.cpu().numpy()
        for bi, (fold, local, gidx, nucs) in enumerate(batch):
            grid = tokens_np[bi]
            for (pid, cy, cx, ymin, xmin, ymax, xmax, r) in nucs:
                ty = min(int(cy) // TOKEN_PATCH, TOKEN_GRID - 1)
                tx = min(int(cx) // TOKEN_PATCH, TOKEN_GRID - 1)
                tight[r] = grid[:, ty, tx]
                ty1 = max(0, ymin // TOKEN_PATCH - 1)
                tx1 = max(0, xmin // TOKEN_PATCH - 1)
                ty2 = min(TOKEN_GRID, ymax // TOKEN_PATCH + 2)
                tx2 = min(TOKEN_GRID, xmax // TOKEN_PATCH + 2)
                ctx[r] = grid[:, ty1:ty2, tx1:tx2].reshape(1024, -1).mean(axis=-1)
        if (ci // GPU_BS) % 20 == 0:
            rate = (ci + len(batch)) / max(time.time()-t0, 1e-3)
            print(f"  patch {ci+len(batch)}/{len(per_patch)}  ({rate:.0f} patch/s)", flush=True)

    print(f"[done] UNI in {time.time()-t0:.1f}s")
    print(f"  tight norm mean={np.linalg.norm(tight, axis=1).mean():.3f}")
    print(f"  ctx   norm mean={np.linalg.norm(ctx, axis=1).mean():.3f}")

    with h5py.File(OUT_H5, 'w') as f:
        f.create_dataset('tight_features',   data=tight, compression='gzip', compression_opts=4)
        f.create_dataset('context_features', data=ctx,   compression='gzip', compression_opts=4)
        f.attrs['aligned_to'] = TABLE_H5
        f.attrs['source'] = 'AGAFNet instances + raw Lizard images (UNI ViT-L)'
    print(f"[saved] {OUT_H5}")


if __name__ == '__main__':
    main()
