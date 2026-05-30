"""Step 2 (PARALLEL) — Extract 71-d handcrafted features from AGAFNet instances.

Parallelized across CPU cores with multiprocessing. Each worker handles a
contiguous block of patches; results are written into the correct row slice so
the final array is identical to the single-core version and row-aligned with
agafnet_nucleus_table.h5.

Correctness guarantees:
  - Per-worker we replay the EXACT same fold->local->pid iteration as step1, so
    the row index each nucleus lands at is deterministic and matches the table.
  - A global row offset is precomputed per patch (cumulative nuclei count) so
    each worker knows exactly which rows it owns — no races, no appends.
  - Final assertions verify every row was filled and ordering matches the table.

Output: /home/sbarua/fanseg/agafnet_mask_fusion/features/handcrafted_agafnet.h5
"""
import os, sys, time
import numpy as np
import h5py
import cv2
from multiprocessing import Pool

sys.path.insert(0, "/home/sbarua/CoNSeP/fusion_classification_new")
from step1_extract_handcrafted import extract_nucleus_features, get_he_channels, FEATURE_NAMES

AGAF_MASKS    = '/home/sbarua/Lizard/agafnet_replication/fusion/features/instance_maps/fold{N}_inst_maps.npz'
REINHARD_IMGS = '/home/sbarua/fanseg/predictions/lizard/fusion_features_phase2_reinhard/images_reinhard.npy'
METADATA      = '/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy'
TABLE_H5      = '/home/sbarua/fanseg/agafnet_mask_fusion/features/agafnet_nucleus_table.h5'
OUT_H5        = '/home/sbarua/fanseg/agafnet_mask_fusion/features/handcrafted_agafnet.h5'

FOLDS = [1, 2, 3, 4, 5]
N_WORKERS = 32
assert len(FEATURE_NAMES) == 71

# Globals each worker initializes once (avoids re-loading per task)
_REINHARD = None
_NPZ_CACHE = {}     # per-worker cache: fold -> loaded NpzFile


def _init_worker():
    """Each worker process loads the shared read-only Reinhard images once."""
    global _REINHARD, _NPZ_CACHE
    _REINHARD = np.load(REINHARD_IMGS, mmap_mode='r')
    _NPZ_CACHE = {}


def _get_npz(fold):
    """Return the AGAFNet npz for a fold, cached per worker (open zip once)."""
    global _NPZ_CACHE
    if fold not in _NPZ_CACHE:
        _NPZ_CACHE[fold] = np.load(AGAF_MASKS.format(N=fold))
    return _NPZ_CACHE[fold]


def _process_patch(task):
    """Extract features for ALL nuclei in one patch.

    task = (fold, local, global_idx, [(pid, row_index), ...])
    Returns list of (row_index, feature_vector(71,)).
    Row indices are the GLOBAL positions in the output array (precomputed).
    """
    fold, local, global_idx, nuc_list = task
    d = _get_npz(fold)
    inst_pred = d[f'inst_{local}'].astype(np.int32)
    image_rgb = np.ascontiguousarray(_REINHARD[global_idx])
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h_chan, e_chan = get_he_channels(image_rgb)

    out = []
    for pid, row in nuc_list:
        mask = inst_pred == pid
        feats = extract_nucleus_features(mask, image_rgb, h_chan, e_chan, gray)
        out.append((row, np.asarray(feats, dtype=np.float32)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='only first 200 patches, verify correctness')
    ap.add_argument('--workers', type=int, default=N_WORKERS)
    args = ap.parse_args()

    # ---- Load table to know exact ordering + count ----
    with h5py.File(TABLE_H5, 'r') as f:
        tab_fold  = f['fold_id'][:]
        tab_local = f['local_patch'][:]
        tab_pid   = f['pred_inst_id'][:]
    N = len(tab_fold)
    print(f"[init] table has {N:,} nuclei")

    # ---- Build per-patch task list with PRECOMPUTED global row indices ----
    # Replay the EXACT step1 iteration order, assigning each nucleus its row.
    meta = np.load(METADATA, allow_pickle=True).item()
    fa = meta['fold_assignments']
    tasks = []
    row = 0
    t0 = time.time()
    for fold in FOLDS:
        d = np.load(AGAF_MASKS.format(N=fold))
        fold_globals = sorted([k for k, v in fa.items() if v == fold])
        n_patches = len([k for k in d.files if k.startswith('inst_')])
        for local in range(n_patches):
            inst_pred = d[f'inst_{local}'].astype(np.int32)
            global_idx = fold_globals[local]
            nuc_list = []
            n_pred = int(inst_pred.max())
            for pid in range(1, n_pred + 1):
                m = inst_pred == pid
                if m.sum() < 5:           # SAME threshold as step1
                    continue
                # verify this row matches the table ordering
                assert tab_fold[row] == fold and tab_local[row] == local and tab_pid[row] == pid, \
                    f"row {row} misaligned: table=({tab_fold[row]},{tab_local[row]},{tab_pid[row]}) " \
                    f"current=({fold},{local},{pid})"
                nuc_list.append((pid, row))
                row += 1
            if nuc_list:
                tasks.append((fold, local, int(global_idx), nuc_list))
    assert row == N, f"task-building rows {row} != table {N}"
    print(f"[init] {len(tasks)} patches, {N} nuclei, row mapping verified ({time.time()-t0:.1f}s)")

    # ---- Smoke mode: only first 200 patches ----
    if args.smoke:
        tasks = tasks[:200]
        smoke_rows = set(r for _,_,_,nl in tasks for _, r in nl)
        print(f"[SMOKE] {len(tasks)} patches, {len(smoke_rows)} nuclei")

    # ---- Parallel extraction ----
    feats = np.zeros((N, 71), dtype=np.float32)
    filled = np.zeros(N, dtype=bool)
    print(f"[run] launching {args.workers} workers...")
    t0 = time.time()
    done_patches = 0
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_process_patch, tasks, chunksize=8):
            for row_idx, fvec in result:
                feats[row_idx] = fvec
                filled[row_idx] = True
            done_patches += 1
            if done_patches % 500 == 0:
                el = time.time() - t0
                rate = done_patches / el
                eta = (len(tasks) - done_patches) / max(rate, 1e-9) / 60
                print(f"  {done_patches}/{len(tasks)} patches  ({rate:.0f} patch/s, ETA {eta:.1f} min)", flush=True)

    # ---- Correctness checks ----
    if args.smoke:
        sm = np.array(sorted(smoke_rows))
        print(f"\n[SMOKE verify] rows filled in smoke set: {filled[sm].all()}  ({filled[sm].sum()}/{len(sm)})")
        sub = feats[sm]
        print(f"[SMOKE verify] NaN={int(np.isnan(sub).any(axis=1).sum())}  "
              f"all-zero={int((np.abs(sub).sum(axis=1)<1e-12).sum())}")
        print(f"[SMOKE verify] norm mean={np.linalg.norm(sub,axis=1).mean():.3g}")
        print("[SMOKE] OK — not saving. Re-run without --smoke for full extraction.")
        return

    print(f"\n[verify] all rows filled: {filled.all()}  ({filled.sum()}/{N})")
    assert filled.all(), f"MISSING {(~filled).sum()} rows — extraction incomplete!"
    n_nan = int(np.isnan(feats).any(axis=1).sum())
    n_inf = int(np.isinf(feats).any(axis=1).sum())
    n_zero = int((np.abs(feats).sum(axis=1) < 1e-12).sum())
    print(f"[verify] NaN rows={n_nan}  inf rows={n_inf}  all-zero rows={n_zero}")
    norms = np.linalg.norm(feats, axis=1)
    print(f"[verify] norm mean={norms.mean():.3g}  min={norms.min():.3g}  max={norms.max():.3g}")

    # ---- Save ----
    with h5py.File(OUT_H5, 'w') as f:
        f.create_dataset('features', data=feats, compression='gzip', compression_opts=4)
        f.create_dataset('feature_names', data=np.array(FEATURE_NAMES, dtype=h5py.string_dtype()))
        f.attrs['aligned_to'] = TABLE_H5
        f.attrs['source'] = 'AGAFNet instances + Reinhard Lizard images (parallel extraction)'
        f.attrs['n_workers'] = N_WORKERS
    print(f"\n[done] {feats.shape} in {(time.time()-t0)/60:.1f} min")
    print(f"[saved] {OUT_H5}")


if __name__ == '__main__':
    main()
