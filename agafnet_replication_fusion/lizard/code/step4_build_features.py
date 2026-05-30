"""Step 4 — Build per-fold combined feature H5s for the AGAFNet-mask fusion experiment.

Combines:
  hand   (71)   from handcrafted_agafnet.h5  -> signed_log1p + per-fold StandardScaler
  tight  (1024) from uni_agafnet.h5          -> per-fold StandardScaler (no PCA)
  ctx    (1024) from uni_agafnet.h5          -> per-fold StandardScaler (no PCA)
  + metadata from agafnet_nucleus_table.h5   (fold, global_patch, pred_inst_id,
                                              centroids, gt_class, agafnet_class)

Per fold N (test fold), fit scalers on TRAIN-labeled rows (fold != N & gt_class > 0),
transform all rows. Output one H5 per fold.

Output: features/fusion_features_fold{N}.h5  (N=1..5)
"""
import os, time, pickle
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

FEAT_DIR  = '/home/sbarua/fanseg/agafnet_mask_fusion/features'
TABLE_H5  = f'{FEAT_DIR}/agafnet_nucleus_table.h5'
HAND_H5   = f'{FEAT_DIR}/handcrafted_agafnet.h5'
UNI_H5    = f'{FEAT_DIR}/uni_agafnet.h5'
OUT_TPL   = f'{FEAT_DIR}/fusion_features_fold{{N}}.h5'
SCALER_PKL= f'{FEAT_DIR}/scalers_per_fold.pkl'
FOLDS = [1, 2, 3, 4, 5]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    print("[load] table, handcrafted, UNI...")
    with h5py.File(TABLE_H5, 'r') as f:
        fold_id  = f['fold_id'][:].astype(np.int32)
        patch    = f['local_patch'][:].astype(np.int32)
        gpatch   = f['global_patch'][:].astype(np.int32)
        pid      = f['pred_inst_id'][:].astype(np.int32)
        cy       = f['centroid_y'][:].astype(np.float32)
        cx       = f['centroid_x'][:].astype(np.float32)
        gt_class = f['gt_class'][:].astype(np.int32)
        agaf_cls = f['agafnet_class'][:].astype(np.int32)
    with h5py.File(HAND_H5, 'r') as f:
        hand = f['features'][:].astype(np.float64)
    with h5py.File(UNI_H5, 'r') as f:
        tight = f['tight_features'][:]
        ctx   = f['context_features'][:]
    N = len(fold_id)
    assert len(hand) == N == len(tight) == len(ctx), "row count mismatch across feature files!"
    print(f"  N={N:,}  hand={hand.shape}  tight={tight.shape}  ctx={ctx.shape}")

    # signed_log1p on hand (once, before per-fold scaling)
    hand_log = signed_log1p(hand)
    assert not np.isnan(hand_log).any() and not np.isinf(hand_log).any()

    scalers = {}
    for test_fold in FOLDS:
        train_mask = (fold_id != test_fold) & (gt_class > 0)
        print(f"\n=== fold {test_fold}: fit on {int(train_mask.sum()):,} train-labeled rows ===")
        sc_h = StandardScaler().fit(hand_log[train_mask])
        sc_t = StandardScaler().fit(tight[train_mask])
        sc_c = StandardScaler().fit(ctx[train_mask])
        hand_s  = sc_h.transform(hand_log).astype(np.float32)
        tight_s = sc_t.transform(tight).astype(np.float32)
        ctx_s   = sc_c.transform(ctx).astype(np.float32)
        for a, nm in [(hand_s,'hand'),(tight_s,'tight'),(ctx_s,'ctx')]:
            assert not np.isnan(a).any(), f"{nm} NaN after scaling!"

        out = OUT_TPL.format(N=test_fold)
        with h5py.File(out, 'w') as f:
            f.create_dataset('hand_scaled', data=hand_s,  compression='gzip', compression_opts=4)
            f.create_dataset('tight_std',   data=tight_s, compression='gzip', compression_opts=4)
            f.create_dataset('ctx_std',     data=ctx_s,   compression='gzip', compression_opts=4)
            f.create_dataset('fold_id',  data=fold_id)
            f.create_dataset('patch_id', data=patch)         # LOCAL patch within fold
            f.create_dataset('global_patch', data=gpatch)
            f.create_dataset('nuc_id',   data=pid)
            f.create_dataset('gt_class', data=gt_class)
            f.create_dataset('agafnet_class', data=agaf_cls)
            f.create_dataset('centroid_x', data=cx)
            f.create_dataset('centroid_y', data=cy)
            f.attrs['test_fold'] = test_fold
        print(f"  saved {out}  (tight_std norm mean={np.linalg.norm(tight_s,axis=1).mean():.2f})")
        scalers[test_fold] = {'hand': (sc_h.mean_.tolist(), sc_h.scale_.tolist())}

    with open(SCALER_PKL, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"\n[done] 5 fold feature files + scalers saved")


if __name__ == '__main__':
    main()
