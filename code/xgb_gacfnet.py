"""XGBoost handoff for GACF-Net — supports the full ablation table.

Inputs (per fold):
  raw fixed features (hand 71 + tight 1024 + ctx 1024 = 2119-d)  from fusion_features_uni_raw
  graph_emb (192-d)  + cross_emb (192-d)                          from gacfnet_embeddings

Per ablation row in the spec:
  Row 1: raw concat only                                            -> 2119-d
  Row 2: raw + graph                                                -> 2311-d
  Row 3: raw + graph + cross (HEADLINE)                             -> 2503-d
  Row 4: raw + graph + Hadamard cross only (no sum-pool/sqrt/L2)   -> ablation for MFB tricks
  Row 6: raw + graph + RANDOM projection of same size (NULL ctrl)   -> 2503-d

XGBoost settings per the cross-term spec:
  colsample_bytree  = 0.8
  colsample_bylevel = 1.0    (OFF — protects multiplicative signal)
  colsample_bynode  = 1.0    (OFF — protects multiplicative signal)
  reg_lambda        = 1.5    (heavier reg for heavy-tailed cross-terms)
"""
import os, sys, time, json, argparse
import numpy as np
import h5py
import xgboost as xgb
from sklearn.metrics import f1_score


def load_fold_data(dataset, fold):
    """Returns dict with raw features + gacfnet embeddings."""
    base = f'/home/sbarua/fanseg/predictions/{dataset}/fusion_features_uni_raw'
    with h5py.File(os.path.join(base, f'fusion_features_fold{fold}.h5'), 'r') as f:
        h = f['hand_scaled'][:]; t = f['tight_std'][:]; c = f['ctx_std'][:]
        fold_id  = f['fold_id'][:]
        patch_id = f['patch_id'][:]
        nuc_id   = f['nuc_id'][:]
        gt_class = f['gt_class'][:]
    raw = np.concatenate([h, t, c], axis=1).astype(np.float32)

    emb_path = os.path.join(base, 'gacfnet_embeddings', f'embeddings_fold{fold}.npz')
    d = np.load(emb_path)
    graph_emb = d['graph_emb'].astype(np.float32)
    cross_emb = d['cross_emb'].astype(np.float32)
    # Sanity: row alignment via shapes
    assert len(graph_emb) == len(raw), f"emb {len(graph_emb)} != raw {len(raw)}"
    return {
        'raw': raw, 'graph': graph_emb, 'cross': cross_emb,
        'fold_id': fold_id, 'patch_id': patch_id, 'nuc_id': nuc_id,
        'gt_class': gt_class,
    }


def build_features(d, mode):
    """Build the X matrix per ablation mode.

    raw = hand_scaled(71) + tight_std(1024) + ctx_std(1024) = 2119-d concat
    hand = hand_scaled(71) only — handcrafted morphology baseline, no deep features
    deep = tight_std(1024) + ctx_std(1024) = 2048-d — UNI deep features only, no hand
    """
    raw, g, c = d['raw'], d['graph'], d['cross']
    if mode == 'hand':                                     # Row 0a — handcrafted-only baseline
        return raw[:, :71]                                 # first 71 cols are hand_scaled
    if mode == 'deep':                                     # Row 0b — UNI deep-only baseline
        return raw[:, 71:]                                 # cols 71..2118 are tight+ctx (2048-d)
    if mode == 'raw':                                      # Row 1
        return raw
    if mode == 'raw_graph':                                # Row 2
        return np.concatenate([raw, g], axis=1)
    if mode == 'raw_graph_cross':                          # Row 3 (HEADLINE)
        return np.concatenate([raw, g, c], axis=1)
    if mode == 'raw_graph_random':                         # Row 6 (NULL CONTROL)
        rng = np.random.default_rng(seed=0)
        rand = rng.standard_normal(c.shape).astype(np.float32)
        return np.concatenate([raw, g, rand], axis=1)
    raise ValueError(f"unknown mode: {mode}")


def train_eval_fold(dataset, fold, mode, num_classes, gpu_id=0, out_dir=None,
                    n_estimators=400, max_depth=6, lr=0.1,
                    subsample=0.85, colsample_bytree=0.7, min_child_weight=5,
                    colsample_bylevel=1.0, colsample_bynode=1.0,
                    reg_lambda=1.5):
    print(f"\n[Fold {fold} / mode={mode}]")
    d = load_fold_data(dataset, fold)
    X = build_features(d, mode)
    fold_id = d['fold_id']; gt = d['gt_class']
    print(f"  X shape: {X.shape}")

    tr_mask = (fold_id != fold) & (gt > 0)
    te_mask = (fold_id == fold)
    Xtr, ytr = X[tr_mask], gt[tr_mask] - 1
    Xte = X[te_mask]
    print(f"  train: {len(ytr):,}  test: {int(te_mask.sum()):,}")

    counts = np.bincount(ytr, minlength=num_classes).astype(np.float64)
    cw = (counts.sum() / (num_classes * counts.clip(min=1))).astype(np.float32)
    sample_weight = cw[ytr]

    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,    # KEY: 1.0 for cross-block modes
        colsample_bynode=colsample_bynode,      # KEY: 1.0 for cross-block modes
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        objective='multi:softprob', num_class=num_classes,
        tree_method='hist', device=f'cuda:{gpu_id}',
        eval_metric='mlogloss', n_jobs=8, random_state=0,
    )
    clf.fit(Xtr, ytr, sample_weight=sample_weight, verbose=False)
    yp = clf.predict(Xte) + 1
    print(f"  trained+predicted in {time.time()-t0:.1f}s")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"fold{fold}_preds.npz")
        np.savez(out_path,
            patch_id=d['patch_id'][te_mask].astype(np.int32),
            fold_id=fold_id[te_mask].astype(np.int32),
            nuc_id=d['nuc_id'][te_mask].astype(np.int32),
            gt_class=gt[te_mask].astype(np.int32),
            xgb=yp.astype(np.int32),
        )
        print(f"  saved -> {out_path}")

    # Matched-only macro F1 (sanity, not Conv C)
    yt = gt[te_mask]; matched = yt > 0
    if matched.sum() > 0:
        macro = f1_score(yt[matched], yp[matched], average='macro')
        return macro * 100, clf
    return 0.0, clf


def run_ablation(dataset, folds, num_classes, gpu_id=0,
                 modes=('raw', 'raw_graph', 'raw_graph_cross', 'raw_graph_random')):
    out_base = f'/home/sbarua/fanseg/predictions/{dataset}/fusion_features_uni_raw/gacfnet_xgb'
    results = {}
    for mode in modes:
        out_dir = os.path.join(out_base, mode)
        per_fold = []
        for f in folds:
            macro, _ = train_eval_fold(dataset, f, mode, num_classes, gpu_id=gpu_id,
                                        out_dir=out_dir)
            per_fold.append(macro)
        results[mode] = per_fold
        avg = np.mean(per_fold)
        print(f"\n[{mode}] per-fold matched-only macro F1: {[round(x,2) for x in per_fold]}")
        print(f"[{mode}] avg = {avg:.2f}")

    print("\n" + "="*60)
    print("ABLATION SUMMARY (matched-only macro F1, NOT Conv C)")
    print("="*60)
    print(f"{'Mode':<25s} " + " ".join(f"f{f}".rjust(7) for f in folds) + "  avg")
    for mode in modes:
        row = " ".join(f"{x:7.2f}" for x in results[mode])
        print(f"{mode:<25s} {row}  {np.mean(results[mode]):.2f}")
    print("\nNote: Run eval_convention_c_*.py separately for the paper-comparable numbers.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['consep', 'lizard', 'pannuke'], required=True)
    ap.add_argument('--folds', type=str, default=None)
    ap.add_argument('--modes', type=str, default='raw,raw_graph,raw_graph_cross,raw_graph_random',
                    help='Comma-separated ablation modes to run')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    cfg = {
        'consep':  {'num_classes': 4, 'all_folds': [1, 2, 3, 4, 5]},
        'lizard':  {'num_classes': 6, 'all_folds': [1, 2, 3, 4, 5]},
        'pannuke': {'num_classes': 5, 'all_folds': [1, 2, 3]},
    }[args.dataset]
    folds = [int(x) for x in args.folds.split(',')] if args.folds else cfg['all_folds']
    modes = [m.strip() for m in args.modes.split(',')]
    print(f"[main] dataset={args.dataset}  folds={folds}  modes={modes}")
    run_ablation(args.dataset, folds, cfg['num_classes'], gpu_id=args.gpu, modes=modes)


if __name__ == "__main__":
    main()
