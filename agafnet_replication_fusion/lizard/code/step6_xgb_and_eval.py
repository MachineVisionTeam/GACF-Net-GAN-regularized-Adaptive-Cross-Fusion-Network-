"""Step 6+7+8 — XGBoost on fusion embeddings + Convention C head-to-head vs AGAFNet.

Controlled comparison (SAME AGAFNet masks, only the classifier differs):
  OUR     : XGBoost on [ raw(2119) || graph_emb(192) || cross_emb(192) ]
  AGAFNet : its own per-nucleus majority-vote class (agafnet_class column)

Convention C per class c (HoVer-Net, weights (2,2,1,1)):
  F_c = 2*TP / (2*TP + 2*FP + 2*FN + FP_d + FN_d)

FN_d (undetected GT) is IDENTICAL for both methods (same masks/GT), computed once
from the GT masks by checking which GT instances AGAFNet detected at IoU>=0.5.

Also runs ablation modes: hand / deep / raw / +graph / +cross  for OUR side.
"""
import os, sys, time, json, argparse
import numpy as np
import h5py
import xgboost as xgb
from sklearn.metrics import f1_score

FEAT_TPL = '/home/sbarua/fanseg/agafnet_mask_fusion/features/fusion_features_fold{N}.h5'
EMB_TPL  = '/home/sbarua/fanseg/agafnet_mask_fusion/features/fusion_emb_fold{N}.npz'
AGAF_MASKS = '/home/sbarua/Lizard/agafnet_replication/fusion/features/instance_maps/fold{N}_inst_maps.npz'
GT_MASKS   = '/mnt/storage1/Lizard/agafnet_replication/data_conic/fold{N}/masks.npy'
OUT_DIR  = '/home/sbarua/fanseg/agafnet_mask_fusion/preds'
NUM_CLASSES = 6
FOLDS = [1, 2, 3, 4, 5]
CLASS_NAMES = ['Neutrophil','Epithelial','Lymphocyte','Plasma','Eosinophil','Connective']
# AGAFNet PAPER per-class F1 (Lizard) for reference
PAPER_F1 = [24.80, 70.64, 53.31, 71.20, 68.69, 70.76]

os.makedirs(OUT_DIR, exist_ok=True)


def load_fold(fold):
    with h5py.File(FEAT_TPL.format(N=fold), 'r') as f:
        h = f['hand_scaled'][:]; t = f['tight_std'][:]; c = f['ctx_std'][:]
        fold_id = f['fold_id'][:]; gt = f['gt_class'][:]; agaf = f['agafnet_class'][:]
        nuc = f['nuc_id'][:]; patch = f['patch_id'][:]
    raw = np.concatenate([h, t, c], axis=1).astype(np.float32)
    d = np.load(EMB_TPL.format(N=fold))
    return dict(raw=raw, graph=d['graph_emb'], cross=d['cross_emb'],
                fold_id=fold_id, gt=gt, agaf=agaf, nuc=nuc, patch=patch)


def build_X(d, mode):
    raw, g, c = d['raw'], d['graph'], d['cross']
    if mode == 'hand':            return raw[:, :71]
    if mode == 'deep':            return raw[:, 71:]
    if mode == 'raw':             return raw
    if mode == 'raw_graph':       return np.concatenate([raw, g], 1)
    if mode == 'raw_graph_cross': return np.concatenate([raw, g, c], 1)
    raise ValueError(mode)


def compute_total_and_matched_gt():
    """Per class: total GT instances, and how many GT instances AGAFNet detected (IoU>=0.5).
    Returns total_gt[c], matched_gt[c]. (FN_d = total - matched, identical for both methods.)"""
    meta = np.load('/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy', allow_pickle=True).item()
    fa = meta['fold_assignments']
    total = {c: 0 for c in range(1, NUM_CLASSES+1)}
    matched = {c: 0 for c in range(1, NUM_CLASSES+1)}
    t0 = time.time()
    for fold in FOLDS:
        agaf = np.load(AGAF_MASKS.format(N=fold))
        gt = np.load(GT_MASKS.format(N=fold), mmap_mode='r')
        n_patches = len([k for k in agaf.files if k.startswith('inst_')])
        for local in range(n_patches):
            inst_pred = agaf[f'inst_{local}'].astype(np.int32)
            gt_patch = gt[local]
            # build GT instance map + class
            gt_inst = np.zeros((256,256), np.int32); gcls = {}; gid=0
            for ch in range(NUM_CLASSES):
                chan = gt_patch[:,:,ch].astype(np.int32)
                for oid in np.unique(chan):
                    if oid==0: continue
                    gid+=1; gt_inst[chan==oid]=gid; gcls[gid]=ch+1
            # for each GT instance, is it detected by some pred at IoU>=0.5?
            for g in range(1, gid+1):
                gm = gt_inst == g
                total[gcls[g]] += 1
                overlap = np.unique(inst_pred[gm]); overlap = overlap[overlap>0]
                det = False
                for p in overlap:
                    pm = inst_pred == p
                    iou = (gm&pm).sum()/max((gm|pm).sum(),1)
                    if iou >= 0.5: det = True; break
                if det: matched[gcls[g]] += 1
        print(f"  fn_d fold{fold} done ({time.time()-t0:.0f}s)", flush=True)
    return total, matched


def convention_c(pred_class, gt_class, fn_d, w=(2,2,1,1)):
    """pred_class, gt_class: arrays over ALL AGAFNet nuclei (gt_class==0 means unmatched)."""
    f1 = {}
    for c in range(1, NUM_CLASSES+1):
        matched = gt_class > 0
        tp = int(((gt_class==c) & (pred_class==c) & matched).sum())
        fp = int(((gt_class!=c) & (pred_class==c) & matched).sum())
        fn = int(((gt_class==c) & (pred_class!=c) & matched).sum())
        fp_d = int(((~matched) & (pred_class==c)).sum())
        num = 2*tp
        den = 2*tp + w[0]*fp + w[1]*fn + w[2]*fp_d + w[3]*fn_d[c]
        f1[c] = (num/den*100) if den>0 else 0.0
    return f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    # ---- Load all folds ----
    print("[load] all folds...")
    data = {f: load_fold(f) for f in FOLDS}

    # ---- FN_d (shared, computed once) ----
    print("[fn_d] computing total + matched GT per class (this re-matches from GT side)...")
    total_gt, matched_gt = compute_total_and_matched_gt()
    fn_d = {c: max(0, total_gt[c] - matched_gt[c]) for c in range(1, NUM_CLASSES+1)}
    print(f"  total_gt={total_gt}")
    print(f"  matched_gt={matched_gt}")
    print(f"  FN_d={fn_d}")

    # ---- Build global arrays for AGAFNet eval (concatenate all folds' TEST rows) ----
    # For per-fold CV, each nucleus is a TEST nucleus exactly once (in its own fold).
    all_gt, all_agaf = [], []
    for f in FOLDS:
        m = data[f]['fold_id'] == f
        all_gt.append(data[f]['gt'][m]); all_agaf.append(data[f]['agaf'][m])
    all_gt = np.concatenate(all_gt); all_agaf = np.concatenate(all_agaf)

    # ---- AGAFNet majority-vote Convention C ----
    agaf_f1 = convention_c(all_agaf, all_gt, fn_d)
    print(f"\n[AGAFNet majority-vote] per-class F1: "
          f"{[round(agaf_f1[c],2) for c in range(1,NUM_CLASSES+1)]}  "
          f"AVG={np.mean(list(agaf_f1.values())):.2f}")

    # ---- OUR XGBoost, multiple ablation modes ----
    modes = ['hand', 'deep', 'raw', 'raw_graph', 'raw_graph_cross']
    our_results = {}
    for mode in modes:
        all_pred, all_gt2 = [], []
        for test_fold in FOLDS:
            d = data[test_fold]
            X = build_X(d, mode)
            tr = (d['fold_id'] != test_fold) & (d['gt'] > 0)
            te = (d['fold_id'] == test_fold)
            Xtr, ytr = X[tr], d['gt'][tr]-1
            counts = np.bincount(ytr, minlength=NUM_CLASSES).astype(np.float64)
            cw = (counts.sum()/(NUM_CLASSES*counts.clip(min=1))).astype(np.float32)
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.85, colsample_bytree=0.7, colsample_bylevel=1.0,
                colsample_bynode=1.0, min_child_weight=5, reg_lambda=1.5,
                objective='multi:softprob', num_class=NUM_CLASSES,
                tree_method='hist', device=f'cuda:{args.gpu}', n_jobs=8, random_state=0)
            clf.fit(Xtr, ytr, sample_weight=cw[ytr], verbose=False)
            pred = clf.predict(X[te]) + 1
            all_pred.append(pred); all_gt2.append(d['gt'][te])
        all_pred = np.concatenate(all_pred); all_gt2 = np.concatenate(all_gt2)
        f1 = convention_c(all_pred, all_gt2, fn_d)
        our_results[mode] = f1
        print(f"[OUR {mode:16s}] per-class: {[round(f1[c],2) for c in range(1,NUM_CLASSES+1)]}  "
              f"AVG={np.mean(list(f1.values())):.2f}")

    # ---- Report ----
    print(f"\n{'='*90}")
    print(f"HEAD-TO-HEAD on AGAFNet's predicted masks (Convention C, Lizard)")
    print(f"{'='*90}")
    hdr = f"{'Class':<12}{'Paper':>8}{'AGAFNet-MV':>12}" + ''.join(f"{m[:10]:>12}" for m in modes)
    print(hdr)
    for c in range(1, NUM_CLASSES+1):
        row = f"{CLASS_NAMES[c-1]:<12}{PAPER_F1[c-1]:>8.2f}{agaf_f1[c]:>12.2f}"
        row += ''.join(f"{our_results[m][c]:>12.2f}" for m in modes)
        print(row)
    avg_paper = np.mean(PAPER_F1); avg_agaf = np.mean(list(agaf_f1.values()))
    row = f"{'AVG F1':<12}{avg_paper:>8.2f}{avg_agaf:>12.2f}"
    row += ''.join(f"{np.mean(list(our_results[m].values())):>12.2f}" for m in modes)
    print(row)

    out = {
        'fn_d': fn_d, 'total_gt': total_gt, 'matched_gt': matched_gt,
        'agafnet_majority_vote': {CLASS_NAMES[c-1]: agaf_f1[c] for c in range(1,NUM_CLASSES+1)},
        'our': {m: {CLASS_NAMES[c-1]: our_results[m][c] for c in range(1,NUM_CLASSES+1)} for m in modes},
        'paper': {CLASS_NAMES[c-1]: PAPER_F1[c-1] for c in range(NUM_CLASSES)},
    }
    with open(f'{OUT_DIR}/headtohead_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[saved] {OUT_DIR}/headtohead_results.json")


if __name__ == '__main__':
    main()
