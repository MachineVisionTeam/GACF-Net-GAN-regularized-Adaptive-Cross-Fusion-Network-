"""Train GACF-Net — context-aware hybrid fusion on fixed features.

Per-fold training (5-fold for Lizard/CoNSeP, 3-fold for PanNuke).

Data layout (per fold H5 from feature-fix session):
  hand_scaled (N, 71)   tight_std (N, 1024)   ctx_std (N, 1024)
  fold_id, patch_id, nuc_id, gt_class
  (centroid_x, centroid_y for CoNSeP; need to derive for Lizard from matching H5)

Training:
  - Patch-batched: each "item" in the batch is one whole patch (variable # nuclei).
  - Per-batch: pack N patches as a flat (N_total, ...) tensor with global patch_ids.
  - Loss: CB-focal (Cui et al. 2019, beta=0.999, gamma=2.0)
          + balanced softmax (Ren et al. 2020, adds log(prior) to logits)
  - Optimizer: AdamW, lr=1e-3, wd=1e-4, cosine schedule.
  - Train only on labeled nuclei (gt_class > 0). Background rows kept for graph context.

After training: extract per-nucleus graph_emb + cross_emb for train + test, save NPZ.
"""
import os, sys, time, json, argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gacfnet import GACFNet, count_params


# =============================================================================
# Dataset — group nuclei by patch
# =============================================================================
class PatchGroupedDataset(Dataset):
    """Each item = ALL nuclei in one (fold, patch) for graph construction.

    Returns a dict per item — collate_fn flattens across patches into one batch.
    """
    def __init__(self, hand, tight, ctx, gt_class, patch_id, fold_id, nuc_id,
                 centroid_x, centroid_y, indices):
        self.hand = hand
        self.tight = tight
        self.ctx = ctx
        self.gt_class = gt_class
        self.patch_id = patch_id
        self.fold_id  = fold_id
        self.nuc_id   = nuc_id
        self.cx = centroid_x
        self.cy = centroid_y
        self.indices = indices  # rows in master arrays

        # Group indices by (fold_id, patch_id)
        keys = np.stack([fold_id[indices], patch_id[indices]], axis=1)
        unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        self.patch_groups = [np.where(inv == g)[0] for g in range(len(unique_keys))]
        self.patch_keys = unique_keys

    def __len__(self):
        return len(self.patch_groups)

    def __getitem__(self, i):
        loc = self.patch_groups[i]
        rows = self.indices[loc]
        return {
            'hand':     torch.from_numpy(self.hand[rows]).float(),
            'tight':    torch.from_numpy(self.tight[rows]).float(),
            'ctx':      torch.from_numpy(self.ctx[rows]).float(),
            'gt_class': torch.from_numpy(self.gt_class[rows]).long(),
            'cx':       torch.from_numpy(self.cx[rows]).float(),
            'cy':       torch.from_numpy(self.cy[rows]).float(),
            'rows':     torch.from_numpy(rows).long(),   # master indices, for embedding extraction
        }


def patch_collate(batch_items):
    """Flatten across patches. Assign a unique local patch_id within the batch."""
    hand = torch.cat([b['hand']     for b in batch_items], dim=0)
    tight = torch.cat([b['tight']   for b in batch_items], dim=0)
    ctx = torch.cat([b['ctx']       for b in batch_items], dim=0)
    gt = torch.cat([b['gt_class']   for b in batch_items], dim=0)
    cx = torch.cat([b['cx']         for b in batch_items], dim=0)
    cy = torch.cat([b['cy']         for b in batch_items], dim=0)
    rows = torch.cat([b['rows']     for b in batch_items], dim=0)
    pids = torch.cat([torch.full((b['hand'].size(0),), i, dtype=torch.long)
                      for i, b in enumerate(batch_items)], dim=0)
    centroids = torch.stack([cx, cy], dim=-1)
    return {
        'hand': hand, 'tight': tight, 'ctx': ctx,
        'gt': gt, 'patch_ids': pids, 'centroids': centroids,
        'rows': rows,
    }


# =============================================================================
# Losses — CB-focal + balanced softmax
# =============================================================================
def cb_focal_loss(logits, targets, samples_per_cls, beta=0.999, gamma=2.0):
    """Class-balanced focal loss (Cui et al. 2019).

    Effective number of samples: E_n = (1 - beta^n) / (1 - beta)
    Class weight: w_c = (1 - beta) / (1 - beta^{n_c})  -> normalize to sum to num_classes.
    Then standard focal loss with these per-class weights as alpha.

    logits: (B, C)         targets: (B,) in 0..C-1
    samples_per_cls: tensor (C,) of training-fold counts per class.
    """
    C = logits.size(-1)
    samples_per_cls = samples_per_cls.float().clamp(min=1.0)
    effective_num = 1.0 - beta ** samples_per_cls
    weights = (1.0 - beta) / effective_num.clamp(min=1e-12)
    weights = weights / weights.sum() * C   # normalize to mean=1, sum=C

    weights = weights.to(logits.device)
    alpha_t = weights[targets]                # (B,)
    log_probs = F.log_softmax(logits, dim=-1)
    log_p_t   = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t       = log_p_t.exp().clamp(min=1e-12, max=1.0)
    focal     = (1.0 - p_t) ** gamma
    loss = -alpha_t * focal * log_p_t
    return loss.mean()


def balanced_softmax_logits(logits, samples_per_cls, tau=1.0):
    """Balanced softmax (Ren et al. 2020): add tau * log(prior) to logits BEFORE softmax.

    Effect: trained classifier behaves as if classes were balanced.
    """
    samples_per_cls = samples_per_cls.float().clamp(min=1.0)
    prior = samples_per_cls / samples_per_cls.sum()
    log_prior = torch.log(prior).to(logits.device)
    return logits + tau * log_prior.unsqueeze(0)


# =============================================================================
# Train one fold
# =============================================================================
def train_fold(feat_h5, matching_h5, test_fold, num_classes, out_dir,
               epochs=40, lr=1e-3, wd=1e-4, batch_patches=8,
               d_proj=192, k_graph=64, n_layers=1,
               device='cuda:0', seed=42, smoke=False):
    print(f"\n{'='*70}\nFOLD {test_fold}  (smoke={smoke})\n{'='*70}")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- Load fold features ----
    print(f"[load] {feat_h5}")
    with h5py.File(feat_h5, 'r') as f:
        hand     = f['hand_scaled'][:]
        tight    = f['tight_std'][:]
        ctx      = f['ctx_std'][:]
        fold_id  = f['fold_id'][:]
        patch_id = f['patch_id'][:]
        nuc_id   = f['nuc_id'][:]
        gt_class = f['gt_class'][:]
    N = len(hand)
    print(f"  N={N:,}  hand={hand.shape}  tight={tight.shape}  ctx={ctx.shape}")

    # ---- Centroids from matching H5 ----
    # CoNSeP/PanNuke use LOCAL patch_id in both hand and matching H5.
    # Lizard uses LOCAL patch_id in hand H5 but GLOBAL patch_idx in matching H5
    # → need local→global mapping via metadata for Lizard.
    print(f"[load] centroids from {matching_h5}")
    with h5py.File(matching_h5, 'r') as f:
        keys = list(f.keys())
        m_fold = f['fold_id'][:]
        m_nid  = f['nuc_id'][:]
        cx_all = f['centroid_x'][:]
        cy_all = f['centroid_y'][:]
        # Determine which key matching H5 uses
        if 'patch_id' in keys:
            m_pid = f['patch_id'][:]
            patch_key_type = 'local'
        elif 'patch_idx' in keys:
            m_pid = f['patch_idx'][:]
            patch_key_type = 'global'
        else:
            raise KeyError(f"matching H5 has neither patch_id nor patch_idx; keys={keys}")
    print(f"  matching H5 patch key type: {patch_key_type}")

    # If global, build local->global map from metadata (Lizard convention)
    if patch_key_type == 'global':
        meta = np.load('/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy',
                       allow_pickle=True).item()
        fa = meta['fold_assignments']
        local_to_global = {}
        for split in [1, 2, 3, 4, 5]:
            for local_pos, gpidx in enumerate(sorted([k for k, v in fa.items() if v == split])):
                local_to_global[(int(split), int(local_pos))] = int(gpidx)
        # Convert hand-H5 (fold, local_pid) -> global_pid for lookup
        hand_patch_for_match = np.zeros(N, dtype=np.int64)
        for i in range(N):
            hand_patch_for_match[i] = local_to_global.get((int(fold_id[i]), int(patch_id[i])), -1)
    else:
        hand_patch_for_match = patch_id

    # Build matching-H5 lookup keyed by (fold, m_pid, nuc_id)
    lookup = {(int(m_fold[i]), int(m_pid[i]), int(m_nid[i])): i for i in range(len(m_fold))}
    centroid_x = np.zeros(N, dtype=np.float32)
    centroid_y = np.zeros(N, dtype=np.float32)
    miss = 0
    for i in range(N):
        j = lookup.get((int(fold_id[i]), int(hand_patch_for_match[i]), int(nuc_id[i])))
        if j is None: miss += 1; continue
        centroid_x[i] = cx_all[j]; centroid_y[i] = cy_all[j]
    if miss: print(f"  WARN: {miss} nuclei missing centroid (will get [0,0])")

    # ---- Indices: train vs test ----
    train_mask = (fold_id != test_fold) & (gt_class > 0)
    test_mask  = (fold_id == test_fold)   # incl bg for Conv C
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(test_mask)[0]
    print(f"[split] train: {len(train_idx):,} labeled  |  test: {len(test_idx):,} (incl bg)")

    # Smoke: take first 2000 train, first 1000 test
    if smoke:
        train_idx = train_idx[:2000]
        test_idx  = test_idx[:1000]
        print(f"  SMOKE: shrunk to train={len(train_idx)}  test={len(test_idx)}")

    # Class counts on training fold (for CB-focal + balanced softmax)
    train_targets = gt_class[train_idx] - 1   # 0..C-1
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    samples_per_cls = torch.from_numpy(counts).float()
    print(f"  class dist (train): {counts.astype(int).tolist()}")

    # ---- Build datasets/loaders ----
    train_ds = PatchGroupedDataset(hand, tight, ctx, gt_class, patch_id, fold_id, nuc_id,
                                    centroid_x, centroid_y, train_idx)
    test_ds  = PatchGroupedDataset(hand, tight, ctx, gt_class, patch_id, fold_id, nuc_id,
                                    centroid_x, centroid_y, test_idx)
    print(f"  patches: train={len(train_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_patches, shuffle=True,
                              num_workers=2, collate_fn=patch_collate, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_patches, shuffle=False,
                              num_workers=2, collate_fn=patch_collate, drop_last=False)

    # ---- Model ----
    model = GACFNet(
        hand_dim=71, tight_dim=1024, ctx_dim=1024,
        d_proj=d_proj, n_heads=4, n_layers=n_layers,
        k_graph=k_graph, k_mfb=3, num_classes=num_classes,
        dropout=0.4, modality_dropout=0.15,
    ).to(device)
    total, _, _ = count_params(model)
    print(f"  model: {total:,} params on {device}")

    # ---- Optimizer / scheduler ----
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.01)

    # ---- Train loop ----
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            hand_b   = batch['hand'].to(device, non_blocking=True)
            tight_b  = batch['tight'].to(device, non_blocking=True)
            ctx_b    = batch['ctx'].to(device, non_blocking=True)
            cent_b   = batch['centroids'].to(device, non_blocking=True)
            pids_b   = batch['patch_ids'].to(device, non_blocking=True)
            gt_b     = batch['gt'].to(device, non_blocking=True)

            optim.zero_grad()
            logits = model(hand_b, tight_b, ctx_b, pids_b, cent_b)
            # gt_b is 1..C in the loaded data; map to 0..C-1
            tgt = gt_b - 1
            # Only train on labeled (gt>0). Background (gt==0 -> -1) is skipped.
            mask = tgt >= 0
            if mask.sum() == 0:
                continue
            logits_adj = balanced_softmax_logits(logits[mask], samples_per_cls)
            loss = cb_focal_loss(logits_adj, tgt[mask], samples_per_cls,
                                  beta=0.999, gamma=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            loss_sum += loss.item() * mask.sum().item()
            n_seen += mask.sum().item()
        sched.step()
        avg = loss_sum / max(n_seen, 1)
        print(f"  [ep {ep:3d}/{epochs}] loss={avg:.4f}  lr={optim.param_groups[0]['lr']:.2e}  "
              f"t={time.time()-t0:.1f}s  ({n_seen} labeled nuclei)", flush=True)

    # ---- Extract embeddings for ALL nuclei (train+test, all rows in this fold's H5) ----
    print("\n[extract] extracting graph + cross embeddings for ALL nuclei...")
    model.eval()
    graph_emb_all = np.zeros((N, d_proj), dtype=np.float32)
    cross_emb_all = np.zeros((N, (d_proj // 3) * 3), dtype=np.float32)

    # Use ALL indices (no labeling filter), grouped by patch
    all_indices = np.arange(N)
    all_ds = PatchGroupedDataset(hand, tight, ctx, gt_class, patch_id, fold_id, nuc_id,
                                  centroid_x, centroid_y, all_indices)
    all_loader = DataLoader(all_ds, batch_size=batch_patches, shuffle=False,
                            num_workers=2, collate_fn=patch_collate, drop_last=False)
    t0 = time.time()
    with torch.no_grad():
        for batch in all_loader:
            hand_b  = batch['hand'].to(device);  tight_b = batch['tight'].to(device)
            ctx_b   = batch['ctx'].to(device);   cent_b  = batch['centroids'].to(device)
            pids_b  = batch['patch_ids'].to(device)
            rows    = batch['rows'].numpy()
            _, ge, ce = model(hand_b, tight_b, ctx_b, pids_b, cent_b, return_embeddings=True)
            graph_emb_all[rows] = ge.cpu().numpy()
            cross_emb_all[rows] = ce.cpu().numpy()
    print(f"  extracted in {time.time()-t0:.1f}s")
    print(f"  graph_emb shape: {graph_emb_all.shape}  mean_norm={np.linalg.norm(graph_emb_all, axis=1).mean():.2f}")
    print(f"  cross_emb shape: {cross_emb_all.shape}  mean_norm={np.linalg.norm(cross_emb_all, axis=1).mean():.2f}")

    out_path = os.path.join(out_dir, f"embeddings_fold{test_fold}.npz")
    np.savez(out_path,
        graph_emb=graph_emb_all, cross_emb=cross_emb_all,
        fold_id=fold_id.astype(np.int32), patch_id=patch_id.astype(np.int32),
        nuc_id=nuc_id.astype(np.int32), gt_class=gt_class.astype(np.int32),
    )
    print(f"  saved -> {out_path}")
    return out_path


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['consep', 'lizard', 'pannuke'], required=True)
    ap.add_argument('--folds', type=str, default=None,
                    help='Comma-separated fold IDs to run (default: all for that dataset)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_patches', type=int, default=8)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='Tiny smoke test (2K train, 2 epochs)')
    args = ap.parse_args()

    # Per-dataset config
    cfg = {
        'consep': {
            'feat_tpl': '/home/sbarua/fanseg/predictions/consep/fusion_features_uni_raw/fusion_features_fold{N}.h5',
            'matching': '/home/sbarua/fanseg/predictions/consep/fusion_features_reinhard/phase2/nucleus_matching.h5',
            'num_classes': 4, 'all_folds': [1, 2, 3, 4, 5],
        },
        'lizard': {
            'feat_tpl': '/home/sbarua/fanseg/predictions/lizard/fusion_features_uni_raw/fusion_features_fold{N}.h5',
            'matching': '/home/sbarua/fanseg/predictions/lizard/fusion_features_phase2/nucleus_matching.h5',
            'num_classes': 6, 'all_folds': [1, 2, 3, 4, 5],
        },
        'pannuke': {
            'feat_tpl': '/home/sbarua/fanseg/predictions/pannuke/fusion_features_uni_raw/fusion_features_fold{N}.h5',
            'matching': '/home/sbarua/fanseg/predictions/pannuke/fusion_features_reinhard/phase2/nucleus_matching.h5',
            'num_classes': 5, 'all_folds': [1, 2, 3],
        },
    }[args.dataset]

    folds = [int(x) for x in args.folds.split(',')] if args.folds else cfg['all_folds']
    if args.smoke:
        folds = folds[:1]
        epochs = 2
    else:
        epochs = args.epochs

    out_dir = f"/home/sbarua/fanseg/predictions/{args.dataset}/fusion_features_uni_raw/gacfnet_embeddings"
    if args.smoke:
        out_dir += '_smoke'
    print(f"[main] dataset={args.dataset}  folds={folds}  epochs={epochs}  out={out_dir}")

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    for f in folds:
        train_fold(
            feat_h5=cfg['feat_tpl'].format(N=f),
            matching_h5=cfg['matching'],
            test_fold=f,
            num_classes=cfg['num_classes'],
            out_dir=out_dir,
            epochs=epochs, lr=args.lr, batch_patches=args.batch_patches,
            device=device, smoke=args.smoke,
        )
    print("\nAll folds done.")


if __name__ == "__main__":
    main()
