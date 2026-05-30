"""Step 5 — Train GACF-Net fusion model on AGAFNet-mask features, extract embeddings.

Per fold: train the fusion model (shared proj + GMU + cell-graph transformer +
MFB cross-block) end-to-end on AGAFNet-detected nuclei, then extract graph_emb
(192) + cross_emb (192) for ALL nuclei. Same recipe as the main GACF-Net run:
CB-focal + balanced softmax + modality dropout, per-patch graph batching.

Output: features/fusion_emb_fold{N}.npz  (graph_emb, cross_emb + metadata)
"""
import os, sys, time, argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gacfnet import GACFNet, count_params

FEAT_TPL = '/home/sbarua/fanseg/agafnet_mask_fusion/features/fusion_features_fold{N}.h5'
OUT_TPL  = '/home/sbarua/fanseg/agafnet_mask_fusion/features/fusion_emb_fold{N}.npz'
NUM_CLASSES = 6
FOLDS = [1, 2, 3, 4, 5]


class PatchDataset(Dataset):
    """Group nuclei by (fold, local_patch) so each item is one patch's graph."""
    def __init__(self, hand, tight, ctx, gt, cx, cy, fold_id, patch_id, indices):
        self.hand, self.tight, self.ctx = hand, tight, ctx
        self.gt, self.cx, self.cy = gt, cx, cy
        self.fold_id, self.patch_id = fold_id, patch_id
        self.indices = indices
        keys = np.stack([fold_id[indices], patch_id[indices]], axis=1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        self.groups = [np.where(inv == g)[0] for g in range(len(uniq))]

    def __len__(self): return len(self.groups)

    def __getitem__(self, i):
        rows = self.indices[self.groups[i]]
        return {
            'hand': torch.from_numpy(self.hand[rows]).float(),
            'tight': torch.from_numpy(self.tight[rows]).float(),
            'ctx': torch.from_numpy(self.ctx[rows]).float(),
            'gt': torch.from_numpy(self.gt[rows]).long(),
            'cx': torch.from_numpy(self.cx[rows]).float(),
            'cy': torch.from_numpy(self.cy[rows]).float(),
            'rows': torch.from_numpy(rows).long(),
        }


def collate(batch):
    hand = torch.cat([b['hand'] for b in batch]); tight = torch.cat([b['tight'] for b in batch])
    ctx = torch.cat([b['ctx'] for b in batch]); gt = torch.cat([b['gt'] for b in batch])
    cx = torch.cat([b['cx'] for b in batch]); cy = torch.cat([b['cy'] for b in batch])
    rows = torch.cat([b['rows'] for b in batch])
    pids = torch.cat([torch.full((b['hand'].size(0),), i, dtype=torch.long) for i, b in enumerate(batch)])
    return {'hand':hand,'tight':tight,'ctx':ctx,'gt':gt,
            'centroids':torch.stack([cx,cy],-1),'patch_ids':pids,'rows':rows}


def cb_focal(logits, targets, spc, beta=0.999, gamma=2.0):
    C = logits.size(-1)
    spc = spc.float().clamp(min=1.0)
    w = (1 - beta) / (1 - beta ** spc).clamp(min=1e-12)
    w = (w / w.sum() * C).to(logits.device)
    at = w[targets]
    logp = F.log_softmax(logits, -1)
    logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = logpt.exp().clamp(1e-12, 1.0)
    return (-at * (1-pt)**gamma * logpt).mean()


def bal_softmax(logits, spc, tau=1.0):
    spc = spc.float().clamp(min=1.0)
    return logits + tau * torch.log((spc/spc.sum()).to(logits.device)).unsqueeze(0)


def train_fold(fold, device='cuda:0', epochs=40, lr=1e-3, bs=8, seed=42):
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
    torch.manual_seed(seed); np.random.seed(seed)
    with h5py.File(FEAT_TPL.format(N=fold), 'r') as f:
        hand = f['hand_scaled'][:]; tight = f['tight_std'][:]; ctx = f['ctx_std'][:]
        fold_id = f['fold_id'][:]; patch_id = f['patch_id'][:]
        gt = f['gt_class'][:]; cx = f['centroid_x'][:]; cy = f['centroid_y'][:]
        nuc = f['nuc_id'][:]; gpatch = f['global_patch'][:]
    N = len(hand)
    tr = np.where((fold_id != fold) & (gt > 0))[0]
    print(f"  N={N:,}  train(labeled)={len(tr):,}")
    spc = torch.from_numpy(np.bincount(gt[tr]-1, minlength=NUM_CLASSES)).float()
    print(f"  class dist: {np.bincount(gt[tr]-1, minlength=NUM_CLASSES).tolist()}")

    ds = PatchDataset(hand, tight, ctx, gt, cx, cy, fold_id, patch_id, tr)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, collate_fn=collate)

    model = GACFNet(hand_dim=71, tight_dim=1024, ctx_dim=1024, d_proj=192,
                    n_heads=4, n_layers=1, k_graph=64, k_mfb=3,
                    num_classes=NUM_CLASSES, dropout=0.4, modality_dropout=0.15).to(device)
    print(f"  model params: {count_params(model)[0]:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)

    for ep in range(1, epochs+1):
        model.train(); t0=time.time(); ls=0; n=0
        for b in dl:
            h=b['hand'].to(device); t=b['tight'].to(device); c=b['ctx'].to(device)
            ce=b['centroids'].to(device); pi=b['patch_ids'].to(device); g=b['gt'].to(device)
            opt.zero_grad()
            logits = model(h, t, c, pi, ce)
            tgt = g - 1; mask = tgt >= 0
            if mask.sum()==0: continue
            loss = cb_focal(bal_softmax(logits[mask], spc), tgt[mask], spc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls += loss.item()*mask.sum().item(); n += mask.sum().item()
        sch.step()
        if ep % 10 == 0 or ep == 1:
            print(f"  ep{ep:3d}/{epochs} loss={ls/max(n,1):.4f} t={time.time()-t0:.1f}s", flush=True)

    # Extract embeddings for ALL nuclei
    print("  [extract] all nuclei...")
    all_idx = np.arange(N)
    ds_all = PatchDataset(hand, tight, ctx, gt, cx, cy, fold_id, patch_id, all_idx)
    dl_all = DataLoader(ds_all, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate)
    graph_emb = np.zeros((N, 192), dtype=np.float32)
    cross_emb = np.zeros((N, 192), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b in dl_all:
            h=b['hand'].to(device); t=b['tight'].to(device); c=b['ctx'].to(device)
            ce=b['centroids'].to(device); pi=b['patch_ids'].to(device); rows=b['rows'].numpy()
            _, ge, cre = model(h, t, c, pi, ce, return_embeddings=True)
            graph_emb[rows] = ge.cpu().numpy(); cross_emb[rows] = cre.cpu().numpy()

    out = OUT_TPL.format(N=fold)
    np.savez(out, graph_emb=graph_emb, cross_emb=cross_emb,
             fold_id=fold_id, patch_id=patch_id, nuc_id=nuc, gt_class=gt,
             agafnet_class=h5py.File(FEAT_TPL.format(N=fold),'r')['agafnet_class'][:],
             global_patch=gpatch)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=str, default='1,2,3,4,5')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    for f in [int(x) for x in args.folds.split(',')]:
        train_fold(f, device=device, epochs=args.epochs)
    print("\nAll folds done.")


if __name__ == '__main__':
    main()
