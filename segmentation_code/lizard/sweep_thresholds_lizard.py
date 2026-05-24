"""Quick threshold sweep on Lizard split-N predictions to find optimal post-processing.

Re-runs only the watershed step (no model inference) using cached HV+Sem3 outputs.
Since we DIDN'T save raw HV/Sem3 from inference (only the watershed-ed masks), this
script re-runs encoder+decoder inference on test patches first (~3-5 min per split),
THEN sweeps thresholds — much faster than re-training.

Sweep grid:
  center_thresh ∈ {0.3, 0.4, 0.5, 0.6}
  bg_thresh     ∈ {0.3, 0.4, 0.5}
  min_size      ∈ {10, 20, 30}
  → 36 combinations, evaluate PQ + AJI per combination
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.morphology import remove_small_objects

sys.path.insert(0, "/home/sbarua/fanseg/code")
from encoder import UNIEncoder
from decoder_dpt import DPTDecoder
from lizard_loader import LizardDataset
from inference_phase0 import get_fast_pq
from eval_lizard_full_metrics import compute_aji


class FanSegLizard(torch.nn.Module):
    def __init__(self, depths=(5, 11, 17, 23), decoder_ch=128):
        super().__init__()
        self.encoder = UNIEncoder(freeze=True)
        self.decoder = DPTDecoder(in_dim=1024, ch=decoder_ch)
        self.depths = depths
    def forward(self, x):
        ms = self.encoder.forward_multiscale(x, depths=self.depths)
        return self.decoder(ms)


def watershed_postproc(hv, sem3, center_thresh, bg_thresh, min_size):
    bg, interior, boundary = sem3[0], sem3[1], sem3[2]
    fg = bg < bg_thresh
    if fg.sum() == 0:
        return np.zeros_like(fg, dtype=np.int32)
    seeds = (interior > center_thresh) & (boundary < 0.3) & fg
    seeds = remove_small_objects(seeds, min_size=3)
    markers, _ = ndi.label(seeds)
    energy = boundary + np.abs(sobel(hv[0])) + np.abs(sobel(hv[1]))
    inst = watershed(energy, markers=markers, mask=fg)
    if inst.max() > 0:
        inst = remove_small_objects(inst.astype(np.int32), min_size=min_size)
    return inst.astype(np.int32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--n_eval", type=int, default=200, help="N patches to sample for fast sweep")
    p.add_argument("--device", default="cuda:1")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ckpt is None:
        args.ckpt = f"/home/sbarua/fanseg/checkpoints/lizard_phase1/split{args.split}/best.pt"
    device = torch.device(args.device)

    print(f"[init] loading checkpoint {args.ckpt}")
    cp = torch.load(args.ckpt, map_location=device, weights_only=False)
    depths = tuple(cp.get("depths", (5, 11, 17, 23)))
    decoder_ch = cp.get("decoder_ch", 128)
    model = FanSegLizard(depths=depths, decoder_ch=decoder_ch).to(device)
    model.load_state_dict(cp["model"])
    model.eval()

    ds = LizardDataset(split=args.split, mode="test", augment=False)
    print(f"[data] split {args.split}: {len(ds)} test patches, will sample {args.n_eval}")

    # Cache raw HV + Sem3 + GT inst on subset
    rng = np.random.RandomState(42)
    subset = rng.choice(len(ds), min(args.n_eval, len(ds)), replace=False)
    cached = []
    t0 = time.time()
    with torch.inference_mode():
        for i, idx in enumerate(subset):
            sample = ds[int(idx)]
            img = sample["image"].unsqueeze(0).to(device)
            hv, sem3_logits = model(img)
            sem3 = torch.softmax(sem3_logits, dim=1).squeeze(0).cpu().numpy()
            hv_np = hv.squeeze(0).cpu().numpy()
            cached.append((hv_np, sem3, sample["inst_map"].numpy()))
            if (i + 1) % 50 == 0:
                print(f"  [cache {i+1}/{len(subset)}] t={time.time()-t0:.1f}s", flush=True)

    print(f"\n[sweep] grid: 4 center × 3 bg × 3 min_size = 36 combos")
    grid = [(c, b, m) for c in [0.3, 0.4, 0.5, 0.6]
                      for b in [0.3, 0.4, 0.5]
                      for m in [10, 20, 30]]

    results = []
    for ci, (c, b, m) in enumerate(grid):
        pqs, ajis, dqs = [], [], []
        for hv, sem3, gt_inst in cached:
            pred = watershed_postproc(hv, sem3, c, b, m)
            pq, dq, sq = get_fast_pq(gt_inst, pred)
            pqs.append(pq); dqs.append(dq)
            ajis.append(compute_aji(gt_inst, pred))
        mean_pq  = float(np.mean(pqs))
        mean_aji = float(np.mean(ajis))
        mean_dq  = float(np.mean(dqs))
        results.append({"center": c, "bg": b, "min": m,
                        "pq": mean_pq, "aji": mean_aji, "dq": mean_dq})
        if (ci + 1) % 6 == 0:
            print(f"  [combo {ci+1}/36] last: center={c} bg={b} min={m} → PQ={mean_pq:.4f}", flush=True)

    # Sort by PQ
    results.sort(key=lambda r: -r["pq"])
    print(f"\n[result] top 10 by PQ on split {args.split} subset of {len(subset)} patches:")
    print(f"  {'rank':>4} {'center':>7} {'bg':>5} {'min':>5} {'PQ':>8} {'AJI':>8} {'DQ':>8}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>4} {r['center']:>7.2f} {r['bg']:>5.2f} {r['min']:>5d}  {r['pq']:>8.4f} {r['aji']:>8.4f} {r['dq']:>8.4f}")
    print(f"\n[current defaults] center=0.5 bg=0.4 min=20:")
    for r in results:
        if r["center"] == 0.5 and r["bg"] == 0.4 and r["min"] == 20:
            print(f"   PQ={r['pq']:.4f}  AJI={r['aji']:.4f}  DQ={r['dq']:.4f}")
            break
    out = f"/home/sbarua/fanseg/predictions/lizard/split{args.split}/_threshold_sweep.json"
    with open(out, "w") as f:
        json.dump({"split": args.split, "n_eval": len(subset), "grid": results}, f, indent=2)
    print(f"\n[saved] {out}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
