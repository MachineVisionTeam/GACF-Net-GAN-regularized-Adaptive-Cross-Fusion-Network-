"""Full segmentation metrics (ACC, Dice, SE, SP, AJI, PQ, DQ, SQ) on Phase 2 + TTA masks.
Reads .mat files from masks_phase2_tta/ per split. Same protocol as eval_lizard_full_metrics.py.
"""
import os, sys, time, json, argparse
import numpy as np
import scipy.io as sio

sys.path.insert(0, "/home/sbarua/fanseg/code")
from eval_lizard_full_metrics import (
    parse_lizard_masks, compute_pq, compute_aji, compute_pixel_metrics,
)


METADATA_PATH = "/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy"
DATA_DIR      = "/mnt/storage1/Lizard/agafnet_replication/data_conic"
MASKS_BASE    = "/home/sbarua/fanseg/predictions/lizard"
NUM_CLASSES   = 6


def eval_split(split, mask_subdir="masks_phase2_tta"):
    meta = np.load(METADATA_PATH, allow_pickle=True).item()
    fa = meta["fold_assignments"]
    sorted_indices = sorted([k for k, v in fa.items() if v == split])
    gt_masks_npy = np.load(os.path.join(DATA_DIR, f"fold{split}", "masks.npy"))
    assert len(gt_masks_npy) == len(sorted_indices)

    metrics = {"acc": [], "dice": [], "se": [], "sp": [],
               "aji": [], "pq": [], "dq": [], "sq": []}
    t0 = time.time()
    for i, patch_idx in enumerate(sorted_indices):
        mat = os.path.join(MASKS_BASE, f"split{split}", mask_subdir, f"patch_{patch_idx:05d}.mat")
        if not os.path.exists(mat):
            continue
        pred_inst = sio.loadmat(mat)["inst_map"].astype(np.int32)
        gt_inst, _ = parse_lizard_masks(gt_masks_npy[i])
        acc, dice, se, sp = compute_pixel_metrics(gt_inst, pred_inst)
        aji = compute_aji(gt_inst, pred_inst)
        pq, dq, sq, _, _, _ = compute_pq(gt_inst, pred_inst)
        for k, v in zip(metrics.keys(), [acc, dice, se, sp, aji, pq, dq, sq]):
            metrics[k].append(float(v))
        if (i + 1) % 200 == 0:
            print(f"    [split {split}] {i+1}/{len(sorted_indices)}  t={time.time()-t0:.1f}s", flush=True)

    means = {k: float(np.mean(v) * 100) for k, v in metrics.items()}
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_subdir", default="masks_phase2_tta")
    ap.add_argument("--out", default="/home/sbarua/fanseg/predictions/lizard/_phase2_full_metrics_5fold.json")
    args = ap.parse_args()

    all_results = {}
    for s in [1, 2, 3, 4, 5]:
        print(f"\n=== split {s} (mask_subdir={args.mask_subdir}) ===")
        all_results[s] = eval_split(s, mask_subdir=args.mask_subdir)
        r = all_results[s]
        print(f"  ACC={r['acc']:.2f}  Dice={r['dice']:.2f}  SE={r['se']:.2f}  SP={r['sp']:.2f}  "
              f"AJI={r['aji']:.2f}  PQ={r['pq']:.2f}  DQ={r['dq']:.2f}  SQ={r['sq']:.2f}")

    print(f"\n{'='*92}")
    print(f"  5-FOLD MEAN — FAN-Seg Phase 2 + TTA — full segmentation metrics")
    print(f"{'='*92}")
    paper = {"acc": 99.64, "dice": 85.06, "se": 83.09, "sp": 99.82, "aji": 68.69, "pq": 70.76,
             "dq": None, "sq": None}
    repl  = {"acc": 93.81, "dice": 76.76, "se": 75.62, "sp": 96.43, "aji": 47.16, "pq": 52.09,
             "dq": 66.49, "sq": 77.24}
    phase1_tta = {"acc": 94.75, "dice": 80.65, "se": 81.95, "sp": 96.09, "aji": 65.86, "pq": 66.08,
                  "dq": None, "sq": None}
    print(f"  {'Metric':<6} {'Paper':>10} {'AGAF Repl':>11} {'P1+TTA':>10} {'P2+TTA':>14} {'Δ vs P1+TTA':>14} {'Δ vs Paper':>13}")
    print(f"  {'-'*6} {'-'*10} {'-'*11} {'-'*10} {'-'*14} {'-'*14} {'-'*13}")
    five_fold = {}
    for k in ["acc", "dice", "se", "sp", "aji", "pq", "dq", "sq"]:
        vals = [all_results[s][k] for s in [1, 2, 3, 4, 5]]
        m = float(np.mean(vals)); st = float(np.std(vals))
        five_fold[k] = {"mean": m, "std": st}
        p_s = f"{paper[k]:.2f}" if paper[k] is not None else "  —  "
        r_s = f"{repl[k]:.2f}" if repl[k] is not None else "  —  "
        t_s = f"{phase1_tta[k]:.2f}" if phase1_tta[k] is not None else "  —  "
        delta_p1 = m - phase1_tta[k] if phase1_tta[k] is not None else None
        delta_paper = m - paper[k] if paper[k] is not None else None
        d_p1_s = f"{'+' if delta_p1>=0 else ''}{delta_p1:.2f}" if delta_p1 is not None else "—"
        d_paper_s = f"{'+' if delta_paper>=0 else ''}{delta_paper:.2f}" if delta_paper is not None else "—"
        print(f"  {k.upper():<6} {p_s:>10} {r_s:>11} {t_s:>10} {m:>10.2f}±{st:.2f} {d_p1_s:>14} {d_paper_s:>13}")

    with open(args.out, "w") as f:
        json.dump({"per_split": all_results, "five_fold_mean": five_fold,
                   "paper": paper, "agafnet_replication": repl, "phase1_tta": phase1_tta}, f, indent=2)
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
