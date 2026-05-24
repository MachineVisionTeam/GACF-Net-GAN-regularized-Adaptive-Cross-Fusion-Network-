"""Full segmentation metrics on PanNuke (3-fold)."""
import os, sys, time, json, argparse
import numpy as np
import scipy.io as sio

sys.path.insert(0, "/home/sbarua/fanseg/code")
from eval_lizard_full_metrics import compute_pq, compute_aji, compute_pixel_metrics

LABELS_ROOT = "/home/sbarua/Panuke/hover_net_x_panuke/data/masks"
PRED_BASE   = "/home/sbarua/fanseg/predictions/pannuke"


def eval_split(split, mask_subdir="masks_tta"):
    gt_labels = np.load(os.path.join(LABELS_ROOT, f"fold{split}", "labels.npy"), mmap_mode="r")
    n_patches = len(gt_labels)
    metrics = {"acc":[],"dice":[],"se":[],"sp":[],"aji":[],"pq":[],"dq":[],"sq":[]}
    t0 = time.time(); n_missing = 0
    for i in range(n_patches):
        mat = os.path.join(PRED_BASE, f"split{split}", mask_subdir, f"patch_{i:05d}.mat")
        if not os.path.exists(mat):
            n_missing += 1; continue
        pred_inst = sio.loadmat(mat)["inst_map"].astype(np.int32)
        gt_inst = gt_labels[i, :, :, 0].astype(np.int32)
        acc, dice, se, sp = compute_pixel_metrics(gt_inst, pred_inst)
        aji = compute_aji(gt_inst, pred_inst)
        pq, dq, sq, _, _, _ = compute_pq(gt_inst, pred_inst)
        for k, v in zip(metrics.keys(), [acc, dice, se, sp, aji, pq, dq, sq]):
            metrics[k].append(float(v))
        if (i + 1) % 200 == 0:
            print(f"    [split {split}] {i+1}/{n_patches}  t={time.time()-t0:.1f}s", flush=True)
    means = {k: float(np.mean(v) * 100) for k, v in metrics.items()}
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_subdir", default="masks_tta")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        suffix = "phase2" if "phase2" in args.mask_subdir else "phase1"
        args.out = f"/home/sbarua/fanseg/predictions/pannuke/_{suffix}_full_metrics_3fold.json"

    all_results = {}
    for s in [1, 2, 3]:
        print(f"\n=== split {s} (mask_subdir={args.mask_subdir}) ===")
        all_results[s] = eval_split(s, mask_subdir=args.mask_subdir)
        r = all_results[s]
        print(f"  ACC={r['acc']:.2f} Dice={r['dice']:.2f} SE={r['se']:.2f} SP={r['sp']:.2f} "
              f"AJI={r['aji']:.2f} PQ={r['pq']:.2f} DQ={r['dq']:.2f} SQ={r['sq']:.2f}")

    print(f"\n{'='*92}\n  3-FOLD MEAN — PanNuke — {args.mask_subdir}\n{'='*92}")
    # AGAFNet paper PanNuke values (Table IV, page 105)
    paper = {"acc":98.70,"dice":82.72,"se":83.94,"sp":99.49,"aji":81.49,"pq":76.82,"dq":None,"sq":None}
    print(f"  {'Metric':<6} {'AGAFNet Paper':>14} {'Ours (3-fold)':>18}   {'Δ vs Paper':>12}")
    print(f"  {'-'*6} {'-'*14} {'-'*18}   {'-'*12}")
    three_fold = {}
    for k in ["acc","dice","se","sp","aji","pq","dq","sq"]:
        vals = [all_results[s][k] for s in [1, 2, 3]]
        m = float(np.mean(vals)); st = float(np.std(vals))
        three_fold[k] = {"mean": m, "std": st}
        p_s = f"{paper[k]:.2f}" if paper[k] is not None else "  —  "
        d_s = f"{m - paper[k]:+.2f}" if paper[k] is not None else "—"
        print(f"  {k.upper():<6} {p_s:>14} {m:>13.2f}±{st:.2f}   {d_s:>12}")

    with open(args.out, "w") as f:
        json.dump({"per_split": all_results, "three_fold_mean": three_fold,
                   "agafnet_paper": paper, "mask_subdir": args.mask_subdir}, f, indent=2)
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
