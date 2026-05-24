"""Full segmentation metrics for FAN-Seg on Lizard 5-fold: ACC, Dice, SE, SP, AJI, PQ, DQ, SQ.

Matches the exact protocol AGAFNet's eval_v4_all5.py uses, so numbers are directly
comparable to the replication report.
"""
import os, time, json, argparse
import numpy as np
import scipy.io as sio
from scipy.optimize import linear_sum_assignment


METADATA_PATH = "/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy"
DATA_DIR      = "/mnt/storage1/Lizard/agafnet_replication/data_conic"
MASKS_BASE    = "/home/sbarua/fanseg/predictions/lizard"

NUM_CLASSES = 6


def parse_lizard_masks(masks_7ch):
    H, W = masks_7ch.shape[:2]
    inst = np.zeros((H, W), dtype=np.int32)
    typ = np.zeros((H, W), dtype=np.int32)
    cid = 0
    for ch in range(NUM_CLASSES):
        channel = masks_7ch[:, :, ch].astype(np.int32)
        for old_id in np.unique(channel):
            if old_id == 0: continue
            cid += 1
            mask = channel == old_id
            inst[mask] = cid
            typ[mask] = ch + 1
    return inst, typ


def get_instances(m):
    return [(i, m == i) for i in range(1, m.max() + 1) if (m == i).sum() > 0]


def compute_pq(gt, pred, thresh=0.5):
    gi = get_instances(gt); pi = get_instances(pred)
    if not gi and not pi: return 1, 1, 1, 0, 0, 0
    if not gi: return 0, 0, 0, 0, len(pi), 0
    if not pi: return 0, 0, 0, 0, 0, len(gi)
    iou_m = np.zeros((len(gi), len(pi)))
    for i, (_, gm) in enumerate(gi):
        for j, (_, pm) in enumerate(pi):
            inter = np.logical_and(gm, pm).sum()
            if inter == 0: continue
            iou_m[i, j] = inter / (np.logical_or(gm, pm).sum() + 1e-8)
    ri, ci = linear_sum_assignment(1 - iou_m)
    iou_sum, tp, mg, mp = 0.0, 0, set(), set()
    for r, c in zip(ri, ci):
        if iou_m[r, c] >= thresh:
            tp += 1; iou_sum += iou_m[r, c]; mg.add(r); mp.add(c)
    fp = len(pi) - len(mp); fn = len(gi) - len(mg)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    sq = iou_sum / (tp + 1e-8) if tp else 0
    return dq * sq, dq, sq, tp, fp, fn


def compute_aji(gt, pred):
    gids = np.unique(gt); gids = gids[gids > 0]
    pids = np.unique(pred); pids = pids[pids > 0]
    if not len(gids) and not len(pids): return 1.0
    if not len(gids) or not len(pids): return 0.0
    ti, tu, matched = 0.0, 0.0, set()
    for gid in gids:
        gm = gt == gid; best_iou, best_pid, best_i, best_u = 0, None, 0, 0
        for pid in np.unique(pred[gm]):
            if pid == 0: continue
            pm = pred == pid
            inter = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            iou = inter / (union + 1e-8)
            if iou > best_iou:
                best_iou, best_pid, best_i, best_u = iou, pid, inter, union
        if best_pid is not None:
            ti += best_i; tu += best_u; matched.add(best_pid)
        else:
            tu += gm.sum()
    for pid in pids:
        if pid not in matched:
            tu += (pred == pid).sum()
    return ti / (tu + 1e-8)


def compute_pixel_metrics(gt, pred):
    gf = (gt > 0).astype(int)
    pf = (pred > 0).astype(int)
    tp = np.sum(gf * pf); tn = np.sum((1 - gf) * (1 - pf))
    fp = np.sum((1 - gf) * pf); fn = np.sum(gf * (1 - pf))
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    se   = tp / (tp + fn + 1e-8)
    sp   = tn / (tn + fp + 1e-8)
    return acc, dice, se, sp


def eval_split(split):
    """Compute full metrics for one fold."""
    meta = np.load(METADATA_PATH, allow_pickle=True).item()
    fa = meta["fold_assignments"]
    sorted_indices = sorted([k for k, v in fa.items() if v == split])
    gt_masks_npy = np.load(os.path.join(DATA_DIR, f"fold{split}", "masks.npy"))
    assert len(gt_masks_npy) == len(sorted_indices)

    metrics = {"acc": [], "dice": [], "se": [], "sp": [],
               "aji": [], "pq": [], "dq": [], "sq": []}
    t0 = time.time()
    for i, patch_idx in enumerate(sorted_indices):
        mat = os.path.join(MASKS_BASE, f"split{split}", "masks", f"patch_{patch_idx:05d}.mat")
        if not os.path.exists(mat):
            continue
        pred_inst = sio.loadmat(mat)["inst_map"].astype(np.int32)
        gt_inst, _ = parse_lizard_masks(gt_masks_npy[i])
        # Pixel metrics on instance map binarized
        acc, dice, se, sp = compute_pixel_metrics(gt_inst, pred_inst)
        # Instance metrics
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
    ap.add_argument("--out", default="/home/sbarua/fanseg/predictions/lizard/_full_metrics_5fold.json")
    args = ap.parse_args()

    all_results = {}
    for s in [1, 2, 3, 4, 5]:
        print(f"\n=== split {s} ===")
        all_results[s] = eval_split(s)
        print(f"  ACC={all_results[s]['acc']:.2f}  Dice={all_results[s]['dice']:.2f}  "
              f"SE={all_results[s]['se']:.2f}  SP={all_results[s]['sp']:.2f}  "
              f"AJI={all_results[s]['aji']:.2f}  PQ={all_results[s]['pq']:.2f}  "
              f"DQ={all_results[s]['dq']:.2f}  SQ={all_results[s]['sq']:.2f}")

    # 5-fold mean ± std
    print(f"\n{'='*92}")
    print(f"  5-FOLD MEAN — FAN-Seg Phase 1 (UNI + DPT, no GAN) — full segmentation metrics")
    print(f"{'='*92}")
    paper = {"acc": 99.64, "dice": 85.06, "se": 83.09, "sp": 99.82, "aji": 68.69, "pq": 70.76, "dq": None, "sq": None}
    repl  = {"acc": 93.81, "dice": 76.76, "se": 75.62, "sp": 96.43, "aji": 47.16, "pq": 52.09, "dq": 66.49, "sq": 77.24}
    print(f"  {'Metric':<6} {'Paper':>10} {'AGAF Repl':>11} {'FAN-Seg P1':>14} {'Δ vs Repl':>11}")
    print(f"  {'-'*6} {'-'*10:>10} {'-'*11:>11} {'-'*14:>14} {'-'*11:>11}")
    five_fold = {}
    for k in ["acc", "dice", "se", "sp", "aji", "pq", "dq", "sq"]:
        vals = [all_results[s][k] for s in [1, 2, 3, 4, 5]]
        m = float(np.mean(vals)); st = float(np.std(vals))
        five_fold[k] = {"mean": m, "std": st}
        p_s = f"{paper[k]:.2f}" if paper[k] is not None else "  —  "
        r_s = f"{repl[k]:.2f}" if repl[k] is not None else "  —  "
        delta = m - repl[k] if repl[k] is not None else None
        d_s = f"{'+' if delta>=0 else ''}{delta:.2f}" if delta is not None else "—"
        print(f"  {k.upper():<6} {p_s:>10} {r_s:>11} {m:>10.2f}±{st:.2f} {d_s:>11}")

    with open(args.out, "w") as f:
        json.dump({"per_split": all_results, "five_fold_mean": five_fold,
                   "paper": paper, "agafnet_replication": repl}, f, indent=2)
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    main()
