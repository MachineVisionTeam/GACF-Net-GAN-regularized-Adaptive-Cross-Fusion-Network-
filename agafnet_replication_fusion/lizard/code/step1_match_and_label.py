"""Step 1 — Match AGAFNet predicted instances to GT, build per-nucleus label table.

For every AGAFNet-detected nucleus (across all 5 Lizard folds) we record:
  fold_id, local_patch_idx, global_patch_idx, pred_inst_id
  centroid_y, centroid_x, ymin, xmin, ymax, xmax, area
  iou_with_gt, gt_class  (0 if unmatched at IoU>=0.5)
  agafnet_class          (AGAFNet's own majority-vote class over type_N pixels)

This is the foundation for the controlled experiment:
  - gt_class      -> used to train/eval ANY classifier on the AGAFNet-detected set
  - agafnet_class -> AGAFNet's own per-nucleus classification (the baseline to beat)

Output: /home/sbarua/fanseg/agafnet_mask_fusion/features/agafnet_nucleus_table.h5
"""
import os, time
import numpy as np
import h5py

AGAF_MASKS = '/home/sbarua/Lizard/agafnet_replication/fusion/features/instance_maps/fold{N}_inst_maps.npz'
GT_MASKS   = '/mnt/storage1/Lizard/agafnet_replication/data_conic/fold{N}/masks.npy'
METADATA   = '/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy'
OUT_H5     = '/home/sbarua/fanseg/agafnet_mask_fusion/features/agafnet_nucleus_table.h5'

NUM_CLASSES = 6
IOU_THRESH  = 0.5
FOLDS = [1, 2, 3, 4, 5]


def build_gt_instance_map(gt_patch):
    """From 7-channel GT (256,256,7) build (inst_map, {inst_id: class})."""
    gt_inst = np.zeros((256, 256), dtype=np.int32)
    gt_class_of = {}
    gid = 0
    for ch in range(NUM_CLASSES):          # channels 0..5 = classes 1..6
        chan = gt_patch[:, :, ch].astype(np.int32)
        for oid in np.unique(chan):
            if oid == 0: continue
            gid += 1
            gt_inst[chan == oid] = gid
            gt_class_of[gid] = ch + 1
    return gt_inst, gt_class_of


def main():
    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)
    meta = np.load(METADATA, allow_pickle=True).item()
    fa = meta['fold_assignments']

    rows = {k: [] for k in [
        'fold_id', 'local_patch', 'global_patch', 'pred_inst_id',
        'centroid_y', 'centroid_x', 'ymin', 'xmin', 'ymax', 'xmax', 'area',
        'iou_with_gt', 'gt_class', 'agafnet_class',
    ]}

    t0 = time.time()
    for fold in FOLDS:
        d  = np.load(AGAF_MASKS.format(N=fold))
        gt = np.load(GT_MASKS.format(N=fold), mmap_mode='r')
        fold_globals = sorted([k for k, v in fa.items() if v == fold])
        n_patches = len([k for k in d.files if k.startswith('inst_')])
        assert n_patches == len(gt) == len(fold_globals), \
            f"fold{fold} mismatch: agaf={n_patches} gt={len(gt)} globals={len(fold_globals)}"

        fold_nuclei = 0
        for local in range(n_patches):
            inst_pred = d[f'inst_{local}']
            type_pred = d[f'type_{local}']
            gt_inst, gt_class_of = build_gt_instance_map(gt[local])
            global_idx = fold_globals[local]

            n_pred = int(inst_pred.max())
            for pid in range(1, n_pred + 1):
                pm = inst_pred == pid
                area = int(pm.sum())
                if area < 5:                     # skip noise specks
                    continue
                ys, xs = np.where(pm)
                cy, cx = float(ys.mean()), float(xs.mean())
                ymin, xmin = int(ys.min()), int(xs.min())
                ymax, xmax = int(ys.max()), int(xs.max())

                # IoU match to GT
                overlap = np.unique(gt_inst[pm]); overlap = overlap[overlap > 0]
                best_iou, best_gid = 0.0, 0
                for g in overlap:
                    gm = gt_inst == g
                    iou = (pm & gm).sum() / max((pm | gm).sum(), 1)
                    if iou > best_iou:
                        best_iou, best_gid = float(iou), int(g)
                gt_cls = gt_class_of[best_gid] if (best_iou >= IOU_THRESH and best_gid > 0) else 0

                # AGAFNet's own class = majority vote over type_pred pixels
                votes = type_pred[pm]; votes = votes[(votes > 0) & (votes <= NUM_CLASSES)]
                agaf_cls = int(np.bincount(votes).argmax()) if len(votes) > 0 else 0

                rows['fold_id'].append(fold)
                rows['local_patch'].append(local)
                rows['global_patch'].append(int(global_idx))
                rows['pred_inst_id'].append(pid)
                rows['centroid_y'].append(cy); rows['centroid_x'].append(cx)
                rows['ymin'].append(ymin); rows['xmin'].append(xmin)
                rows['ymax'].append(ymax); rows['xmax'].append(xmax)
                rows['area'].append(area)
                rows['iou_with_gt'].append(best_iou)
                rows['gt_class'].append(gt_cls)
                rows['agafnet_class'].append(agaf_cls)
                fold_nuclei += 1
        print(f"fold{fold}: {fold_nuclei} AGAFNet nuclei  ({time.time()-t0:.1f}s)", flush=True)

    # Save
    with h5py.File(OUT_H5, 'w') as f:
        for k, v in rows.items():
            arr = np.asarray(v)
            f.create_dataset(k, data=arr)
        f.attrs['iou_thresh'] = IOU_THRESH
        f.attrs['source'] = 'AGAFNet replication predicted instance maps (Lizard)'

    N = len(rows['fold_id'])
    gt_class = np.asarray(rows['gt_class'])
    matched = (gt_class > 0).sum()
    print(f"\n[done] {N:,} AGAFNet nuclei total")
    print(f"  matched to GT (IoU>=0.5): {matched:,} ({100*matched/N:.1f}%)")
    print(f"  unmatched (FP_d):         {N-matched:,} ({100*(N-matched)/N:.1f}%)")
    print(f"  class dist (matched): {np.bincount(gt_class[gt_class>0], minlength=NUM_CLASSES+1)[1:].tolist()}")
    print(f"[saved] {OUT_H5}")


if __name__ == '__main__':
    main()
