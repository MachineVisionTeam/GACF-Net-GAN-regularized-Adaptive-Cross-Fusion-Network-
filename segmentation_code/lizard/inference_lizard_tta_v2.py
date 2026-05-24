"""Phase 1 v2 TTA inference on Lizard — includes type head.

Outputs:
  1. Standard masks_tta_v2/ .mat files (inst_map, type_map) for segmentation eval
  2. per_nucleus_type_features.h5 with (C+1)-dim mean type probability vector per nucleus.
     This h5 is fed to the fusion classifier as an additional feature.

Uses Phase 1 v2 checkpoint (model has hv + sem3 + type heads).
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import h5py
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/sbarua/fanseg/code")
from encoder import UNIEncoder
from decoder_dpt import DPTDecoder
from lizard_loader import LizardDataset
from inference_phase0 import get_fast_pq
from eval_lizard_full_metrics import compute_aji, compute_pixel_metrics
from inference_lizard import watershed_postproc
from inference_lizard_tta import AUGMENTS, apply_aug, invert_aug_hv, invert_aug_sem3


NUM_TYPE_CLASSES = 7


class FanSegLizardV2(nn.Module):
    def __init__(self, num_type_classes=NUM_TYPE_CLASSES, depths=(5, 11, 17, 23), decoder_ch=128):
        super().__init__()
        self.encoder = UNIEncoder(freeze=True)
        self.decoder = DPTDecoder(in_dim=1024, ch=decoder_ch, num_type_classes=num_type_classes)
        self.depths = depths
    def forward(self, x):
        ms = self.encoder.forward_multiscale(x, depths=self.depths)
        return self.decoder(ms)


def invert_aug_type(t, k, hflip):
    """Spatial inverse for type prob map (same as sem3 — channels are class probs)."""
    if hflip: t = torch.flip(t, dims=[-1])
    if k > 0: t = torch.rot90(t, k=-k, dims=[-2, -1])
    return t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out_subdir", default="masks_tta_v2")
    p.add_argument("--center_thresh", type=float, default=0.5)
    p.add_argument("--bg_thresh", type=float, default=0.4)
    p.add_argument("--min_size", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.ckpt is None:
        args.ckpt = f"/home/sbarua/fanseg/checkpoints/lizard_phase1_v2/split{args.split}/best.pt"
    out_mask_dir = f"/home/sbarua/fanseg/predictions/lizard/split{args.split}/{args.out_subdir}"
    out_type_h5  = f"/home/sbarua/fanseg/predictions/lizard/split{args.split}/per_nucleus_type_features_v2.h5"
    os.makedirs(out_mask_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"[init] split={args.split} ckpt={args.ckpt}")
    cp = torch.load(args.ckpt, map_location=device, weights_only=False)
    depths = tuple(cp.get("depths", (5, 11, 17, 23)))
    decoder_ch = cp.get("decoder_ch", 128)
    num_type = cp.get("num_type_classes", NUM_TYPE_CLASSES)
    model = FanSegLizardV2(num_type_classes=num_type, depths=depths, decoder_ch=decoder_ch).to(device)
    model.load_state_dict(cp["model"])
    model.eval()
    print(f"[model] loaded epoch {cp.get('epoch', '?')} val_loss {cp.get('val_loss', '?'):.4f}")
    print(f"[model] num_type_classes={num_type}")

    ds = LizardDataset(split=args.split, mode="test", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[data] {len(ds)} test patches; TTA = 8 augs")

    pqs, ajis, dices, accs = [], [], [], []
    # Per-nucleus type features accumulator (saved as h5 at the end)
    nuc_rows = {"patch_idx": [], "nuc_id": [], "type_probs": []}
    t0 = time.time()

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            img = batch["image"].to(device)
            gt_inst_b = batch["inst_map"].cpu().numpy()
            patch_idx_b = batch["patch_idx"].cpu().numpy()
            B = img.shape[0]

            hv_sum   = torch.zeros(B, 2, 256, 256, device=device)
            sem3_sum = torch.zeros(B, 3, 256, 256, device=device)
            type_sum = torch.zeros(B, num_type, 256, 256, device=device)
            for k, hflip in AUGMENTS:
                aug_img = apply_aug(img, k, hflip)
                hv_aug, sem3_aug, type_aug = model(aug_img)
                sem3_aug = torch.softmax(sem3_aug, dim=1)
                type_aug = torch.softmax(type_aug, dim=1)
                hv_sum   += invert_aug_hv(hv_aug.clone(),     k, hflip)
                sem3_sum += invert_aug_sem3(sem3_aug.clone(), k, hflip)
                type_sum += invert_aug_type(type_aug.clone(), k, hflip)

            hv_avg   = (hv_sum   / len(AUGMENTS)).cpu().numpy()
            sem3_avg = (sem3_sum / len(AUGMENTS)).cpu().numpy()
            type_avg = (type_sum / len(AUGMENTS)).cpu().numpy()   # (B, C+1, 256, 256)

            for b in range(B):
                inst = watershed_postproc(hv_avg[b], sem3_avg[b], args)
                # Eval
                pq, dq, sq = get_fast_pq(gt_inst_b[b], inst)
                aji = compute_aji(gt_inst_b[b], inst)
                acc, dice, se, sp = compute_pixel_metrics(gt_inst_b[b], inst)
                pqs.append(pq); ajis.append(aji); dices.append(dice); accs.append(acc)

                # Per-nucleus type prob extraction
                type_probs = type_avg[b]   # (C+1, 256, 256)
                # For majority-vote type_map (saved for downstream stage_phase2_for_fusion)
                type_pred_per_pixel = type_probs.argmax(axis=0)
                nuc_ids_in_patch = np.unique(inst); nuc_ids_in_patch = nuc_ids_in_patch[nuc_ids_in_patch > 0]
                type_map_inst = np.zeros_like(inst, dtype=np.int32)
                for nid in nuc_ids_in_patch:
                    mask = (inst == nid)
                    if mask.sum() == 0: continue
                    # Mean prob vector over mask pixels
                    pv = type_probs[:, mask].mean(axis=-1).astype(np.float32)  # (C+1,)
                    nuc_rows["patch_idx"].append(int(patch_idx_b[b]))
                    nuc_rows["nuc_id"].append(int(nid))
                    nuc_rows["type_probs"].append(pv)
                    # Type map = argmax (for saved .mat type_map field)
                    cls_for_inst = int(pv.argmax())
                    type_map_inst[mask] = cls_for_inst

                # Save .mat with inst + type
                sio.savemat(
                    os.path.join(out_mask_dir, f"patch_{int(patch_idx_b[b]):05d}.mat"),
                    {"inst_map": inst.astype(np.int32), "type_map": type_map_inst},
                    do_compression=True,
                )
            if (bi + 1) % 20 == 0:
                print(f"  [batch {bi+1}/{len(loader)}] t={time.time()-t0:.1f}s", flush=True)

    pq_m = float(np.mean(pqs)); aji_m = float(np.mean(ajis))
    dice_m = float(np.mean(dices)); acc_m = float(np.mean(accs))
    print()
    print("=" * 75)
    print(f"  Lizard split {args.split} v2 — TTA (8 augs) results")
    print("=" * 75)
    print(f"  Binary PQ   : {pq_m*100:.2f}")
    print(f"  Binary AJI  : {aji_m*100:.2f}")
    print(f"  Pixel Dice  : {dice_m*100:.2f}")
    print(f"  Pixel ACC   : {acc_m*100:.2f}")
    print(f"  total time  : {(time.time()-t0)/60:.1f} min")

    # Save per-nucleus type features
    type_probs_arr = np.stack(nuc_rows["type_probs"]).astype(np.float32)  # (N, C+1)
    with h5py.File(out_type_h5, "w") as f:
        f.create_dataset("patch_idx", data=np.array(nuc_rows["patch_idx"], dtype=np.int32))
        f.create_dataset("nuc_id",    data=np.array(nuc_rows["nuc_id"],   dtype=np.int32))
        f.create_dataset("type_probs", data=type_probs_arr)
        f.attrs["num_type_classes"] = int(num_type)
    print(f"  saved per-nucleus type features ({type_probs_arr.shape}) -> {out_type_h5}")

    out_json = f"/home/sbarua/fanseg/predictions/lizard/split{args.split}/_tta_v2_eval_summary.json"
    with open(out_json, "w") as f:
        json.dump({"split": args.split, "n_augs": len(AUGMENTS),
                   "pq": pq_m, "aji": aji_m, "dice": dice_m, "acc": acc_m,
                   "wall_min": (time.time() - t0) / 60}, f, indent=2)
    print(f"  saved -> {out_json}")


if __name__ == "__main__":
    main()
