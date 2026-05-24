"""TTA inference for Phase 2 (loads G from {G_state_dict, D_state_dict} checkpoint format).

Same 8-augmentation TTA as inference_lizard_tta.py but with Phase 2 checkpoint loader.
Outputs masks_phase2_tta/ per-split + per-split eval summary JSON.
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import scipy.io as sio
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/sbarua/fanseg/code")
from encoder import UNIEncoder
from decoder_dpt import DPTDecoder
from lizard_loader import LizardDataset
from inference_phase0 import get_fast_pq
from eval_lizard_full_metrics import compute_aji, compute_pixel_metrics
from inference_lizard_tta import AUGMENTS, apply_aug, invert_aug_hv, invert_aug_sem3
from inference_lizard import watershed_postproc


class FanSegLizard(torch.nn.Module):
    def __init__(self, depths=(5, 11, 17, 23), decoder_ch=128):
        super().__init__()
        self.encoder = UNIEncoder(freeze=True)
        self.decoder = DPTDecoder(in_dim=1024, ch=decoder_ch)
        self.depths = depths
    def forward(self, x):
        ms = self.encoder.forward_multiscale(x, depths=self.depths)
        return self.decoder(ms)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out_subdir", default="masks_phase2_tta")
    p.add_argument("--center_thresh", type=float, default=0.5)
    p.add_argument("--bg_thresh", type=float, default=0.4)
    p.add_argument("--min_size", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.ckpt is None:
        args.ckpt = f"/home/sbarua/fanseg/checkpoints/lizard_phase2/split{args.split}/best.pt"
    out_dir = f"/home/sbarua/fanseg/predictions/lizard/split{args.split}/{args.out_subdir}"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"[init] split={args.split} ckpt={args.ckpt}")
    cp = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Phase 2 ckpt has G_state_dict (and D_state_dict, ignored)
    if "G_state_dict" in cp:
        g_state = cp["G_state_dict"]
    else:
        g_state = cp["model"]   # fallback (Phase 1 format)
    args_ckpt = cp.get("args", {})
    depths = tuple(int(s) for s in args_ckpt.get("depths", "5,11,17,23").split(",")) \
             if isinstance(args_ckpt.get("depths"), str) else (5, 11, 17, 23)
    decoder_ch = args_ckpt.get("decoder_ch", 128)
    model = FanSegLizard(depths=depths, decoder_ch=decoder_ch).to(device)
    model.load_state_dict(g_state)
    model.eval()
    print(f"[model] loaded epoch {cp.get('epoch', '?')} val_loss {cp.get('val_loss', '?')}")

    ds = LizardDataset(split=args.split, mode="test", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[data] {len(ds)} test patches; TTA = {len(AUGMENTS)} augs")

    pqs, ajis, dices, accs = [], [], [], []
    t0 = time.time()

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            img = batch["image"].to(device)
            gt_inst_b = batch["inst_map"].cpu().numpy()
            patch_idx_b = batch["patch_idx"].cpu().numpy()
            B = img.shape[0]

            hv_sum   = torch.zeros(B, 2, 256, 256, device=device)
            sem3_sum = torch.zeros(B, 3, 256, 256, device=device)
            for k, hflip in AUGMENTS:
                aug_img = apply_aug(img, k, hflip)
                hv_aug, sem3_aug = model(aug_img)
                sem3_aug = torch.softmax(sem3_aug, dim=1)
                hv_inv   = invert_aug_hv(hv_aug.clone(),     k, hflip)
                sem3_inv = invert_aug_sem3(sem3_aug.clone(), k, hflip)
                hv_sum   += hv_inv
                sem3_sum += sem3_inv

            hv_avg   = (hv_sum   / len(AUGMENTS)).cpu().numpy()
            sem3_avg = (sem3_sum / len(AUGMENTS)).cpu().numpy()

            for b in range(B):
                inst = watershed_postproc(hv_avg[b], sem3_avg[b], args)
                pq, dq, sq = get_fast_pq(gt_inst_b[b], inst)
                aji = compute_aji(gt_inst_b[b], inst)
                acc, dice, se, sp = compute_pixel_metrics(gt_inst_b[b], inst)
                pqs.append(pq); ajis.append(aji); dices.append(dice); accs.append(acc)
                sio.savemat(
                    os.path.join(out_dir, f"patch_{int(patch_idx_b[b]):05d}.mat"),
                    {"inst_map": inst, "type_map": np.zeros_like(inst, dtype=np.int32)},
                    do_compression=True,
                )
            if (bi + 1) % 20 == 0:
                print(f"  [batch {bi+1}/{len(loader)}] t={time.time()-t0:.1f}s", flush=True)

    pq_m = float(np.mean(pqs)); aji_m = float(np.mean(ajis))
    dice_m = float(np.mean(dices)); acc_m = float(np.mean(accs))
    print()
    print("=" * 75)
    print(f"  Lizard split {args.split} — Phase 2 + TTA results")
    print("=" * 75)
    print(f"  Binary PQ   : {pq_m*100:.2f}")
    print(f"  Binary AJI  : {aji_m*100:.2f}")
    print(f"  Pixel Dice  : {dice_m*100:.2f}")
    print(f"  Pixel ACC   : {acc_m*100:.2f}")
    print(f"  total time  : {(time.time()-t0)/60:.1f} min")

    out_json = f"/home/sbarua/fanseg/predictions/lizard/split{args.split}/_phase2_tta_eval_summary.json"
    with open(out_json, "w") as f:
        json.dump({"split": args.split, "n_augs": len(AUGMENTS),
                   "pq": pq_m, "aji": aji_m, "dice": dice_m, "acc": acc_m,
                   "wall_min": (time.time() - t0) / 60}, f, indent=2)
    print(f"  saved → {out_json}")


if __name__ == "__main__":
    main()
