"""Phase 1 v2 trainer on Lizard — adds TYPE HEAD with focal class-balanced CE.

Same UNI+DPT architecture as train_lizard.py, with one extra head:
  - Type head: (C+1)-channel per-pixel class prediction (bg + 6 Lizard classes)
  - Loss: focal CE with inverse-frequency class weights (Lin et al. 2017 + Cui et al. 2019)

Per-nucleus class probability vector (C+1 dim) extracted at inference time and fed
as an additional feature into the fusion classifier (preserves the GNN+XGB architecture).

Usage:
    python train_lizard_v2.py --split 1 --epochs 60 --batch_size 8
"""
import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, "/home/sbarua/fanseg/code")
from encoder import UNIEncoder
from decoder_dpt import DPTDecoder
from lizard_loader import LizardDataset
from losses import (hv_loss, msge_loss, sem3_loss,
                   focal_class_balanced_ce, compute_inverse_freq_class_weights)


NUM_TYPE_CLASSES = 7   # 0 = bg, 1..6 = Lizard nucleus classes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, default=1, help="Lizard 5-fold split (1..5)")
    p.add_argument("--out_dir", default="/home/sbarua/fanseg/checkpoints/lizard_phase1_v2")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--lambda_hv",   type=float, default=2.0)
    p.add_argument("--lambda_msge", type=float, default=1.0)
    p.add_argument("--lambda_sem3", type=float, default=1.0)
    p.add_argument("--lambda_type", type=float, default=1.0,
                   help="Weight on the per-pixel type CE loss")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal modulation exponent for type loss (0 = no focal)")
    p.add_argument("--depths", type=str, default="5,11,17,23")
    return p.parse_args()


class FanSegLizardV2(nn.Module):
    """UNI ViT-L (frozen) + DPT decoder with HV + Sem3 + Type heads."""
    def __init__(self, num_type_classes=NUM_TYPE_CLASSES, depths=(5, 11, 17, 23), decoder_ch=128):
        super().__init__()
        self.encoder = UNIEncoder(freeze=True)
        self.decoder = DPTDecoder(in_dim=1024, ch=decoder_ch, num_type_classes=num_type_classes)
        self.depths = depths

    def forward(self, x):
        ms = self.encoder.forward_multiscale(x, depths=self.depths)
        return self.decoder(ms)   # (hv, sem3, type_logits)


def compute_class_weights_from_dataset(ds, num_classes, n_sample_patches=500):
    """Estimate inverse-frequency weights from a sample of training patches.
    Iterates n_sample_patches random patches, counts per-class pixels in their cls maps.
    """
    print(f"  [class weights] sampling {n_sample_patches} patches for inverse-freq weights...")
    counts = {c: 0 for c in range(num_classes)}
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(ds), size=min(n_sample_patches, len(ds)), replace=False)
    for k, i in enumerate(idxs):
        cls_map = ds[int(i)]["cls"].numpy()
        for c in range(num_classes):
            counts[c] += int((cls_map == c).sum())
        if (k + 1) % 100 == 0:
            print(f"    [{k+1}/{len(idxs)}]")
    w = compute_inverse_freq_class_weights(counts, smooth=1.0, normalize=True)
    print(f"  per-class pixel counts (sampled): {counts}")
    print(f"  inverse-freq weights (normalized mean=1): {w.tolist()}")
    return w, counts


def main():
    args = parse_args()
    out_dir = os.path.join(args.out_dir, f"split{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    depths = tuple(int(s) for s in args.depths.split(","))
    print(f"[init] device={device} split={args.split} epochs={args.epochs} bs={args.batch_size} "
          f"depths={depths} decoder_ch={args.decoder_ch}")
    print(f"[init] num_type_classes={NUM_TYPE_CLASSES} (bg + 6 Lizard classes)")
    print(f"[init] lambda_type={args.lambda_type}  focal_gamma={args.focal_gamma}")

    train_ds = LizardDataset(split=args.split, mode="train", augment=True)
    val_ds   = LizardDataset(split=args.split, mode="test",  augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"[data] train_patches={len(train_ds)} val_patches={len(val_ds)}")

    # Compute class weights for type head (inverse frequency)
    cls_weights, cls_counts = compute_class_weights_from_dataset(train_ds, NUM_TYPE_CLASSES)
    cls_weights_t = cls_weights.to(device)

    model = FanSegLizardV2(num_type_classes=NUM_TYPE_CLASSES, depths=depths, decoder_ch=args.decoder_ch).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_train/1e6:.2f}M (encoder frozen)")

    sem3_weight = torch.tensor([1.0, 1.5, 2.0], device=device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    history = []
    best_val = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ep_l = ep_hv = ep_ms = ep_sem = ep_type = 0.0; n_b = 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            hv  = batch["hv"].to(device, non_blocking=True)
            sem = batch["sem"].to(device, non_blocking=True)
            fg  = batch["fg_mask"].to(device, non_blocking=True)
            cls = batch["cls"].to(device, non_blocking=True)         # (B, H, W) int64
            hv_pred, sem3_pred, type_logits = model(img)
            l_hv   = hv_loss(hv_pred, hv, fg)
            l_msge = msge_loss(hv_pred, hv, fg)
            l_sem3, _, _ = sem3_loss(sem3_pred, sem, weight=sem3_weight)
            l_type = focal_class_balanced_ce(type_logits, cls, cls_weights_t, gamma=args.focal_gamma)
            loss = (args.lambda_hv*l_hv + args.lambda_msge*l_msge
                  + args.lambda_sem3*l_sem3 + args.lambda_type*l_type)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step(); sched.step()
            ep_l += loss.item(); ep_hv += l_hv.item(); ep_ms += l_msge.item()
            ep_sem += l_sem3.item(); ep_type += l_type.item()
            n_b += 1

        model.eval()
        v_l = v_hv = v_ms = v_sem = v_type = 0.0; v_n = 0
        with torch.inference_mode():
            for batch in val_loader:
                img = batch["image"].to(device); hv = batch["hv"].to(device)
                sem = batch["sem"].to(device);   fg = batch["fg_mask"].to(device)
                cls = batch["cls"].to(device)
                hv_pred, sem3_pred, type_logits = model(img)
                l_hv   = hv_loss(hv_pred, hv, fg)
                l_msge = msge_loss(hv_pred, hv, fg)
                l_sem3, _, _ = sem3_loss(sem3_pred, sem, weight=sem3_weight)
                l_type = focal_class_balanced_ce(type_logits, cls, cls_weights_t, gamma=args.focal_gamma)
                v_l += (args.lambda_hv*l_hv + args.lambda_msge*l_msge
                      + args.lambda_sem3*l_sem3 + args.lambda_type*l_type).item()
                v_hv += l_hv.item(); v_ms += l_msge.item(); v_sem += l_sem3.item(); v_type += l_type.item()
                v_n += 1

        rec = {
            "epoch": ep + 1,
            "train_loss": ep_l/n_b, "train_hv": ep_hv/n_b, "train_msge": ep_ms/n_b,
            "train_sem3": ep_sem/n_b, "train_type": ep_type/n_b,
            "val_loss":   v_l/max(v_n,1), "val_hv": v_hv/max(v_n,1), "val_msge": v_ms/max(v_n,1),
            "val_sem3": v_sem/max(v_n,1), "val_type": v_type/max(v_n,1),
            "lr": sched.get_last_lr()[0], "t_min": (time.time()-t0)/60,
        }
        history.append(rec)
        print(f"[ep {ep+1:02d}/{args.epochs}] "
              f"train: l={rec['train_loss']:.4f} hv={rec['train_hv']:.4f} msge={rec['train_msge']:.4f} "
              f"sem={rec['train_sem3']:.4f} type={rec['train_type']:.4f} "
              f"| val: l={rec['val_loss']:.4f} type={rec['val_type']:.4f} "
              f"| {rec['t_min']:.1f}min", flush=True)
        if rec["val_loss"] < best_val:
            best_val = rec["val_loss"]
            torch.save({"model": model.state_dict(), "epoch": ep+1, "val_loss": best_val,
                        "args": vars(args), "depths": depths, "decoder_ch": args.decoder_ch,
                        "num_type_classes": NUM_TYPE_CLASSES,
                        "class_weights": cls_weights.tolist(), "class_counts": cls_counts},
                       os.path.join(out_dir, "best.pt"))

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args),
                "depths": depths, "decoder_ch": args.decoder_ch,
                "num_type_classes": NUM_TYPE_CLASSES},
               os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val_loss = {best_val:.4f}  total {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
