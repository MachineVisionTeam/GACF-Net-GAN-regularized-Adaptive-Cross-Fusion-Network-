"""Phase 2 (BHF-Net + GAN) trainer on CoNSeP.

Mahmood-inspired adversarial training: continuous-target discriminator
on (RGB + HV + Sem3-prob) 8-channel input.
Same architecture as train_lizard_phase2.py — only loader and paths change.
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, "/home/sbarua/fanseg/code")
sys.path.insert(0, "/home/sbarua/fanseg/code/consep")
from encoder import UNIEncoder
from decoder_dpt import DPTDecoder
from discriminator import FanSegDiscriminator
from consep_loader import CoNSePDataset
from losses import hv_loss, msge_loss, sem3_loss, hinge_d_loss, hinge_g_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, required=True)
    p.add_argument("--phase1_ckpt", default=None)
    p.add_argument("--out_dir", default="/home/sbarua/fanseg/checkpoints/consep_phase2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--decoder_ch", type=int, default=128)
    p.add_argument("--lambda_hv",   type=float, default=2.0)
    p.add_argument("--lambda_msge", type=float, default=1.0)
    p.add_argument("--lambda_sem3", type=float, default=1.0)
    p.add_argument("--lambda_adv",  type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--depths", type=str, default="5,11,17,23")
    return p.parse_args()


class FanSegCoNSeP(nn.Module):
    def __init__(self, depths=(5, 11, 17, 23), decoder_ch=128):
        super().__init__()
        self.encoder = UNIEncoder(freeze=True)
        self.decoder = DPTDecoder(in_dim=1024, ch=decoder_ch)
        self.depths = depths
    def forward(self, x):
        ms = self.encoder.forward_multiscale(x, depths=self.depths)
        return self.decoder(ms)


def build_d_input(image_b3hw, hv_b2hw, sem3_b3hw):
    return torch.cat([image_b3hw, hv_b2hw, sem3_b3hw], dim=1)


def sem_index_to_onehot(sem_bhw, num_classes=3):
    return F.one_hot(sem_bhw.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def main():
    args = parse_args()
    if args.phase1_ckpt is None:
        args.phase1_ckpt = f"/home/sbarua/fanseg/checkpoints/consep_phase1/split{args.split}/best.pt"
    out_dir = os.path.join(args.out_dir, f"split{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    depths = tuple(int(s) for s in args.depths.split(","))
    print(f"[init] split={args.split} epochs={args.epochs} bs={args.batch_size} "
          f"lambda_adv={args.lambda_adv} warmup={args.warmup_epochs}")
    print(f"[init] D input: 8-ch (RGB 3 + HV 2 + Sem3 prob 3) — Mahmood-inspired continuous target")

    train_ds = CoNSePDataset(split=args.split, mode="train", augment=True)
    val_ds   = CoNSePDataset(split=args.split, mode="test",  augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"[data] train_patches={len(train_ds)} val_patches={len(val_ds)}")

    G = FanSegCoNSeP(depths=depths, decoder_ch=args.decoder_ch).to(device)
    cp = torch.load(args.phase1_ckpt, map_location=device, weights_only=False)
    G.load_state_dict(cp["model"])
    print(f"[G] loaded Phase 1 ckpt epoch {cp.get('epoch', '?')}, val_loss {cp.get('val_loss', '?')}")

    D = FanSegDiscriminator(in_ch=8).to(device)
    n_d = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"[D] {n_d/1e6:.2f}M trainable params")

    optim_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, G.parameters()),
                                lr=args.lr_g, weight_decay=args.weight_decay)
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.0, 0.999))
    sched_G = CosineAnnealingLR(optim_G, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    sem3_weight = torch.tensor([1.0, 1.5, 2.0], device=device)
    history = []; best_val = float("inf"); t0 = time.time()

    for ep in range(args.epochs):
        cur_lambda_adv = args.lambda_adv * min(1.0, (ep + 1) / max(args.warmup_epochs, 1))
        G.train(); D.train()
        ep_l_sup = ep_l_d = ep_l_g_adv = 0.0; n_batches = 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            hv  = batch["hv"].to(device, non_blocking=True)
            sem = batch["sem"].to(device, non_blocking=True)
            fg  = batch["fg_mask"].to(device, non_blocking=True)

            hv_pred, sem3_logits = G(img)
            l_hv   = hv_loss(hv_pred, hv, fg)
            l_msge = msge_loss(hv_pred, hv, fg)
            l_sem3, _, _ = sem3_loss(sem3_logits, sem, weight=sem3_weight)
            L_sup = args.lambda_hv * l_hv + args.lambda_msge * l_msge + args.lambda_sem3 * l_sem3

            sem3_prob = torch.softmax(sem3_logits, dim=1)
            sem_onehot = sem_index_to_onehot(sem, num_classes=3)

            real_x = build_d_input(img, hv, sem_onehot)
            fake_x_det = build_d_input(img, hv_pred.detach(), sem3_prob.detach())
            real_logits = D(real_x)
            fake_logits_det = D(fake_x_det)
            L_D = hinge_d_loss(real_logits, fake_logits_det)
            optim_D.zero_grad(set_to_none=True); L_D.backward(); optim_D.step()

            L_G_total = L_sup
            l_g_adv_val = 0.0
            if cur_lambda_adv > 0:
                fake_x = build_d_input(img, hv_pred, sem3_prob)
                fake_logits = D(fake_x)
                L_G_adv = hinge_g_loss(fake_logits)
                L_G_total = L_sup + cur_lambda_adv * L_G_adv
                l_g_adv_val = float(L_G_adv.item())

            optim_G.zero_grad(set_to_none=True); L_G_total.backward(); optim_G.step(); sched_G.step()
            ep_l_sup += float(L_sup.item()); ep_l_d += float(L_D.item())
            ep_l_g_adv += l_g_adv_val; n_batches += 1

        G.eval()
        v_l_sup = 0.0; v_n = 0
        with torch.inference_mode():
            for batch in val_loader:
                img = batch["image"].to(device); hv = batch["hv"].to(device)
                sem = batch["sem"].to(device);   fg = batch["fg_mask"].to(device)
                hv_p, sem3_p = G(img)
                lh = hv_loss(hv_p, hv, fg); lm = msge_loss(hv_p, hv, fg)
                ls, _, _ = sem3_loss(sem3_p, sem, weight=sem3_weight)
                v_l_sup += (args.lambda_hv*lh + args.lambda_msge*lm + args.lambda_sem3*ls).item()
                v_n += 1
        val_loss = v_l_sup / max(v_n, 1)

        rec = {"epoch": ep + 1, "lambda_adv_used": cur_lambda_adv,
               "train_loss_sup": ep_l_sup / n_batches,
               "train_loss_d": ep_l_d / n_batches,
               "train_loss_g_adv": ep_l_g_adv / n_batches,
               "val_loss_sup": val_loss,
               "lr_g": sched_G.get_last_lr()[0],
               "t_min": (time.time() - t0) / 60}
        history.append(rec)
        print(f"[ep {ep+1:02d}/{args.epochs}] lambda_adv={cur_lambda_adv:.4f} "
              f"sup={rec['train_loss_sup']:.4f} L_D={rec['train_loss_d']:.4f} "
              f"L_G_adv={rec['train_loss_g_adv']:.4f} val={rec['val_loss_sup']:.4f} | "
              f"{rec['t_min']:.1f}min", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                        "epoch": ep + 1, "val_loss": best_val, "args": vars(args)},
                       os.path.join(out_dir, "best.pt"))

    torch.save({"G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                "epoch": args.epochs, "args": vars(args)},
               os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val_sup = {best_val:.4f}  total {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
