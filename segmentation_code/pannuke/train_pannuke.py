"""Phase 1 (BHF-Net base) trainer on PanNuke. Mirror of train_consep.py."""
import os, sys, time, argparse, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, "/home/sbarua/fanseg/code")
sys.path.insert(0, "/home/sbarua/fanseg/code/pannuke")
from encoder import UNIEncoder
from decoder_dpt import DPTDecoder
from pannuke_loader import PanNukeDataset
from losses import hv_loss, msge_loss, sem3_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=int, required=True, help="PanNuke 3-fold split (1..3)")
    p.add_argument("--out_dir", default="/home/sbarua/fanseg/checkpoints/pannuke_phase1")
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
    p.add_argument("--depths", type=str, default="5,11,17,23")
    return p.parse_args()


class FanSegPanNuke(nn.Module):
    def __init__(self, depths=(5, 11, 17, 23), decoder_ch=128):
        super().__init__()
        self.encoder = UNIEncoder(freeze=True)
        self.decoder = DPTDecoder(in_dim=1024, ch=decoder_ch)
        self.depths = depths
    def forward(self, x):
        ms = self.encoder.forward_multiscale(x, depths=self.depths)
        return self.decoder(ms)


def main():
    args = parse_args()
    out_dir = os.path.join(args.out_dir, f"split{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    depths = tuple(int(s) for s in args.depths.split(","))
    print(f"[init] device={device} split={args.split} epochs={args.epochs} bs={args.batch_size}")

    train_ds = PanNukeDataset(split=args.split, mode="train", augment=True)
    val_ds   = PanNukeDataset(split=args.split, mode="test",  augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"[data] train_patches={len(train_ds)} val_patches={len(val_ds)}")

    model = FanSegPanNuke(depths=depths, decoder_ch=args.decoder_ch).to(device)
    print(f"[model] trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    sem3_weight = torch.tensor([1.0, 1.5, 2.0], device=device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader), eta_min=1e-6)

    history = []; best_val = float("inf"); t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ep_l = ep_hv = ep_ms = ep_sem = 0.0; n_b = 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            hv  = batch["hv"].to(device, non_blocking=True)
            sem = batch["sem"].to(device, non_blocking=True)
            fg  = batch["fg_mask"].to(device, non_blocking=True)
            hv_pred, sem3_pred = model(img)
            l_hv = hv_loss(hv_pred, hv, fg); l_msge = msge_loss(hv_pred, hv, fg)
            l_sem3, _, _ = sem3_loss(sem3_pred, sem, weight=sem3_weight)
            loss = args.lambda_hv*l_hv + args.lambda_msge*l_msge + args.lambda_sem3*l_sem3
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step(); sched.step()
            ep_l += loss.item(); ep_hv += l_hv.item(); ep_ms += l_msge.item(); ep_sem += l_sem3.item()
            n_b += 1

        model.eval()
        v_l = v_n = 0
        with torch.inference_mode():
            for batch in val_loader:
                img = batch["image"].to(device); hv = batch["hv"].to(device)
                sem = batch["sem"].to(device);   fg = batch["fg_mask"].to(device)
                hv_pred, sem3_pred = model(img)
                l_hv = hv_loss(hv_pred, hv, fg); l_msge = msge_loss(hv_pred, hv, fg)
                l_sem3, _, _ = sem3_loss(sem3_pred, sem, weight=sem3_weight)
                v_l += (args.lambda_hv*l_hv + args.lambda_msge*l_msge + args.lambda_sem3*l_sem3).item()
                v_n += 1
        rec = {"epoch": ep+1, "train_loss": ep_l/n_b, "val_loss": v_l/max(v_n,1),
               "lr": sched.get_last_lr()[0], "t_min": (time.time()-t0)/60}
        history.append(rec)
        print(f"[ep {ep+1:02d}/{args.epochs}] train l={rec['train_loss']:.4f} | val l={rec['val_loss']:.4f} | {rec['t_min']:.1f}min", flush=True)
        if rec["val_loss"] < best_val:
            best_val = rec["val_loss"]
            torch.save({"model": model.state_dict(), "epoch": ep+1, "val_loss": best_val,
                        "args": vars(args), "depths": depths, "decoder_ch": args.decoder_ch},
                       os.path.join(out_dir, "best.pt"))

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args),
                "depths": depths, "decoder_ch": args.decoder_ch},
               os.path.join(out_dir, "last.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] best val_loss = {best_val:.4f}  total {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
