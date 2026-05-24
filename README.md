# GACF-Net: GAN-regularized Adaptive Cross-Fusion Network

Two-stage framework for joint nuclei segmentation and classification in H&E-stained histopathology images. Stage 1 combines a frozen UNI ViT-L foundation-model encoder with a DPT decoder and three task heads (HV, Sem3, Boundary), regularized by a Mahmood-style PatchGAN on continuous outputs. Stage 2 fuses per-nucleus handcrafted morphology with raw UNI features via an adaptive multimodal gate, a cell-graph transformer with k=64 neighbours, and a low-rank multiplicative cross-block over raw modality pairs.

> **Status:** active development; private repository. Manuscript in preparation.

---

## Repository layout

```
GACF-Net/
├── README.md                          (this file)
├── code/                              Stage 2 — fusion classifier
│   ├── gacfnet.py                     architecture (~757K params, under 800K budget)
│   ├── train_gacfnet.py               per-fold trainer (CB-focal + balanced softmax + modality dropout)
│   └── xgb_gacfnet.py                 XGBoost handoff with 6-mode ablation
│                                       (hand / deep / raw / +graph / +cross / +random)
│
├── segmentation_code/                 Stage 1 — segmentation pipeline
│   ├── shared/                        UNI encoder, DPT decoder, PatchGAN, losses, augmentations
│   ├── lizard/                        per-dataset loaders, train + Phase 2 GAN, inference (TTA), eval
│   ├── consep/
│   └── pannuke/
│
├── reports/                           paper-writing reference docs
│   ├── GACF_NET_FINAL_RESULTS.txt    complete 3-way (Paper / Replication / Ours) comparison
│   │                                  tables across 6 ablation modes × 3 datasets
│   ├── GACF_NET_ARCHITECTURE_SPEC.txt full architecture spec with equations + citations
│   └── GACF_NET_METRICS_SPEC.txt     segmentation + Convention C classification metric definitions
│
└── xgb_results/                       per-dataset, per-mode Convention C JSONs
                                       (consep|lizard|pannuke) × (hand|deep|raw|+graph|+cross|+random)
```

---

## Quick start

### Stage 1 — Segmentation training and inference

```bash
cd segmentation_code/lizard

# Phase 1 training (no GAN)
python train_lizard_v2.py            --fold 1 --epochs 100

# Phase 2 training (GAN regularization on continuous HV+Sem3 outputs)
python train_lizard_phase2.py        --fold 1 --epochs 50  --gan_weight 0.1

# Phase 2 + D4 TTA inference -> predicted instance masks
python inference_phase2_tta.py       --fold 1

# Evaluate full segmentation metrics (ACC, SE, SP, Dice, AJI, PQ, DQ, SQ)
python eval_phase2_full_metrics.py
```

Same pattern for `segmentation_code/{consep,pannuke}/`.

### Stage 2 — Fusion classifier training and evaluation

```bash
cd code

# Train the fusion model per fold (extracts graph + cross embeddings)
python train_gacfnet.py --dataset consep   --gpu 0
python train_gacfnet.py --dataset pannuke  --gpu 0
python train_gacfnet.py --dataset lizard   --gpu 0

# Hand off to XGBoost across all 6 ablation modes
python xgb_gacfnet.py   --dataset consep   --gpu 0
python xgb_gacfnet.py   --dataset pannuke  --gpu 0
python xgb_gacfnet.py   --dataset lizard   --gpu 0
```

### Final classification evaluation (Convention C, HoVer-Net protocol)

Convention C per-class F1 with HoVer-Net weights `(α₀,α₁,α₂,α₃) = (2,2,1,1)`:

```
F_c = 2·TP_c / (2·TP_c + 2·FP_c + 2·FN_c + FP_d_c + FN_d_c)
```

See `reports/GACF_NET_METRICS_SPEC.txt` for full metric definitions and pseudocode.

---

## Headline results (Convention C macro F1)

| Dataset | AGAFNet Paper | AGAFNet Repl | **hand** | **deep** | **raw** (hand+deep) | **+graph** | **+cross** (headline) | **+random** (null) |
|---|---|---|---|---|---|---|---|---|
| **Lizard** (6-class) | 59.90 | 60.92 | 34.92 | 45.04 | 51.45 | 51.69 | **51.85** | 51.71 |
| **CoNSeP** (4-class) | 59.24 | 41.35 | 32.78 | **51.63** | 50.97 | 49.89 | 49.89 | 49.84 |
| **PanNuke** (5-class) | 61.51 | 49.21 | 36.02 | 52.84 | **53.09** | 52.10 | 52.18 | 52.06 |

Three findings the ablation supports:

1. **UNI foundation features dominate**: hand → deep gives **+10 to +19 F1** across all 3 datasets.
2. **Hand features matter only on the most class-fine dataset**: hand → raw adds **+6.41 on Lizard** (6 classes incl. rare granulocytes), **+0** on CoNSeP and PanNuke.
3. **The fusion architecture adds little**: graph + cross-block contribute at most **+0.40 F1** and on 2 of 3 datasets slightly hurt. The null-control ablation (random Gaussian projection of equivalent dimension) shows the cross-block beats random by **≤0.14 F1** on every dataset — its bilinear structure is statistically indistinguishable from a same-sized random projection.

See `reports/GACF_NET_FINAL_RESULTS.txt` for the complete per-class breakdown including segmentation metrics.

### Segmentation results (GACF-Net Stage 1 vs AGAFNet replication)

Selected wins over AGAFNet replication across all 3 datasets:

| | Sensitivity (SE) | AJI | PQ |
|---|---|---|---|
| Lizard | **+8.17** | **+18.70** | **+14.19** |
| CoNSeP | **+5.23** | **+5.07** | **+7.59** |
| PanNuke | **+8.17** | **+5.14** | **+6.99** |

vs AGAFNet **Paper**: SE wins on all 3 datasets (+0.7 / +0.7 / +3.4), CoNSeP AJI +12.69, PanNuke PQ +13.47.

---

## Key components

| Stage | Component | Description |
|---|---|---|
| 1 | UNI ViT-L encoder | Frozen MahmoodLab pathology foundation model (1024-d tokens, 16×16 patches) |
| 1 | DPT decoder | Multi-scale dense prediction transformer fusing ViT-L block features |
| 1 | HV head | 2-channel horizontal/vertical gradient for instance separation (HoVer-Net protocol) |
| 1 | Sem3 head | 3-channel softmax: {background, nucleus, boundary} |
| 1 | Boundary head | 1-channel auxiliary boundary supervision |
| 1 | PatchGAN discriminator | Adversarial regularization on continuous HV+Sem3 outputs (Phase 2) |
| 1 | TTA | D4 group test-time augmentation (8 orientations) |
| — | Per-nucleus features | hand (71-d morphology) + UNI tight (1024-d centroid token) + UNI ctx (1024-d bbox-mean tokens) |
| 2 | Shared projections | Per-modality learned linear projections to d_proj=192 + tanh |
| 2 | GMU gating | Gated Multimodal Unit for per-nucleus modality weighting |
| 2 | Cell-graph transformer | Sparse attention over per-patch k=64-NN graph (the headline block) |
| 2 | MFB cross-block | Low-rank multiplicative cross-pairs over RAW modalities (h⊗t, h⊗c, t⊗c) |
| 2 | XGBoost classifier | Final per-nucleus class prediction with cross-term-protecting hyperparameters |

Full architecture spec in `reports/GACF_NET_ARCHITECTURE_SPEC.txt`.

---

## Citations (used in the architecture)

- **UNI** — Chen et al., *Nature Medicine* 2024 — pathology foundation model
- **DPT** — Ranftl et al., *ICCV* 2021 — dense prediction transformer decoder
- **PatchGAN** — Isola et al., *CVPR* 2017 — pix2pix-style adversarial regularization
- **HoVer-Net** — Graham et al., *Medical Image Analysis* 2019 — HV decoding + Convention C F1
- **MFB** — Yu et al., *ICCV* 2017 — multi-modal factorized bilinear pooling
- **GMU** — Arevalo et al., 2017 — Gated Multimodal Units
- **CB-focal loss** — Cui et al., *CVPR* 2019 — class-balanced focal loss
- **Balanced softmax** — Ren et al., 2020 — long-tail learning
- **AGAFNet** — Naing et al., *IEEE TIP* 2026 — comparison baseline

---

## Datasets

| Dataset | Classes | Folds | # Nuclei | Reference |
|---|---|---|---|---|
| CoNSeP | 4 (Misc, Inflam, Epi, Spindle) | 5 | 24,392 | Graham et al., MIA 2019 |
| Lizard | 6 (Neu, Epi, Lym, Pla, Eos, Con) | 5 | 568,653 | Graham et al., ICCVW 2021 |
| PanNuke | 5 (Neo, Inflam, Con, Dead, Epi) | 3 | 176,258 | Gamper et al., ECDP 2019 |

---

## License

To be determined. For now, this repository is for internal MachineVisionTeam research use only.

---

## Authors

Shemonti Barua — Kennesaw State University

This project is part of an ongoing PhD research program at KSU on histopathology image analysis. Manuscript in preparation.
