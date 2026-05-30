# Lizard — AGAFNet masks + GACF-Net fusion classifier

Controlled experiment: take AGAFNet's predicted instance masks (segmentation
held fixed), apply our GACF-Net fusion classifier (+ XGBoost head), and
compare classification head-to-head against AGAFNet's own pixel-majority-vote
classifier on the *same* nuclei.

## Run order

```
step1_match_and_label.py            # AGAFNet inst → GT match, build label table
step2_extract_handcrafted_parallel.py   # 71-d handcrafted feats (32 CPU workers)
step3_extract_uni.py                # UNI ViT-L tight (1024) + ctx (1024)
step4_build_features.py             # per-fold scaling → fusion_features_fold{N}.h5
step5_train_fusion.py               # train GACFNet → graph_emb + cross_emb
step6_xgb_and_eval.py               # XGBoost + Convention C head-to-head
```

`gacfnet.py` is the GACFNet model definition (copied from the main repo, kept
local so this folder is self-contained).

## Inputs read (hardcoded absolute paths)

- AGAFNet predicted masks: `/home/sbarua/Lizard/agafnet_replication/fusion/features/instance_maps/fold{N}_inst_maps.npz`
- GT masks: `/mnt/storage1/Lizard/agafnet_replication/data_conic/fold{N}/masks.npy`
- Reinhard-normalized images (handcrafted): `/home/sbarua/fanseg/predictions/lizard/fusion_features_phase2_reinhard/images_reinhard.npy`
- Raw images (UNI): `/mnt/storage1/Lizard/conic_patches/data/images.npy`
- Fold assignments / metadata: `/mnt/storage1/Lizard/agafnet_replication/data_conic/metadata.npy`

## Outputs written (not checked into this folder)

All intermediate features + final results currently live under
`/home/sbarua/fanseg/agafnet_mask_fusion/{features,preds,reports}/`.
Final head-to-head numbers will be added to this folder once the
CoNSeP and PanNuke runs are also done.

