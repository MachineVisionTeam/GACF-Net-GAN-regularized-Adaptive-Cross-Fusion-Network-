"""GACF-Net — Context-Aware Hybrid Fusion on Fixed Features.

Architecture (per the v2 + cross-term spec):
  INPUT: hand(71) + tight(1024) + ctx(1024)  per-nucleus

  Step 1 — SHARED PROJECTIONS to d_proj=192 + tanh
      One per modality. Used by BOTH the main pipeline AND the cross-block.
      Shared = principled alignment (one coord system per modality), not just cheap.

  Step 2 — GMU GATING (lightweight, ~1.7K params)
      Per-nucleus softmax gate over 3 streams -> single 192-d gated vector.

  Step 3 — CELL-GRAPH TRANSFORMER (THE HEADLINE BLOCK)
      Per-patch kNN graph (k=64) over centroids.
      1 transformer layer, 4 heads, hidden=192.
      Sparse attention via scatter ops, supports variable patch sizes.

  Step 4 — CROSS-BLOCK (3 MFB tails over RAW pairs, parallel to graph branch)
      Tail per pair (h,t), (h,c), (t,c):
        Hadamard -> sum-pool over k_mfb=3 groups -> dropout -> signed-sqrt -> L2 norm
      Each tail outputs 64-d -> concat 3 = 192-d cross-block.

  Step 5 — CLASSIFIER HEAD
      MLP from [graph(192) || cross(192)] = 384 -> 128 -> num_classes.
      Used for end-to-end training; discarded for the XGBoost handoff.

Honest framing: graph transformer (Step 3) is the headline contribution.
Cross-block (Step 4) is a principled-novelty add-on; expected gain +0.5-2.0.

Param budget target: under 800K total.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Cell-Graph Transformer (sparse attention over kNN edges per patch)
# =============================================================================
class CellGraphLayer(nn.Module):
    """Single transformer layer with sparse attention over a kNN edge list.

    Pre-norm, multi-head attention restricted to edge_index, residual + FFN.
    Uses scatter ops (index_add_, index_reduce_) for variable-size patches.
    """
    def __init__(self, d_model=192, n_heads=4, dropout=0.3, ffn_mult=2):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """
        x: (N, d_model)
        edge_index: (2, E)  rows are (src, dst).
                    Convention: dst attends to src (i.e. dst's query × src's key).
        """
        N, d = x.shape
        if edge_index.numel() == 0:
            # No edges (single-nucleus patches). Just FFN with residual.
            return x + self.dropout(self.ffn(self.norm2(x)))

        src, dst = edge_index[0], edge_index[1]  # both (E,)

        # ----- Multi-head sparse attention -----
        h = self.norm1(x)
        q = self.q_proj(h).view(N, self.n_heads, self.d_head)  # (N, H, D)
        k = self.k_proj(h).view(N, self.n_heads, self.d_head)
        v = self.v_proj(h).view(N, self.n_heads, self.d_head)

        # Score per edge: dst's query dot src's key
        q_dst = q[dst]   # (E, H, D)
        k_src = k[src]   # (E, H, D)
        v_src = v[src]   # (E, H, D)
        scores = (q_dst * k_src).sum(dim=-1) * self.scale  # (E, H)

        # Per-dst, per-head softmax over incoming edges (sparse).
        # Stable softmax: subtract per-(dst, head) max, then exp, then normalize.
        scores_max = torch.full((N, self.n_heads), float('-inf'), device=x.device, dtype=scores.dtype)
        scores_max = scores_max.index_reduce_(0, dst, scores, 'amax', include_self=True)
        # Replace -inf (nodes with no incoming edges) with 0 to avoid NaN
        scores_max = torch.where(torch.isinf(scores_max), torch.zeros_like(scores_max), scores_max)
        scores_exp = torch.exp(scores - scores_max[dst])  # (E, H)

        scores_sum = torch.zeros(N, self.n_heads, device=x.device, dtype=scores.dtype)
        scores_sum.index_add_(0, dst, scores_exp)
        denom = scores_sum[dst].clamp(min=1e-12)  # (E, H)
        alpha = scores_exp / denom  # (E, H)

        # Aggregate: out[dst, h, d] = sum over edges with this dst of alpha[edge, h] * v_src[edge, h, d]
        weighted_v = alpha.unsqueeze(-1) * v_src  # (E, H, D)
        out = torch.zeros(N, self.n_heads, self.d_head, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, weighted_v)
        out = out.reshape(N, self.d_model)
        out = self.o_proj(out)

        # Residual + FFN
        x = x + self.dropout(out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CellGraphTransformer(nn.Module):
    """Multi-layer transformer over a per-patch kNN graph (k=64 neighbours)."""
    def __init__(self, d_model=192, n_heads=4, n_layers=1, k_graph=64, dropout=0.3):
        super().__init__()
        self.k_graph = k_graph
        self.layers = nn.ModuleList([
            CellGraphLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    @staticmethod
    def build_edges(patch_ids, centroids, k_graph):
        """Return edge_index (2, E) with src->dst pairs from per-patch kNN.

        - kNN within each patch, k = min(k_graph, n_nuclei - 1).
        - Self-edges excluded (we already have residuals).
        - Symmetric: include (i, j) and (j, i) so info flows both ways under sparse attention.
        """
        device = centroids.device
        edges = []
        unique_patches = torch.unique(patch_ids)
        for pid in unique_patches.tolist():
            mask = patch_ids == pid
            idx = torch.where(mask)[0]  # global indices in batch
            n = idx.size(0)
            if n < 2:
                continue
            k = min(k_graph, n - 1)
            cent = centroids[idx]  # (n, 2)
            d = torch.cdist(cent, cent)  # (n, n)
            d.fill_diagonal_(float('inf'))
            _, nb = d.topk(k, dim=-1, largest=False)  # (n, k) local indices
            src_local = torch.arange(n, device=device).repeat_interleave(k)
            dst_local = nb.flatten()
            src_glob = idx[src_local]
            dst_glob = idx[dst_local]
            # Symmetric: add both directions
            edges.append(torch.stack([src_glob, dst_glob], dim=0))
            edges.append(torch.stack([dst_glob, src_glob], dim=0))
        if not edges:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        return torch.cat(edges, dim=1)

    def forward(self, x, patch_ids, centroids):
        edge_index = self.build_edges(patch_ids, centroids, self.k_graph)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


# =============================================================================
# GACF-Net main module
# =============================================================================
class GACFNet(nn.Module):
    def __init__(self,
                 hand_dim=71, tight_dim=1024, ctx_dim=1024,
                 d_proj=192, n_heads=4, n_layers=1,
                 k_graph=64, k_mfb=3,
                 num_classes=4,
                 dropout=0.4,
                 modality_dropout=0.15):
        super().__init__()
        assert d_proj % k_mfb == 0, f"d_proj={d_proj} must divide by k_mfb={k_mfb}"
        self.d_proj = d_proj
        self.k_mfb  = k_mfb
        self.modality_dropout = modality_dropout
        self.num_classes = num_classes

        # ----- Step 1: SHARED projections (one per modality, used by both branches)
        self.proj_h = nn.Linear(hand_dim,  d_proj)
        self.proj_t = nn.Linear(tight_dim, d_proj)
        self.proj_c = nn.Linear(ctx_dim,   d_proj)
        self.ln_h = nn.LayerNorm(d_proj)
        self.ln_t = nn.LayerNorm(d_proj)
        self.ln_c = nn.LayerNorm(d_proj)
        self.proj_dropout = nn.Dropout(dropout * 0.75)  # lighter on early features

        # ----- Step 2: GMU gating (lightweight) -----
        self.gate_proj = nn.Linear(3 * d_proj, 3)

        # ----- Step 3: Cell-graph transformer (headline block) -----
        self.transformer = CellGraphTransformer(
            d_model=d_proj, n_heads=n_heads, n_layers=n_layers,
            k_graph=k_graph, dropout=dropout)

        # ----- Step 4: Cross-block (3 MFB tails). Per-pair tail params: only dropout, no Linear.
        self.cross_dropout = nn.Dropout(dropout)
        # 3 pairs × (d_proj / k_mfb) sum-pool dim each = 3 × (192/3=64) = 192
        self.cross_out_dim = (d_proj // k_mfb) * 3

        # ----- Step 5: Classifier head (end-to-end training; discarded for XGBoost handoff) -----
        head_in = d_proj + self.cross_out_dim   # 192 + 192 = 384
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _mfb_tail(a, b, k_mfb, dropout):
        """Compute one MFB tail: Hadamard -> sum-pool(k) -> dropout -> signed-sqrt -> L2 norm.

        a, b: (N, d_proj)
        Returns: (N, d_proj // k_mfb)
        """
        d = a.size(-1)
        out_dim = d // k_mfb
        had = a * b                                            # (N, d_proj)
        had = had.view(had.size(0), out_dim, k_mfb).sum(dim=-1)  # (N, out_dim)
        had = dropout(had)
        had = torch.sign(had) * torch.sqrt(torch.abs(had) + 1e-8)   # signed-sqrt
        had = F.normalize(had, p=2, dim=-1)
        return had

    def project(self, hand, tight, ctx):
        """Step 1: shared projections + tanh (no GMU yet)."""
        h = torch.tanh(self.ln_h(self.proj_h(hand)))   # (N, d_proj)
        t = torch.tanh(self.ln_t(self.proj_t(tight)))
        c = torch.tanh(self.ln_c(self.proj_c(ctx)))
        if self.training:
            h = self.proj_dropout(h); t = self.proj_dropout(t); c = self.proj_dropout(c)
        return h, t, c

    def forward(self, hand, tight, ctx, patch_ids, centroids, return_embeddings=False):
        """
        hand: (N, hand_dim)
        tight: (N, tight_dim)
        ctx: (N, ctx_dim)
        patch_ids: (N,) long
        centroids: (N, 2) float (x, y)

        Returns:
          logits: (N, num_classes)
          (if return_embeddings) graph_emb: (N, d_proj), cross_emb: (N, cross_out_dim)
        """
        # Modality dropout (training-only): zero out one full stream with prob = modality_dropout
        if self.training and self.modality_dropout > 0:
            if torch.rand((), device=hand.device).item() < self.modality_dropout:
                drop_idx = torch.randint(0, 3, (1,)).item()
                if drop_idx == 0:   hand  = torch.zeros_like(hand)
                elif drop_idx == 1: tight = torch.zeros_like(tight)
                else:               ctx   = torch.zeros_like(ctx)

        # Step 1: shared projections
        h, t, c = self.project(hand, tight, ctx)        # each (N, 192)

        # Step 2: GMU gating
        gate_in = torch.cat([h, t, c], dim=-1)          # (N, 576)
        gates   = F.softmax(self.gate_proj(gate_in), dim=-1)  # (N, 3)
        gated   = gates[:, 0:1] * h + gates[:, 1:2] * t + gates[:, 2:3] * c  # (N, 192)

        # Step 3: cell-graph transformer (main branch)
        graph_emb = self.transformer(gated, patch_ids, centroids)   # (N, 192)

        # Step 4: cross-block (parallel branch — uses h, t, c directly, NOT graph_emb)
        cross_ht = self._mfb_tail(h, t, self.k_mfb, self.cross_dropout)  # (N, 64)
        cross_hc = self._mfb_tail(h, c, self.k_mfb, self.cross_dropout)  # (N, 64)
        cross_tc = self._mfb_tail(t, c, self.k_mfb, self.cross_dropout)  # (N, 64)
        cross    = torch.cat([cross_ht, cross_hc, cross_tc], dim=-1)     # (N, 192)

        # Step 5: classifier head
        combined = torch.cat([graph_emb, cross], dim=-1)   # (N, 384)
        logits   = self.classifier(combined)                # (N, num_classes)

        if return_embeddings:
            return logits, graph_emb, cross
        return logits


# =============================================================================
# Param count helper
# =============================================================================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_name = {}
    for n, p in model.named_parameters():
        prefix = n.split('.')[0]
        by_name[prefix] = by_name.get(prefix, 0) + p.numel()
    return total, trainable, by_name


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    import sys

    print("=" * 72)
    print("GACF-Net — Smoke Tests")
    print("=" * 72)

    # ----- Smoke Test 1: instantiate, check param count -----
    print("\n[1] Instantiating model (CoNSeP-style, 4 classes)...")
    torch.manual_seed(0)
    model = GACFNet(
        hand_dim=71, tight_dim=1024, ctx_dim=1024,
        d_proj=192, n_heads=4, n_layers=1, k_graph=64, k_mfb=3,
        num_classes=4, dropout=0.4, modality_dropout=0.15,
    )
    total, trainable, by_name = count_params(model)
    print(f"  Total params:     {total:>10,}")
    print(f"  Trainable params: {trainable:>10,}")
    print(f"  Param budget:     {800_000:>10,}  (target <800K)")
    print(f"  Status: {'✓ UNDER BUDGET' if total < 800_000 else '✗ OVER BUDGET'}")
    print(f"\n  Top-level breakdown:")
    for n, p in sorted(by_name.items(), key=lambda x: -x[1]):
        print(f"    {n:20s} {p:>10,}")

    if total >= 800_000:
        print("\n⚠ Over budget — investigate before proceeding.")
        sys.exit(1)

    # ----- Smoke Test 2: forward pass on random input -----
    print("\n[2] Forward pass on random input (60 nuclei, 3 patches)...")
    N = 60
    n_patches = 3
    hand   = torch.randn(N, 71)
    tight  = torch.randn(N, 1024)
    ctx    = torch.randn(N, 1024)
    # Patches of size 20, 20, 20
    patch_ids = torch.tensor([0]*20 + [1]*20 + [2]*20, dtype=torch.long)
    centroids = torch.rand(N, 2) * 256

    model.eval()
    with torch.no_grad():
        logits, graph_emb, cross_emb = model(hand, tight, ctx, patch_ids, centroids, return_embeddings=True)
    print(f"  logits shape:    {logits.shape}     expected (60, 4)")
    print(f"  graph_emb shape: {graph_emb.shape}     expected (60, 192)")
    print(f"  cross_emb shape: {cross_emb.shape}     expected (60, 192)")
    print(f"  logits stats:    mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}  "
          f"min={logits.min().item():+.3f}  max={logits.max().item():+.3f}")
    print(f"  any NaN/inf:     logits={torch.isnan(logits).any().item() or torch.isinf(logits).any().item()}  "
          f"graph={torch.isnan(graph_emb).any().item()}  cross={torch.isnan(cross_emb).any().item()}")
    print(f"  Status: {'✓ FORWARD OK' if logits.shape == (N, 4) and not torch.isnan(logits).any() else '✗ FORWARD BROKEN'}")

    # ----- Smoke Test 3: backward pass + gradient sanity -----
    print("\n[3] Backward pass + gradient flow...")
    model.train()
    logits = model(hand, tight, ctx, patch_ids, centroids)
    target = torch.randint(0, 4, (N,))
    loss = F.cross_entropy(logits, target)
    loss.backward()
    print(f"  loss value:   {loss.item():.4f}")
    no_grad = []
    zero_grad = []
    for n, p in model.named_parameters():
        if p.grad is None:
            no_grad.append(n)
        elif p.grad.abs().sum().item() == 0:
            zero_grad.append(n)
    print(f"  params without grad: {len(no_grad)} (expected 0)")
    if no_grad:
        for n in no_grad: print(f"    no grad: {n}")
    print(f"  params with zero grad: {len(zero_grad)} (expected 0)")
    print(f"  Status: {'✓ BACKWARD OK' if not no_grad and not zero_grad else '✗ GRAD FLOW BROKEN'}")

    # ----- Smoke Test 4: Lizard-style with 6 classes, larger N -----
    print("\n[4] Lizard-style instantiation (6 classes)...")
    model_lizard = GACFNet(num_classes=6)
    total_l, _, _ = count_params(model_lizard)
    print(f"  Total params: {total_l:>10,}  (6 classes)")
    print(f"  Status: {'✓ UNDER BUDGET' if total_l < 800_000 else '✗ OVER BUDGET'}")

    # ----- Smoke Test 5: realistic Lizard patch density (-100 nuclei / patch) -----
    print("\n[5] Realistic patch density (5 patches × ~100 nuclei = 500 total)...")
    rng = torch.Generator().manual_seed(42)
    per_patch = torch.randint(50, 150, (5,), generator=rng).tolist()
    N = sum(per_patch)
    print(f"  Patches: {per_patch}  total N={N}")
    hand   = torch.randn(N, 71)
    tight  = torch.randn(N, 1024)
    ctx    = torch.randn(N, 1024)
    patch_ids = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(per_patch)])
    centroids = torch.rand(N, 2) * 256

    model_lizard.eval()
    import time
    t0 = time.time()
    with torch.no_grad():
        logits = model_lizard(hand, tight, ctx, patch_ids, centroids)
    dt = time.time() - t0
    print(f"  Forward pass:  {dt*1000:.1f} ms for {N} nuclei across 5 patches")
    print(f"  logits shape:  {logits.shape}  expected ({N}, 6)")
    print(f"  Status: {'✓ SCALING OK' if logits.shape == (N, 6) and not torch.isnan(logits).any() else '✗ SCALING ISSUE'}")

    print("\n" + "=" * 72)
    print("All smoke tests done.")
    print("=" * 72)
