"""
model.py — GNN + Macro-Micro Bidirectional Attention for MILP
=============================================================

Architecture
------------
1. Feature split & embedding
   var_feats [N,21] -> static [N,9]  -> BipartiteGCN -> static_out  [N,d]
                    -> dynamic [N,12] -> BipartiteGCN -> dynamic_out [N,d]

2. Macro-Micro bidirectional attention  (Task 3)
   static_out  -> W_k -> K_micro          Q_macro [M,d] (learnable, orthogonal)
   dynamic_out -> W_v -> V_micro
   S = Q @ K^T * log(N) / sqrt(d)       [M, N]  (size-adaptive temperature)
   A_fwd = softmax(S, dim=N)              -> H_macro = A_fwd @ V   [M, d]
   H_post = SelfAttn + FFN (pre-norm)     [M, d]
   A_bwd = softmax(S^T, dim=M)            -> H_feedback = A_bwd @ W_v_back(H_post)

3. Micro-Macro gated fusion  (Task 4)
   S_i = sum_m  w_m * A_fwd[m,i] * N    w_m = ||H_macro_m||  (N-rescaled)
   c_i = sigmoid(gamma * S_i + beta)
   V_updated = c * H_feedback + (1-c) * dynamic_out

4. Static-Dynamic gated fusion  (Task 4)
   alpha = sigmoid(W_gate · [static_out || V_updated])
   Feature_final = alpha * V_updated + (1-alpha) * static_out

5. Type-aware prediction head  (Task 4)
   Binary     -> P(x=1)  via sigmoid
   Integer    -> discretised logistic  (mu, sigma)
   Continuous -> direct value
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Feature column indices (must match build_dataset.py) ──────────────────
STATIC_VAR_IDX = [0, 1, 2, 3, 4, 5, 6, 19, 20]   # 9 cols
DYNAMIC_VAR_IDX = [7, 8, 9, 10, 11, 12, 13, 14,
                   15, 16, 17, 18]                  # 12 cols
N_STATIC = len(STATIC_VAR_IDX)    # 9
N_DYNAMIC = len(DYNAMIC_VAR_IDX)  # 12
N_CON = 6

# Variable type column indices within the 21-dim var_feats
_COL_BIN = 1    # is_type_binary
_COL_INT = 2    # is_type_integer
_COL_IMP = 3    # is_type_implicit_integer
_COL_CON = 4    # is_type_continuous


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def _scatter_mean(src: torch.Tensor, idx: torch.Tensor, dim_size: int):
    """Scatter-mean along dim-0.  Pure PyTorch, no extra packages."""
    out = src.new_zeros(dim_size, src.size(-1))
    cnt = idx.new_zeros(dim_size, dtype=src.dtype)
    expand_idx = idx.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, expand_idx, src)
    cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)


# ══════════════════════════════════════════════════════════════════════════
#  0.  Feature Normalizer  (per-instance preprocessing)
# ══════════════════════════════════════════════════════════════════════════

class FeatureNormalizer(nn.Module):
    """
    Per-instance feature normalisation applied before embedding.

    Strategy per column type:
      binary / indicator  (0 or 1)        -> leave unchanged
      degree              (integer, >0)   -> log(1 + x)  then standardise
      continuous / large-range            -> per-instance standardise
      edge coefficients                   -> per-instance standardise

    Standardise  = (x - mean) / (std + eps),  then clip to [-clip, +clip].
    All statistics are computed *per single instance* (not across the batch)
    so the model stays scale-invariant across different problem types.
    """

    # Variable features (21 cols) ──────────────────────────────────────────
    #   binary cols (no-op): 1,2,3,4,5,6, 10,11,13, 15,16,17,18
    #   log+std cols:        19 (degree)
    #   std-only cols:       0 (obj), 7 (red_cost), 8 (sol_val), 9 (sol_frac),
    #                        12 (age), 14 (avg_incumb), 20 (raw_obj)
    _VAR_LOG  = [19]
    _VAR_STD  = [0, 7, 8, 9, 12, 14, 19, 20]

    # Constraint features (6 cols) ─────────────────────────────────────────
    #   binary cols (no-op): 2 (is_tight)
    #   log+std cols:        5 (degree)
    #   std-only cols:       0 (bias), 1 (obj_cos), 3 (dual_val), 4 (age)
    _CON_LOG  = [5]
    _CON_STD  = [0, 1, 3, 4, 5]

    def __init__(self, clip: float = 5.0):
        super().__init__()
        self.clip = clip

    @torch.no_grad()
    def normalize_var(self, x: torch.Tensor) -> torch.Tensor:
        """[N, 21] -> [N, 21]  normalised (new tensor, no in-place)."""
        out = x.clone()
        for c in self._VAR_LOG:
            out[:, c] = torch.log1p(out[:, c].abs()) * out[:, c].sign()
        cols = self._VAR_STD
        v = out[:, cols]
        out[:, cols] = ((v - v.mean(0)) / (v.std(0) + 1e-6)).clamp(
            -self.clip, self.clip)
        return out

    @torch.no_grad()
    def normalize_con(self, x: torch.Tensor) -> torch.Tensor:
        """[C, 6] -> [C, 6]  normalised."""
        out = x.clone()
        for c in self._CON_LOG:
            out[:, c] = torch.log1p(out[:, c].abs()) * out[:, c].sign()
        cols = self._CON_STD
        v = out[:, cols]
        out[:, cols] = ((v - v.mean(0)) / (v.std(0) + 1e-6)).clamp(
            -self.clip, self.clip)
        return out

    @torch.no_grad()
    def normalize_edge(self, e: torch.Tensor) -> torch.Tensor:
        """[E] -> [E]  standardised edge coefficients."""
        return ((e - e.mean()) / (e.std() + 1e-6)).clamp(
            -self.clip, self.clip)


# ══════════════════════════════════════════════════════════════════════════
#  1.  Bipartite GCN  (edge-gated message passing)
# ══════════════════════════════════════════════════════════════════════════

class BipartiteGCNLayer(nn.Module):
    """One round of var <-> con message passing with sigmoid edge gating."""

    def __init__(self, h: int, dropout: float = 0.1):
        super().__init__()
        self.v2c_lin = nn.Linear(h, h)
        self.v2c_gate = nn.Sequential(nn.Linear(1, h), nn.Sigmoid())
        self.v2c_upd = nn.Sequential(
            nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout))
        self.c2v_lin = nn.Linear(h, h)
        self.c2v_gate = nn.Sequential(nn.Linear(1, h), nn.Sigmoid())
        self.c2v_upd = nn.Sequential(
            nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, vh, ch, edge_index, ew):
        ci, vi = edge_index[0], edge_index[1]
        e = ew.unsqueeze(-1)
        # var -> con
        msg = self.v2c_lin(vh)[vi] * self.v2c_gate(e)
        agg = _scatter_mean(msg, ci, ch.size(0))
        ch_new = self.v2c_upd(torch.cat([agg, ch], -1))
        # con -> var
        msg = self.c2v_lin(ch_new)[ci] * self.c2v_gate(e)
        agg = _scatter_mean(msg, vi, vh.size(0))
        vh_new = self.c2v_upd(torch.cat([agg, vh], -1))
        return vh_new, ch_new


class BipartiteGCN(nn.Module):
    """2-layer MLP projection  +  N GCN layers with residual connections."""

    def __init__(self, var_in: int, con_in: int, h: int,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # 2-layer MLP: captures non-linear feature interactions before GCN
        self.var_emb = nn.Sequential(
            nn.Linear(var_in, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h),      nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout))
        self.con_emb = nn.Sequential(
            nn.Linear(con_in, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h),      nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout))
        self.layers = nn.ModuleList(
            [BipartiteGCNLayer(h, dropout) for _ in range(n_layers)])

    def forward(self, var_feats, con_feats, edge_index, edge_val):
        vh = self.var_emb(var_feats)
        ch = self.con_emb(con_feats)
        for layer in self.layers:
            dv, dc = layer(vh, ch, edge_index, edge_val)
            vh = vh + dv
            ch = ch + dc
        return vh, ch


# ══════════════════════════════════════════════════════════════════════════
#  2.  Macro-Micro Bidirectional Attention
# ══════════════════════════════════════════════════════════════════════════

class MacroMicroAttention(nn.Module):
    """
    Forward  (micro->macro):  M probes aggregate over N variable nodes
    Self-attn:                multi-head self-attention on macro vectors
    Backward (macro->micro):  redistribute refined macro info to micro

    Returns
    -------
    H_feedback : [N, d]
    A_forward  : [M, N]   (needed by MicroMacroFusion)
    H_macro    : [M, d]   (needed by MicroMacroFusion for norm weights)
    """

    def __init__(self, d: int, n_probes: int = 16,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.scale = math.sqrt(d)

        self.Q_macro = nn.Parameter(torch.empty(n_probes, d))
        nn.init.orthogonal_(self.Q_macro)

        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)

        # macro self-attention  (pre-norm transformer block)
        self.sa_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True)

        self.ff_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d, d), nn.Dropout(dropout))

        self.W_v_back = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, K_in: torch.Tensor, V_in: torch.Tensor,
                return_aux: bool = False):
        K = self.W_k(K_in)                            # [N, d]
        V = self.W_v(V_in)                            # [N, d]
        N = K.size(0)

        # ---- forward: micro -> macro ----
        # Size-adaptive temperature: log(N) sharpens attention for larger
        # graphs, preventing the softmax from collapsing to uniform.
        log_n = torch.log(torch.tensor(N, dtype=K.dtype, device=K.device))
        log_n = log_n.clamp(min=1.0)
        S = self.Q_macro @ K.t() * log_n / self.scale  # [M, N]
        A_fwd = F.softmax(S, dim=-1)                   # [M, N]
        H_macro = self.drop(A_fwd) @ V                 # [M, d]

        # ---- macro self-attention  (pre-norm) ----
        h = H_macro.unsqueeze(0)                       # [1, M, d]
        h_norm = self.sa_norm(h)
        h_attn, _ = self.self_attn(h_norm, h_norm, h_norm)
        h = h + h_attn
        h = h + self.ffn(self.ff_norm(h))
        H_post = h.squeeze(0)                          # [M, d]

        # ---- backward: macro -> micro ----
        A_bwd_clean = F.softmax(S.t(), dim=-1)         # [N, M]  (no dropout)
        H_feedback = self.drop(A_bwd_clean) @ self.W_v_back(H_post)  # [N, d]

        if return_aux:
            return H_feedback, A_fwd, H_macro, {
                "A_backward": A_bwd_clean,
                "V_micro": V,
            }
        return H_feedback, A_fwd, H_macro


# ══════════════════════════════════════════════════════════════════════════
#  3.  Feature Fusion  (Task 4)
# ══════════════════════════════════════════════════════════════════════════

class MicroMacroFusion(nn.Module):
    """
    Gated fusion of macro feedback and micro dynamic features.

    S_i = sum_m  w_m * A_forward[m, i]     w_m = ||H_macro_m||_2 (normalised)
    c_i = sigmoid(gamma * S_i + beta)      gamma, beta learnable scalars
    V_updated_i = c_i * H_feedback_i  +  (1 - c_i) * Micro_Feature_i
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, H_feedback, micro_feat, A_forward, H_macro):
        """
        H_feedback : [N, d]
        micro_feat : [N, d]   (dynamic_out from GCN)
        A_forward  : [M, N]
        H_macro    : [M, d]
        """
        # importance weight per probe: normalised L2 norm
        w = H_macro.norm(dim=-1)                       # [M]
        w = w / (w.sum() + 1e-8)                       # [M]  normalised

        # per-variable attention importance
        S_i = (w.unsqueeze(-1) * A_forward).sum(dim=0)  # [N]
        # Rescale by N: mean(S_i)=1/N without this, making the gate
        # size-dependent.  After rescaling mean(S_i)≈1 for any N.
        S_i = S_i * S_i.size(0)

        # gated blending
        c = torch.sigmoid(self.gamma * S_i + self.beta) # [N]
        c = c.unsqueeze(-1)                             # [N, 1]
        V_updated = c * H_feedback + (1.0 - c) * micro_feat
        return V_updated


class StaticDynamicFusion(nn.Module):
    """
    Gated fusion of V_updated (dynamic+macro) and static features.

    alpha_i = sigmoid(W_gate · [static_i || V_updated_i])    element-wise [d]
    Feature_final_i = alpha_i * V_updated_i + (1 - alpha_i) * static_i
    """

    def __init__(self, d: int):
        super().__init__()
        self.W_gate = nn.Linear(2 * d, d)

    def forward(self, V_updated, static_feat):
        alpha = torch.sigmoid(
            self.W_gate(torch.cat([static_feat, V_updated], dim=-1)))  # [N, d]
        return alpha * V_updated + (1.0 - alpha) * static_feat         # [N, d]


# ══════════════════════════════════════════════════════════════════════════
#  4.  Type-aware Prediction Head  (Task 4)
# ══════════════════════════════════════════════════════════════════════════

class PredictionHead(nn.Module):
    """
    Per-variable-type prediction from Feature_final.

    Binary      -> sigmoid logit  -> P(x = 1)
    Integer     -> (mu, log_sigma) -> discretised logistic distribution
    Continuous  -> direct scalar value

    Includes an **LP skip connection** for binary variables: the raw
    LP-relaxation solution value is added as a bias to the binary logit,
    so the model effectively learns  "LP solution + correction".
    """

    def __init__(self, d: int, dropout: float = 0.1, lp_gate_bias: float = 2.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(d, 1)       # logit (correction over LP)
        self.integer_head = nn.Linear(d, 2)      # mu, log_sigma
        self.continuous_head = nn.Linear(d, 1)   # value

        # Learnable gate for LP skip (how much to trust LP vs learned)
        # sigmoid(bias) controls initial trust: 2.0 → ~88%, 0.0 → 50%
        self.lp_gate = nn.Linear(d, 1)
        nn.init.constant_(self.lp_gate.bias, lp_gate_bias)

    # ── public helpers ───────────────────────────────────────────────────

    @staticmethod
    def integer_log_prob(mu, log_sigma, targets):
        """
        Negative log-likelihood under a discretised logistic for integer vars.
        P(X = k) = sigma((k+0.5 - mu)/s) - sigma((k-0.5 - mu)/s)
        """
        s = log_sigma.exp().clamp(min=1e-4)
        upper = torch.sigmoid((targets + 0.5 - mu) / s)
        lower = torch.sigmoid((targets - 0.5 - mu) / s)
        prob = (upper - lower).clamp(min=1e-8)
        return prob.log()

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, feature_final, var_types, lp_sol=None):
        """
        feature_final : [N, d]
        var_types     : [N, 4]  (is_binary, is_int, is_implicit_int, is_continuous)
        lp_sol        : [N] or None — LP relaxation solution values (col 8)

        Returns dict with pred, binary_logits, integer_mu, integer_log_std,
        continuous_val keys.
        """
        h = self.shared(feature_final)                 # [N, d]

        bin_logits = self.binary_head(h).squeeze(-1)   # [N]

        # ── LP skip connection for binary variables ──
        # Convert LP solution [0,1] → logit space, gate with learned weight
        if lp_sol is not None:
            lp_logit = torch.logit(lp_sol.clamp(0.01, 0.99))    # [N]
            gate = torch.sigmoid(self.lp_gate(h).squeeze(-1))   # [N]
            bin_logits = bin_logits + gate * lp_logit             # learned correction

        int_params = self.integer_head(h)              # [N, 2]
        int_mu = int_params[:, 0]
        int_log_std = int_params[:, 1].clamp(-5.0, 5.0)

        cont_val = self.continuous_head(h).squeeze(-1)  # [N]

        # ── compose a single prediction vector (no in-place ops) ──
        is_bin = var_types[:, 0].bool()
        is_int = (var_types[:, 1].bool()) | (var_types[:, 2].bool())

        bin_pred = torch.sigmoid(bin_logits)
        pred = torch.where(is_bin, bin_pred,
               torch.where(is_int, int_mu, cont_val))

        return {
            "pred": pred,
            "binary_logits": bin_logits,
            "integer_mu": int_mu,
            "integer_log_std": int_log_std,
            "continuous_val": cont_val,
        }


# ══════════════════════════════════════════════════════════════════════════
#  5.  Full Model
# ══════════════════════════════════════════════════════════════════════════

class MILPGNNModel(nn.Module):
    """
    Complete model: dual-stream BipartiteGCN
                  + MacroMicro attention
                  + gated fusion (micro-macro + static-dynamic)
                  + type-aware prediction head.

    Parameters
    ----------
    hidden_dim   : int   embedding / hidden dimension          (default 64)
    n_gcn_layers : int   GCN rounds per stream                 (default 2)
    n_probes     : int   number of macro probe vectors M       (default 16)
    n_heads      : int   self-attention heads in macro layer    (default 4)
    dropout      : float dropout rate                           (default 0.1)
    """

    def __init__(self, hidden_dim: int = 64, n_gcn_layers: int = 2,
                 n_probes: int = 16, n_heads: int = 4,
                 dropout: float = 0.1, lp_gate_bias: float = 2.0):
        super().__init__()
        # ---- feature normalizer ----
        self.normalizer = FeatureNormalizer()

        # ---- dual-stream GCN ----
        self.static_gcn = BipartiteGCN(
            N_STATIC, N_CON, hidden_dim, n_gcn_layers, dropout)
        self.dynamic_gcn = BipartiteGCN(
            N_DYNAMIC, N_CON, hidden_dim, n_gcn_layers, dropout)

        # ---- attention ----
        self.attention = MacroMicroAttention(
            hidden_dim, n_probes, n_heads, dropout)

        # ---- fusion ----
        self.mm_fusion = MicroMacroFusion()
        self.sd_fusion = StaticDynamicFusion(hidden_dim)

        # ---- prediction ----
        self.pred_head = PredictionHead(hidden_dim, dropout, lp_gate_bias)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2 or "Q_macro" in name:
                continue
            nn.init.xavier_uniform_(p)

    # ── internal: full pipeline on a single graph ────────────────────────
    def _forward_single(self, var_feats, con_feats, edge_index, edge_val,
                        return_aux=False):
        """Run the full pipeline on one graph.  Returns dict of predictions."""
        # extract var types BEFORE normalisation (binary cols are safe either
        # way, but reading from the original tensor is semantically cleaner)
        var_types = var_feats[:, [_COL_BIN, _COL_INT, _COL_IMP, _COL_CON]]
        lp_sol = var_feats[:, 8]                         # LP relaxation solution

        # 0. Normalise
        var_feats = self.normalizer.normalize_var(var_feats)
        con_feats = self.normalizer.normalize_con(con_feats)
        edge_val  = self.normalizer.normalize_edge(edge_val)

        static_f = var_feats[:, STATIC_VAR_IDX]         # [N, 9]
        dynamic_f = var_feats[:, DYNAMIC_VAR_IDX]        # [N, 12]

        # 1. GCN
        static_out, _ = self.static_gcn(
            static_f, con_feats, edge_index, edge_val)
        dynamic_out, _ = self.dynamic_gcn(
            dynamic_f, con_feats, edge_index, edge_val)

        # 2. Attention  (returns H_feedback, A_forward, H_macro[, aux])
        if return_aux:
            H_fb, A_fwd, H_macro, attn_aux = self.attention(
                static_out, dynamic_out, return_aux=True)
        else:
            H_fb, A_fwd, H_macro = self.attention(static_out, dynamic_out)

        # 3. Micro-Macro gated fusion
        V_updated = self.mm_fusion(H_fb, dynamic_out, A_fwd, H_macro)

        # 4. Static-Dynamic gated fusion
        Feature_final = self.sd_fusion(V_updated, static_out)

        # 5. Type-aware prediction (with LP skip connection)
        result = self.pred_head(Feature_final, var_types, lp_sol)

        if return_aux:
            result["aux"] = {
                **attn_aux,                     # A_backward, V_micro
                "H_macro": H_macro,             # [M, d]
                "static_out": static_out,       # [N, d]
                "dynamic_out": dynamic_out,     # [N, d]
            }
        return result

    # ── public: single graph ─────────────────────────────────────────────
    def forward(self, var_feats, con_feats, edge_index, edge_val,
                return_aux=False):
        """
        var_feats  : [N, 21]
        con_feats  : [C, 6]
        edge_index : [2, E]  (row0=con_idx, row1=var_idx)
        edge_val   : [E]
        return_aux : if True, result['aux'] contains intermediate tensors
        Returns    : dict  (see PredictionHead.forward)
        """
        return self._forward_single(
            var_feats, con_feats, edge_index, edge_val, return_aux)

    # ── public: PyG-batched graphs ───────────────────────────────────────
    def forward_batch(self, var_feats, con_feats, edge_index, edge_val,
                      var_batch, con_batch):
        """
        Same inputs as forward() but concatenated over multiple graphs.
        var_batch / con_batch : [total_N] / [total_C]  graph id per node.
        """
        # extract types before normalisation
        var_types = var_feats[:, [_COL_BIN, _COL_INT, _COL_IMP, _COL_CON]]

        # 0. Per-graph normalisation
        var_feats = var_feats.clone()
        con_feats = con_feats.clone()
        edge_val  = edge_val.clone()
        # determine edge-to-graph mapping via variable indices
        edge_graph = var_batch[edge_index[1]]
        for gid in var_batch.unique():
            vm = var_batch == gid
            cm = con_batch == gid
            em = edge_graph == gid
            var_feats[vm] = self.normalizer.normalize_var(var_feats[vm])
            con_feats[cm] = self.normalizer.normalize_con(con_feats[cm])
            edge_val[em]  = self.normalizer.normalize_edge(edge_val[em])

        static_f = var_feats[:, STATIC_VAR_IDX]
        dynamic_f = var_feats[:, DYNAMIC_VAR_IDX]

        # GCN runs on the full batched graph (message passing is local)
        static_out, _ = self.static_gcn(
            static_f, con_feats, edge_index, edge_val)
        dynamic_out, _ = self.dynamic_gcn(
            dynamic_f, con_feats, edge_index, edge_val)

        # Attention + micro-macro fusion must run per-graph
        V_updated = torch.zeros_like(static_out)
        for gid in var_batch.unique():
            mask = var_batch == gid
            s_g, d_g = static_out[mask], dynamic_out[mask]
            H_fb, A_fwd, H_macro = self.attention(s_g, d_g)
            V_updated[mask] = self.mm_fusion(H_fb, d_g, A_fwd, H_macro)

        # Static-dynamic fusion and prediction run on the full batch
        Feature_final = self.sd_fusion(V_updated, static_out)
        return self.pred_head(Feature_final, var_types)
