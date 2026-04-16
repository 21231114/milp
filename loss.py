"""
loss.py — Multi-objective composite loss for MILP GNN
=====================================================

Components
----------
1. **Supervised loss** (dynamic-temperature Boltzmann over all solutions, type-aware)
   - Binary:     BCE with label smoothing
   - Integer:    discretised-logistic NLL
   - Continuous: MSE
   - Variable-type weighting: binary > small-range int > large-range int > cont
   - Solution weighting: dynamic Boltzmann  T = mean_gap / ln(1/α), α=0.05

2. **STE + constraint violation**
   - Straight-through round for discrete variables
   - Constraint violation  max(0, Ax − b)
   - Predicted objective value penalty

3. **Orthogonality regularisation**   ||Q_norm · Q_norm^T − I||_F^2

4. **Reconstruction regularisation**
   V̂ = A_backward · W_v_back(LayerNorm(H_macro))
   L_recon = MSE(V_micro, V̂)       (V_micro detached as target)

5. **Entropy regularisation** (extra trick)
   Encourages decisive binary predictions: −mean(p·log p + (1−p)·log(1−p))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PredictionHead          # for integer_log_prob

# ── Column indices in raw var_feats (before model normalisation) ─────────
_COL_BIN = 1
_COL_INT = 2
_COL_IMP = 3
_COL_CON = 4
_COL_RAW_OBJ = 20


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator: forward = round(x), backward = identity."""
    return x + (torch.round(x) - x).detach()


def _type_masks(var_feats: torch.Tensor):
    """Return boolean masks for binary / integer / continuous variables."""
    is_bin  = var_feats[:, _COL_BIN].bool()
    is_int  = var_feats[:, _COL_INT].bool() | var_feats[:, _COL_IMP].bool()
    is_cont = var_feats[:, _COL_CON].bool()
    # variables that are none of the above default to binary treatment
    is_unknown = ~(is_bin | is_int | is_cont)
    is_bin = is_bin | is_unknown
    return is_bin, is_int, is_cont


# ══════════════════════════════════════════════════════════════════════════
#  Composite Loss
# ══════════════════════════════════════════════════════════════════════════

class MILPCompositeLoss(nn.Module):
    """
    Multi-objective loss for the MILP GNN model.

    Parameters
    ----------
    hidden_dim       : int    must match model hidden_dim
    w_sup            : float  weight for supervised loss              (1.0)
    w_viol           : float  weight for STE constraint-violation     (0.1)
    w_orth           : float  weight for Q_macro orthogonality        (0.01)
    w_recon          : float  weight for reconstruction loss          (0.1)
    w_entropy        : float  weight for binary entropy regulariser   (0.01)
    label_smoothing  : float  label smoothing for binary BCE          (0.01)
    type_w_bin       : float  variable-type weight for binary vars    (1.0)
    type_w_int       : float  base weight for integer vars            (0.8)
    type_w_cont      : float  weight for continuous vars              (0.3)
    minimise         : bool   fallback when data has no minimise key  (True)
    sup_loss_type    : str    supervised loss type: 'default' (BCE/NLL/MSE per
                              variable type) or 'mse' (MSE for all types)  ('default')
    """

    def __init__(self, hidden_dim: int,
                 w_sup: float = 1.0,
                 w_viol: float = 0.1,
                 w_orth: float = 0.01,
                 w_recon: float = 0.1,
                 w_entropy: float = 0.01,
                 label_smoothing: float = 0.01,
                 type_w_bin: float = 1.0,
                 type_w_int: float = 0.8,
                 type_w_cont: float = 0.3,
                 minimise: bool = True,
                 sup_loss_type: str = "default"):
        super().__init__()
        self.w_sup = w_sup
        self.w_viol = w_viol
        self.w_orth = w_orth
        self.w_recon = w_recon
        self.w_entropy = w_entropy
        self.label_smoothing = label_smoothing
        self.type_w_bin = type_w_bin
        self.type_w_int = type_w_int
        self.type_w_cont = type_w_cont
        self.minimise = minimise
        if sup_loss_type not in ("default", "mse"):
            raise ValueError(f"sup_loss_type must be 'default' or 'mse', got '{sup_loss_type}'")
        self.sup_loss_type = sup_loss_type

        # LayerNorm used in reconstruction loss
        self.recon_norm = nn.LayerNorm(hidden_dim)

    # ══════════════════════════════════════════════════════════════════════
    #  1.  Supervised loss
    # ══════════════════════════════════════════════════════════════════════

    def _supervised_loss(self, model_out, sols, objs, var_feats, minimise):
        """
        仅使用最优解（Best Solution）作为监督信号的损失函数。
        """
        is_bin, is_int, is_cont = _type_masks(var_feats)
        K, N = sols.shape
        device = sols.device
        eps = self.label_smoothing

        # ── Step 1: 提取唯一的最优解 ──
        if minimise:
            best_idx = objs.argmin()
        else:
            best_idx = objs.argmax()
        
        best_sol = sols[best_idx]  # 仅提取最好的解，形状变为 [N]

        # ── Step 2: Variable-type weights ──
        # 仍然使用所有解来计算整数变量的极差（为了归一化权重），这样比较稳妥
        var_w = torch.zeros(N, device=device)
        var_w[is_bin] = self.type_w_bin

        if is_int.any():
            int_ranges = sols[:, is_int].max(0).values - sols[:, is_int].min(0).values
            var_w[is_int] = self.type_w_int / (1.0 + int_ranges)

        if is_cont.any():
            var_w[is_cont] = self.type_w_cont

        var_w = var_w / (var_w.sum() + 1e-8) * N              # normalise, mean=1

        # ── Step 3: 仅针对最优解计算 Per-variable loss ──
        # 不再有 K 维度，直接计算一维 [N] 的损失
        per_var_loss = torch.zeros(N, device=device)

        if self.sup_loss_type == "mse":
            # MSE for all variable types
            pred = model_out["pred"]
            per_var_loss = F.mse_loss(pred, best_sol, reduction="none")
        else:
            # Default: type-aware losses (BCE / NLL / MSE)
            if is_bin.any():
                logits = model_out["binary_logits"][is_bin]
                targets = best_sol[is_bin]
                targets = targets * (1.0 - eps) + 0.5 * eps        # label smoothing
                per_var_loss[is_bin] = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none")

            if is_int.any():
                mu  = model_out["integer_mu"][is_int]
                lsd = model_out["integer_log_std"][is_int]
                per_var_loss[is_int] = -PredictionHead.integer_log_prob(
                    mu, lsd, best_sol[is_int])

            if is_cont.any():
                val = model_out["continuous_val"][is_cont]
                per_var_loss[is_cont] = F.mse_loss(
                    val, best_sol[is_cont], reduction="none")

        # 变量加权平均即可，不再需要对多解进行 Boltzmann 加权
        return (per_var_loss * var_w).mean()
    # ══════════════════════════════════════════════════════════════════════
    #  2.  STE + constraint violation + objective
    # ══════════════════════════════════════════════════════════════════════

    def _ste_violation_loss(self, model_out, var_feats, con_feats,
                           edge_index, edge_values, objs, minimise):
        """
        Compute constraint violation and objective quality using STE-rounded
        predictions.  Uses raw (ecole-normalised) coefficients and bias.

        Returns (scalar_loss, per_constraint_violations [C], x_ste [N]).
        The per-constraint violations are also stored in model_out so they
        can be reused by the Lagrangian without recomputation.
        """
        is_bin, is_int, is_cont = _type_masks(var_feats)
        pred = model_out["pred"]
        n_cons = con_feats.size(0)

        # ── STE rounding ──
        x_ste = torch.zeros_like(pred)
        x_ste[is_bin]  = ste_round(pred[is_bin])
        x_ste[is_int]  = ste_round(pred[is_int])
        x_ste[is_cont] = pred[is_cont]

        # ── Per-constraint violation ──
        ci, vi = edge_index[0], edge_index[1]
        weighted_x = edge_values * x_ste[vi]                   # [E]
        ax = torch.zeros(n_cons, device=pred.device)
        ax.scatter_add_(0, ci, weighted_x)

        con_bias = con_feats[:, 0]                              # RHS (ecole norm)
        slack = ax - con_bias                                   # [C]  >0 = violated
        hard_viol = F.relu(slack)                               # exact violations

        # Use softplus for the LOSS (provides gradient BEFORE violation):
        #   softplus(x, β) ≈ relu(x) for large x, but grad > 0 near x=0
        # β=5 → transition zone ≈ ±0.5 around the boundary
        loss_viol = F.softplus(slack, beta=5.0).mean()

        # ── Predicted objective penalty ──
        raw_obj = var_feats[:, _COL_RAW_OBJ]                   # un-normalised
        obj_pred = (raw_obj * x_ste).sum()

        obj_best  = objs.min() if minimise else objs.max()
        obj_range = (objs.max() - objs.min()).clamp(min=1e-6)
        if minimise:
            loss_obj = (obj_pred - obj_best) / obj_range
        else:
            loss_obj = (obj_best - obj_pred) / obj_range
        loss_obj = F.relu(loss_obj)                             # only penalise if worse

        return loss_viol + 0.1 * loss_obj, hard_viol, loss_obj

    # ══════════════════════════════════════════════════════════════════════
    #  3.  Orthogonality regularisation
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _orthogonality_loss(Q_macro: torch.Tensor) -> torch.Tensor:
        """||Q̂ · Q̂ᵀ − I||²_F   where Q̂ = row-normalised Q_macro."""
        Q_norm = F.normalize(Q_macro, dim=-1)                  # [M, d]
        gram = Q_norm @ Q_norm.t()                             # [M, M]
        I = torch.eye(gram.size(0), device=gram.device)
        return (gram - I).pow(2).mean()

    # ══════════════════════════════════════════════════════════════════════
    #  4.  Reconstruction regularisation
    # ══════════════════════════════════════════════════════════════════════

    def _reconstruction_loss(self, aux, model) -> torch.Tensor:
        """
        V̂ = A_backward · W_v_back(LN(H_macro))
        L = MSE(V_micro, V̂)
        V_micro is detached (used as a fixed target).
        """
        A_bwd   = aux["A_backward"]                            # [N, M]
        H_macro = aux["H_macro"]                               # [M, d]
        V_micro = aux["V_micro"]                               # [N, d]

        H_normed = self.recon_norm(H_macro)                    # [M, d]
        V_hat = A_bwd @ model.attention.W_v_back(H_normed)     # [N, d]

        return F.mse_loss(V_hat, V_micro.detach())

    # ══════════════════════════════════════════════════════════════════════
    #  5.  Binary entropy regularisation  (extra trick)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _entropy_loss(binary_logits: torch.Tensor,
                      is_bin: torch.Tensor) -> torch.Tensor:
        """
        Negative entropy of binary predictions — encourages probabilities
        to move towards 0 or 1 (decisive predictions) rather than sitting
        at 0.5.  Acts as a sharpening regulariser.
        """
        if not is_bin.any():
            return binary_logits.new_tensor(0.0)
        p = torch.sigmoid(binary_logits[is_bin])
        p = p.clamp(1e-6, 1.0 - 1e-6)
        ent = -(p * p.log() + (1.0 - p) * (1.0 - p).log())   # [N_bin]
        return ent.mean()                                      # minimise → sharpen

    # ══════════════════════════════════════════════════════════════════════
    #  Main entry point
    # ══════════════════════════════════════════════════════════════════════

    def forward(self, model_out: dict, data: dict, model: nn.Module) -> dict:
        """
        Parameters
        ----------
        model_out : dict returned by  model(..., return_aux=True)
                    must contain keys: binary_logits, integer_mu, integer_log_std,
                    continuous_val, pred, aux
        data      : dict from the .pkl file with keys:
                    var_feats [N,21], con_feats [C,6], edge_indices [2,E],
                    edge_values [E], sols [K,N], objs [K]
        model     : MILPGNNModel  (for accessing Q_macro, W_v_back)

        Returns
        -------
        dict  with keys: total, sup, viol, orth, recon, entropy
        """
        device = model_out["pred"].device
        vf   = data["var_feats"].to(device)
        cf   = data["con_feats"].to(device)
        ei   = data["edge_indices"].to(device)
        ev   = data["edge_values"].to(device)
        sols = data["sols"].to(device)
        objs = data["objs"].to(device)
        minimise = data.get("minimise", self.minimise)

        is_bin, _, _ = _type_masks(vf)

        losses = {}

        # 1. Supervised
        losses["sup"] = self._supervised_loss(model_out, sols, objs, vf, minimise)

        # 2. STE + constraint violation (returns scalar loss + per-constraint violations + obj_loss)
        losses["viol"], losses["per_con_viol"], losses["obj_loss"] = \
            self._ste_violation_loss(model_out, vf, cf, ei, ev, objs, minimise)

        # 3. Orthogonality
        losses["orth"] = self._orthogonality_loss(model.attention.Q_macro)

        # 4. Reconstruction
        losses["recon"] = self._reconstruction_loss(model_out["aux"], model)

        # 5. Entropy
        losses["entropy"] = self._entropy_loss(model_out["binary_logits"], is_bin)

        # ── Weighted total ──
        losses["total"] = (
            self.w_sup     * losses["sup"]
          + self.w_viol    * losses["viol"]
          + self.w_orth    * losses["orth"]
          + self.w_recon   * losses["recon"]
          + self.w_entropy * losses["entropy"]
        )

        return losses


# ══════════════════════════════════════════════════════════════════════════
#  GradNorm: Automatic Loss Weight Balancing
# ══════════════════════════════════════════════════════════════════════════

class GradNormBalancer(nn.Module):
    """
    GradNorm (Chen et al., 2018) wrapper for MILPCompositeLoss.

    Replaces fixed loss weights with **learned** ones that adapt so
    gradient norms from all loss components stay balanced.  Tasks
    that train slower automatically receive higher weight.

    Algorithm per step
    ------------------
    1.  Compute per-task gradient norm  G_i = ||∇_shared (w_i · L_i)||
    2.  Relative training rate  r_i = L_i / L_i(0)
    3.  Target gradient norm    Ĝ_i = mean(G) · (r_i / mean(r))^α
    4.  Multiplicative weight update toward target  (smoothed, clipped)

    Training loop
    -------------
    >>> balancer = GradNormBalancer(base_loss, alpha=1.5)
    >>> # --- each step ---
    >>> out    = model(vf, cf, ei, ev, return_aux=True)
    >>> losses = balancer(out, data, model)          # forward
    >>> balancer.step(losses, model)                 # update weights
    >>> losses["total"].backward()                   # backward model
    >>> optimizer.step()

    Parameters
    ----------
    base_loss      : MILPCompositeLoss instance
    alpha          : float  asymmetry strength (1.5 recommended)
                     0 → uniform weighting, larger → stronger rebalancing
    update_rate    : float  smoothing factor for weight update  (0.1)
    shared_layer   : str    name fragment to identify the shared layer
                     whose gradient norms are monitored
    """

    TASK_KEYS = MILPCompositeLoss.TASK_KEYS \
        if hasattr(MILPCompositeLoss, "TASK_KEYS") \
        else ["sup", "viol", "orth", "recon", "entropy"]

    def __init__(self, base_loss: MILPCompositeLoss,
                 alpha: float = 1.5,
                 update_rate: float = 0.1,
                 shared_layer: str = "sd_fusion.W_gate",
                 w_min: float = 0.3,
                 w_max: float = 2.5):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.update_rate = update_rate
        self.shared_layer = shared_layer
        self.w_min = w_min
        self.w_max = w_max
        n = len(self.TASK_KEYS)

        # log-space weights (softmax → positive, sum = n)
        self.log_w = nn.Parameter(torch.zeros(n))

        # initial loss snapshot (set on first forward)
        self.register_buffer("L0", torch.ones(n))
        self.register_buffer("_init", torch.tensor(False))

    # ── helpers ──────────────────────────────────────────────────────────

    def _weights(self) -> torch.Tensor:
        """Positive, normalised weights (sum = n_tasks), clamped to [w_min, w_max]."""
        n = len(self.TASK_KEYS)
        w = F.softmax(self.log_w, dim=0) * n
        w = w.clamp(min=self.w_min, max=self.w_max)
        # re-normalise so weights still sum to n
        w = w / (w.sum() + 1e-8) * n
        return w

    def weight_dict(self) -> dict:
        """Current weights as a readable dict (for logging)."""
        w = self._weights()
        return {k: round(w[i].item(), 4) for i, k in enumerate(self.TASK_KEYS)}

    def _find_shared_param(self, model):
        """Return the first weight tensor whose name contains shared_layer."""
        for name, p in model.named_parameters():
            if self.shared_layer in name and "weight" in name:
                return p
        raise ValueError(
            f"No parameter matching '{self.shared_layer}' + 'weight' found")

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, model_out: dict, data: dict, model: nn.Module) -> dict:
        """
        Compute individual losses (via base_loss) then apply learned weights.
        Returns the same dict as MILPCompositeLoss, with ``total`` recomputed.
        """
        raw = self.base_loss(model_out, data, model)

        task_losses = torch.stack([raw[k] for k in self.TASK_KEYS])

        # snapshot initial losses (absolute value — some losses are negative)
        if not self._init:
            self.L0.copy_(task_losses.detach().abs().clamp(min=1e-4))
            self._init.fill_(True)

        w = self._weights()
        raw["total"] = (w * task_losses).sum()

        # stash for step()
        raw["_tl"] = task_losses
        raw["_w"]  = w
        return raw

    # ── weight update ────────────────────────────────────────────────────

    def step(self, losses: dict, model: nn.Module):
        """
        GradNorm weight update.  Must be called **after** forward() and
        **before** losses["total"].backward().

        Internally computes per-task gradient norms using the retained
        computation graph (retain_graph=True) and adjusts weights so that
        slower-learning tasks receive higher weight.
        """
        tl = losses["_tl"]                 # [n] task losses (in graph)
        w  = losses["_w"]                  # [n] current weights
        n  = len(self.TASK_KEYS)
        shared = self._find_shared_param(model)

        # ---- per-task gradient norms G_i = ||∇_shared(w_i · L_i)|| ----
        G = []
        for i in range(n):
            try:
                grad = torch.autograd.grad(
                    w[i] * tl[i], shared,
                    retain_graph=True, allow_unused=True)[0]
                G.append(grad.detach().norm() if grad is not None
                         else tl.new_tensor(0.0))
            except RuntimeError:
                G.append(tl.new_tensor(0.0))
        G = torch.stack(G)

        # ---- skip tasks with no gradient path to shared layer ----
        active = G > 1e-10
        # Also skip tasks whose loss is near-zero (already solved, e.g. viol=0)
        # to prevent them from hogging weight via noisy gradient ratios
        near_zero = tl.detach().abs() < 1e-6
        active = active & ~near_zero
        if active.sum() < 2:
            return                             # need ≥ 2 active tasks

        # ---- relative training rate r̃_i  (abs handles negative losses) --
        r = tl.detach().abs() / (self.L0 + 1e-8)
        r_tilde = r / (r[active].mean() + 1e-8)

        # ---- target gradient norm Ĝ_i ----
        G_target = G[active].mean() * r_tilde.pow(self.alpha)

        # ---- smooth multiplicative update (only active tasks) ----
        with torch.no_grad():
            ratio = torch.ones_like(G)
            ratio[active] = (G_target[active] / (G[active] + 1e-8)).clamp(0.5, 2.0)
            self.log_w.data += ratio.log() * self.update_rate
            self.log_w.data -= self.log_w.data.mean()   # re-centre

