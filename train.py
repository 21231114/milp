#!/usr/bin/env python3
"""
train.py — Training loop with dual-ascent Lagrangian optimisation
=================================================================

Two-level optimisation per step
-------------------------------
1. **Inner** (gradient descent on θ):
      L_total = GradNorm_balanced_losses  +  w_lag · L_Lagrangian
      L_Lagrangian = α·Obj_Loss + Σ_j λ_j · Viol_j
      θ ← θ − lr · ∇_θ L_total

2. **Outer** (gradient ascent on λ):
      λ_j ← max(0,  λ_j + η · Viol_j)

Usage
-----
    # Generate data first (if not done):
    python build_dataset.py --problems CA --workers 20

    # Train on CA only:
    python train.py --problems CA --epochs 50

    # Train on all problems:
    python train.py --epochs 100

    # Resume from checkpoint:
    python train.py --resume runs/run_001/best.pt
"""

import os
import sys
import time
import json
import random
import pickle
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import set_seed
from model import MILPGNNModel
from loss import (MILPCompositeLoss, GradNormBalancer,
                  ste_round, _type_masks, _COL_RAW_OBJ)

# ── Per-problem default batch sizes (same as baseline) ───────────────────
TASK_BATCH_SIZE = {"CA": 4, "WA": 4, "IP": 4, "IS": 4, "SC": 1}

# ── Per-problem objective sense ──────────────────────────────────────────
# True = lower objective is better (minimisation problem)
# False = higher objective is better (maximisation problem)
MINIMISE_MAP = {"CA": False, "IP": False, "IS": False, "SC": True, "WA": True}


# ══════════════════════════════════════════════════════════════════════════
#  Logging utilities
# ══════════════════════════════════════════════════════════════════════════

def setup_logger(run_path: str) -> logging.Logger:
    """Create a logger that writes to both console and a log file."""
    logger = logging.getLogger("milp_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(run_path, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ── Default hyperparameters ──────────────────────────────────────────────
DEFAULTS = {
    "hidden_dim": 64,
    "n_gcn_layers": 2,
    "n_probes": 16,
    "dropout": 0.1,
    "lr": 1e-3,
    "epochs": 100,
    "seed": 42,
    "grad_clip": 1.0,
    "lagrangian_lr": 0.005,
    "lagrangian_weight": 0.1,
    "obj_weight": 0.1,
    "gradnorm_alpha": 1.5,
    "patience": 15,
    "batch_size": None,
    "warmup_epochs": 5,
    "problems": ["CA", "IP", "IS", "SC", "WA"],
    # Fixed-weight baseline (GradNorm disabled)
    "w_sup": 1.0,
    "w_viol": 0.1,
    "w_recon": 0.0,
    "w_orth": 0.0,
    "w_entropy": 0.0,
    # LP skip connection initial trust
    "lp_gate_bias": 2.0,
    # Validation metric for early stopping
    "val_metric": "acc",
}


# ══════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════

class MILPDataset:
    """Lazy-loading dataset of processed .pkl instances."""

    def __init__(self, file_tuples):
        """file_tuples: list of (file_path, problem_type) sorted by path."""
        self.files = [t[0] for t in file_tuples]
        self.prob_types = [t[1] for t in file_tuples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            d = pickle.load(f)
        prob = self.prob_types[idx]
        minimise = MINIMISE_MAP.get(prob, True)

        var_feats = torch.as_tensor(d["var_feats"], dtype=torch.float32)
        objs      = torch.as_tensor(d["objs"],      dtype=torch.float32)

        # Unify to minimisation: negate objective-related features for max problems.
        # Col 0 : ecole-normalised obj coefficient
        # Col 20: raw (un-normalised) obj coefficient  (used in obj-quality penalty)
        # objs  : negate so argmin always selects the best solution
        if not minimise:
            var_feats = var_feats.clone()
            var_feats[:, 0]  = -var_feats[:, 0]
            var_feats[:, 20] = -var_feats[:, 20]
            objs = -objs

        return {
            "var_feats":   var_feats,
            "con_feats":   torch.as_tensor(d["con_feats"],   dtype=torch.float32),
            "edge_indices": torch.as_tensor(d["edge_indices"], dtype=torch.long),
            "edge_values": torch.as_tensor(d["edge_values"], dtype=torch.float32),
            "sols":        torch.as_tensor(d["sols"],        dtype=torch.float32),
            "objs":        objs,
            "n_vars":      d["n_vars"],
            "n_cons":      d["n_cons"],
            "instance_id": os.path.basename(self.files[idx]),
            "minimise":    True,   # always minimise after sign flip
        }


def collect_files(data_dir, problems, split):
    """Gather all .pkl paths for given problems and split."""
    files = []
    for prob in problems:
        d = os.path.join(data_dir, prob, split)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".pkl"):
                    files.append((os.path.join(d, f), prob))
    return sorted(files, key=lambda x: x[0])


# ══════════════════════════════════════════════════════════════════════════
#  Lagrangian Dual Ascent
# ══════════════════════════════════════════════════════════════════════════

class LagrangianManager:
    """
    Maintains per-instance Lagrangian multipliers λ ∈ R^{J} (one per
    constraint).  After each model weight update, performs gradient ascent:
        λ_j ← max(0, λ_j + η · Viol_j)
    """

    def __init__(self, lr: float = 0.005):
        self.lr = lr
        self._lam: dict[str, torch.Tensor] = {}

    def get(self, instance_id: str, n_cons: int, device) -> torch.Tensor:
        if instance_id not in self._lam:
            self._lam[instance_id] = torch.zeros(n_cons, device=device)
        return self._lam[instance_id].to(device)

    def update(self, instance_id: str, violations: torch.Tensor):
        """Dual ascent step."""
        lam = self._lam[instance_id]
        lam.data = (lam + self.lr * violations.detach()).clamp_(min=0.0)

    def mean_lambda(self) -> float:
        if not self._lam:
            return 0.0
        return float(np.mean([v.mean().item() for v in self._lam.values()]))

    def state_dict(self):
        return {k: v.cpu() for k, v in self._lam.items()}

    def load_state_dict(self, s):
        self._lam = {k: v for k, v in s.items()}


def compute_per_constraint_violation(model_out, var_feats, con_feats,
                                     edge_index, edge_values):
    """STE-rounded prediction → per-constraint violation [C].
    NOTE: prefer using losses['per_con_viol'] from MILPCompositeLoss instead
    to avoid double computation."""
    is_bin, is_int, is_cont = _type_masks(var_feats)
    pred = model_out["pred"]

    x = torch.zeros_like(pred)
    x[is_bin]  = ste_round(pred[is_bin])
    x[is_int]  = ste_round(pred[is_int])
    x[is_cont] = pred[is_cont]

    ci, vi = edge_index[0], edge_index[1]
    ax = torch.zeros(con_feats.size(0), device=pred.device)
    ax.scatter_add_(0, ci, edge_values * x[vi])
    return F.relu(ax - con_feats[:, 0])          # [C]


def lagrangian_loss(per_con_viol, lambdas, obj_norm):
    """
    L_RL = Σ_j λ_j · Viol_j  +  0.1 · obj_norm

    Uses pre-computed per-constraint violations from MILPCompositeLoss
    (avoids duplicate STE computation).
    """
    return (lambdas * per_con_viol).sum() + 0.1 * obj_norm


# ══════════════════════════════════════════════════════════════════════════
#  Training & Validation
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataset, loss_fn, lag_mgr,
                    optimizer, device, cfg, epoch=0):
    model.train()
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    batch_size = cfg.get("batch_size", 1)

    meters = defaultdict(float)
    n = len(indices)
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        actual_bs = len(batch_indices)

        optimizer.zero_grad()

        for j, idx in enumerate(batch_indices):
            data = dataset[idx]
            vf = data["var_feats"].to(device)
            cf = data["con_feats"].to(device)
            ei = data["edge_indices"].to(device)
            ev = data["edge_values"].to(device)
            objs = data["objs"].to(device)
            inst_id = data["instance_id"]

            lambdas = lag_mgr.get(inst_id, data["n_cons"], device)

            # ── forward ──
            out = model(vf, cf, ei, ev, return_aux=True)
            base = loss_fn(out, data, model)

            # Reuse pre-computed per-constraint violations from base loss
            viols = base["per_con_viol"]
            lag = lagrangian_loss(viols, lambdas, obj_norm=base["obj_loss"])

            # ── Lagrangian warmup ──
            warmup_epochs = max(1, int(cfg["epochs"] * 0.1))
            lag_w = cfg["lagrangian_weight"] * min(1.0, epoch / warmup_epochs)

            # ── backward (accumulate, scale by 1/batch_size) ──
            total = (base["total"] + lag_w * lag) / actual_bs
            total.backward()

            # ── dual ascent on λ (per instance) ──
            lag_mgr.update(inst_id, viols)

            # ── bookkeeping (unscaled values) ──
            total_unscaled = base["total"].item() + lag_w * lag.item()
            meters["total"]    += total_unscaled
            meters["lag"]      += lag.item()
            meters["viol_mean"] += viols.mean().item()
            for k in ("sup", "viol", "orth", "recon", "entropy"):
                meters[k] += base[k].item()

        # ── optimizer step (once per batch) ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

    return {k: v / n for k, v in meters.items()}


@torch.no_grad()
def validate(model, dataset, base_loss, device):
    model.eval()
    meters = defaultdict(float)
    n = len(dataset)

    for idx in range(n):
        data = dataset[idx]
        vf = data["var_feats"].to(device)
        cf = data["con_feats"].to(device)
        ei = data["edge_indices"].to(device)
        ev = data["edge_values"].to(device)
        objs = data["objs"].to(device)
        sols = data["sols"].to(device)

        out = model(vf, cf, ei, ev, return_aux=True)
        losses = base_loss(out, data, model)

        for k in ("sup", "viol", "orth", "recon", "entropy", "total"):
            meters[f"loss/{k}"] += losses[k].item()

        # ── practical metrics ──
        is_bin = vf[:, 1].bool()
        pred = out["pred"]

        # binary accuracy  (vs best Gurobi solution)
        minimise = data.get("minimise", True)
        if is_bin.any():
            best_idx = objs.argmin() if minimise else objs.argmax()
            best_sol = sols[best_idx]
            acc = ((pred[is_bin] > 0.5).float() == best_sol[is_bin]).float().mean()
            meters["binary_acc"] += acc.item()

        # constraint violation rate & magnitude
        x = torch.zeros_like(pred)
        x[is_bin] = (pred[is_bin] > 0.5).float()
        is_int = vf[:, 2].bool() | vf[:, 3].bool()
        x[is_int] = torch.round(pred[is_int])
        is_cont = vf[:, 4].bool()
        x[is_cont] = pred[is_cont]

        ci, vi = ei[0], ei[1]
        ax = torch.zeros(cf.size(0), device=device)
        ax.scatter_add_(0, ci, ev * x[vi])
        v = F.relu(ax - cf[:, 0])
        meters["viol_rate"]  += (v > 1e-6).float().mean().item()
        meters["viol_mag"]   += v.mean().item()

        # objective gap  (penalised if infeasible)
        is_feasible = (v.max() < 1e-6)
        obj_pred = (vf[:, _COL_RAW_OBJ] * x).sum()
        obj_best = objs.min() if minimise else objs.max()
        obj_range = (objs.max() - objs.min()).clamp(min=1e-6)
        if minimise:
            raw_gap = ((obj_pred - obj_best) / obj_range).item()
        else:
            raw_gap = ((obj_best - obj_pred) / obj_range).item()
        # infeasible solutions get penalty = max(1, gap) + viol_sum
        if is_feasible:
            meters["obj_gap"] += max(0.0, raw_gap)
            meters["n_feasible"] += 1
        else:
            meters["obj_gap"] += max(1.0, raw_gap) + v.sum().item()

    result = {k: v / n for k, v in meters.items()}
    result["feasible_rate"] = meters["n_feasible"] / n if n > 0 else 0.0
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ══════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scheduler, balancer, lag_mgr,
                    epoch, best_metric, cfg):
    torch.save({
        "epoch":       epoch,
        "best_metric": best_metric,
        "cfg":         cfg,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict() if scheduler else None,
        "balancer":    balancer.state_dict(),
        "lagrangian":  lag_mgr.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, balancer, lag_mgr):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except (KeyError, ValueError, TypeError):
            # Checkpoint may have been saved with a different scheduler type
            # (e.g., CosineAnnealingLR vs SequentialLR). Skip loading scheduler
            # state — it will restart from its initial state.
            pass
    balancer.load_state_dict(ckpt["balancer"])
    lag_mgr.load_state_dict(ckpt["lagrangian"])
    return ckpt["epoch"], ckpt["best_metric"], ckpt.get("cfg", {})


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train MILP GNN")
    parser.add_argument("--problems", nargs="+",
                        default=["CA", "IP", "IS", "SC", "WA"])
    parser.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "processed"))
    parser.add_argument("--run_dir", default=os.path.join(SCRIPT_DIR, "runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_gcn_layers", type=int, default=2,
                        help="Number of GCN message-passing rounds per stream (default 2).")
    parser.add_argument("--n_probes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lagrangian_lr", type=float, default=0.005)
    parser.add_argument("--lagrangian_weight", type=float, default=0.1)
    parser.add_argument("--obj_weight", type=float, default=0.1)
    parser.add_argument("--gradnorm_alpha", type=float, default=1.5)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for model layers.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Mini-batch size for gradient accumulation. "
                             "If not set, uses per-problem defaults from "
                             "TASK_BATCH_SIZE.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of linear warmup epochs before cosine decay.")
    parser.add_argument("--w_sup", type=float, default=1.0,
                        help="Fixed weight for supervised loss.")
    parser.add_argument("--w_viol", type=float, default=0.1,
                        help="Fixed weight for STE constraint-violation loss.")
    parser.add_argument("--w_recon", type=float, default=0.0,
                        help="Fixed weight for reconstruction loss (0 = disabled).")
    parser.add_argument("--w_orth", type=float, default=0.0,
                        help="Fixed weight for orthogonality loss (0 = disabled).")
    parser.add_argument("--w_entropy", type=float, default=0.0,
                        help="Fixed weight for binary entropy regulariser (0 = disabled).")
    parser.add_argument("--sup_loss_type", type=str, default="default",
                        choices=["default", "mse"],
                        help="Supervised loss type: 'default' uses BCE/NLL/MSE per variable "
                             "type; 'mse' uses MSE for all variable types.")
    parser.add_argument("--lp_gate_bias", type=float, default=2.0,
                        help="Initial bias for LP skip-connection gate. "
                             "sigmoid(bias) sets initial LP trust: "
                             "2.0→88%%, 0.0→50%%, -2.0→12%%.")
    parser.add_argument("--val_metric", type=str, default="acc",
                        choices=["acc", "sup", "viol"],
                        help="Validation metric for early stopping and best model selection. "
                             "'acc'=binary accuracy (higher is better, default), "
                             "'sup'=supervised loss (lower is better), "
                             "'viol'=constraint violation loss (lower is better).")
    args = parser.parse_args()

    cfg = {**DEFAULTS, **vars(args)}

    # ── resolve batch_size ──
    if cfg["batch_size"] is None:
        # When training multiple problems, use the minimum batch size
        cfg["batch_size"] = min(
            TASK_BATCH_SIZE.get(p, 4) for p in cfg["problems"])
    print(f"Batch size: {cfg['batch_size']}")  # before logger is created

    # ── seed ──
    set_seed(cfg["seed"])

    # ── device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")  # before logger is created

    # ── data ──
    train_files = collect_files(cfg["data_dir"], cfg["problems"], "train")
    val_files   = collect_files(cfg["data_dir"], cfg["problems"], "val")
    if not train_files:
        sys.exit(f"No training data in {cfg['data_dir']} for {cfg['problems']}.\n"
                 f"Run: python build_dataset.py --problems {' '.join(cfg['problems'])}")
    train_set = MILPDataset(train_files)
    val_set   = MILPDataset(val_files) if val_files else None
    print(f"Train: {len(train_set)} instances  |  Val: {len(val_set) if val_set else 0}")  # before logger

    # ── model ──
    model = MILPGNNModel(
        hidden_dim=cfg["hidden_dim"],
        n_gcn_layers=cfg["n_gcn_layers"],
        n_probes=cfg["n_probes"],
        dropout=cfg["dropout"],
        lp_gate_bias=cfg["lp_gate_bias"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")  # before logger

    # ── loss (fixed weights, no GradNorm) ──
    base_loss = MILPCompositeLoss(
        hidden_dim=cfg["hidden_dim"],
        w_sup=cfg["w_sup"],
        w_viol=cfg["w_viol"],
        w_recon=cfg["w_recon"],
        w_orth=cfg["w_orth"],
        w_entropy=cfg["w_entropy"],
        sup_loss_type=cfg["sup_loss_type"],
    ).to(device)
    balancer = base_loss   # alias — fixed weights, no learnable balancer

    # ── Lagrangian ──
    lag_mgr = LagrangianManager(lr=cfg["lagrangian_lr"])

    # ── optimiser + scheduler (warmup + cosine) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"], weight_decay=1e-4)

    warmup_epochs = cfg["warmup_epochs"]
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["epochs"] - warmup_epochs),
        eta_min=cfg["lr"] * 0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])

    # ── run directory ──
    run_id = 1
    while os.path.exists(os.path.join(cfg["run_dir"], f"run_{run_id:03d}")):
        run_id += 1
    run_path = os.path.join(cfg["run_dir"], f"run_{run_id:03d}")
    os.makedirs(run_path, exist_ok=True)
    writer = SummaryWriter(log_dir=run_path)
    epoch_log_path = os.path.join(run_path, "epoch_metrics.log")
    epoch_log_fh = open(epoch_log_path, "w")
    logger = setup_logger(run_path)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    logger.info(f"Run dir: {run_path}")

    # ── resume ──
    start_epoch = 0
    val_metric = cfg["val_metric"]
    # For 'acc': higher is better → track best as highest; init to -inf
    # For 'sup'/'viol': lower is better → track best as lowest; init to +inf
    best_val_metric = float("-inf") if val_metric == "acc" else float("inf")
    if cfg["resume"]:
        start_epoch, best_val_metric, _ = load_checkpoint(
            cfg["resume"], model, optimizer, scheduler, balancer, lag_mgr)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_{val_metric}={best_val_metric:.4f}")

    # ── training loop ──
    no_improve = 0
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']}  lr={cur_lr:.6f}")
        logger.info(f"{'='*60}")

        # ---- train ----
        train_m = train_one_epoch(
            model, train_set, balancer, lag_mgr,
            optimizer, device, cfg, epoch=epoch)
        scheduler.step()

        # log train
        for k, v in train_m.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        w = {"sup": cfg["w_sup"], "viol": cfg["w_viol"],
             "recon": cfg["w_recon"], "orth": cfg["w_orth"], "entropy": cfg["w_entropy"]}
        for k, v in w.items():
            writer.add_scalar(f"weights/{k}", v, epoch)
        writer.add_scalar("lagrangian/mean_lambda", lag_mgr.mean_lambda(), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - t0
        logger.info(f"\n  Train  total={train_m['total']:.4f}  sup={train_m['sup']:.4f}  "
              f"lag={train_m['lag']:.4f}  viol={train_m['viol_mean']:.5f}  "
              f"[{elapsed:.0f}s]")
        logger.info(f"  Fixed weights: {w}")

        # ---- validate ----
        if val_set and len(val_set) > 0:
            val_m = validate(model, val_set, balancer, device)
            for k, v in val_m.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            val_total = val_m["loss/total"]
            val_sup = val_m["loss/sup"]
            logger.info(f"  Val    total={val_total:.4f}  "
                  f"sup={val_sup:.4f}  "
                  f"acc={val_m.get('binary_acc',0):.4f}  "
                  f"feasible={val_m.get('feasible_rate',0):.1%}  "
                  f"viol_rate={val_m.get('viol_rate',0):.4f}  "
                  f"obj_gap={val_m.get('obj_gap',0):.4f}")

            # ── Epoch metrics log ──
            gn_str = "  ".join(f"{k}={v:.4f}" for k, v in w.items())
            epoch_log_fh.write(
                f"Epoch {epoch+1:>3d}/{cfg['epochs']}  "
                f"lr={cur_lr:.6f}  elapsed={elapsed:.0f}s\n"
                f"  Train | total={train_m['total']:.4f}  sup={train_m['sup']:.4f}  "
                f"viol={train_m.get('viol',0):.4f}  orth={train_m.get('orth',0):.4f}  "
                f"recon={train_m.get('recon',0):.4f}  entropy={train_m.get('entropy',0):.4f}  "
                f"lag={train_m['lag']:.4f}  viol_mean={train_m['viol_mean']:.5f}\n"
                f"  Val   | total={val_total:.4f}  sup={val_sup:.4f}  "
                f"acc={val_m.get('binary_acc',0):.4f}  "
                f"feasible={val_m.get('feasible_rate',0):.1%}  "
                f"viol_rate={val_m.get('viol_rate',0):.4f}  "
                f"viol_mag={val_m.get('viol_mag',0):.4f}  "
                f"obj_gap={val_m.get('obj_gap',0):.4f}\n"
                f"  Weights | {gn_str}\n"
                f"  Lag   | mean_lambda={lag_mgr.mean_lambda():.4f}  "
                f"best_val_{val_metric}={best_val_metric:.4f}  no_improve={no_improve}\n"
                f"{'-'*70}\n"
            )
            epoch_log_fh.flush()

            # ---- early stopping / best model ----
            # Determine current metric value and improvement direction
            if val_metric == "acc":
                current_metric = val_m.get("binary_acc", 0.0)
                improved = current_metric > best_val_metric
            elif val_metric == "sup":
                current_metric = val_sup
                improved = current_metric < best_val_metric
            else:  # viol
                current_metric = val_m["loss/viol"]
                improved = current_metric < best_val_metric

            if improved:
                best_val_metric = current_metric
                save_checkpoint(
                    os.path.join(run_path, "best.pt"),
                    model, optimizer, scheduler, balancer, lag_mgr,
                    epoch, best_val_metric, cfg)
                logger.info(f"  >> New best model saved (val_{val_metric}={best_val_metric:.4f})")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    logger.info(f"\n  Early stopping after {cfg['patience']} epochs "
                          f"without improvement.")
                    break
        else:
            # No validation set — still log training metrics
            gn_str = "  ".join(f"{k}={v:.4f}" for k, v in w.items())
            epoch_log_fh.write(
                f"Epoch {epoch+1:>3d}/{cfg['epochs']}  "
                f"lr={cur_lr:.6f}  elapsed={elapsed:.0f}s\n"
                f"  Train | total={train_m['total']:.4f}  sup={train_m['sup']:.4f}  "
                f"viol={train_m.get('viol',0):.4f}  orth={train_m.get('orth',0):.4f}  "
                f"recon={train_m.get('recon',0):.4f}  entropy={train_m.get('entropy',0):.4f}  "
                f"lag={train_m['lag']:.4f}  viol_mean={train_m['viol_mean']:.5f}\n"
                f"  Weights | {gn_str}\n"
                f"  Lag   | mean_lambda={lag_mgr.mean_lambda():.4f}\n"
                f"{'-'*70}\n"
            )
            epoch_log_fh.flush()

        # periodic checkpoint
        if (epoch + 1) % 1000 == 0:
            save_checkpoint(
                os.path.join(run_path, f"epoch_{epoch+1:03d}.pt"),
                model, optimizer, scheduler, balancer, lag_mgr,
                epoch, best_val_metric, cfg)

    # ── final ──
    save_checkpoint(
        os.path.join(run_path, "last.pt"),
        model, optimizer, scheduler, balancer, lag_mgr,
        epoch, best_val_metric, cfg)
    epoch_log_fh.close()
    writer.close()
    logger.info(f"\nTraining complete.  Best val_{val_metric}: {best_val_metric:.4f}")
    logger.info(f"Logs & checkpoints: {run_path}")
    logger.info(f"Epoch metrics log: {epoch_log_path}")


if __name__ == "__main__":
    main()
