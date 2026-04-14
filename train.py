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
}


# ══════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════

class MILPDataset:
    """Lazy-loading dataset of processed .pkl instances."""

    def __init__(self, file_paths):
        self.files = sorted(file_paths)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            d = pickle.load(f)
        return {
            "var_feats":   torch.as_tensor(d["var_feats"],   dtype=torch.float32),
            "con_feats":   torch.as_tensor(d["con_feats"],   dtype=torch.float32),
            "edge_indices": torch.as_tensor(d["edge_indices"], dtype=torch.long),
            "edge_values": torch.as_tensor(d["edge_values"], dtype=torch.float32),
            "sols":        torch.as_tensor(d["sols"],        dtype=torch.float32),
            "objs":        torch.as_tensor(d["objs"],        dtype=torch.float32),
            "n_vars":      d["n_vars"],
            "n_cons":      d["n_cons"],
            "instance_id": os.path.basename(self.files[idx]),
        }


def collect_files(data_dir, problems, split):
    """Gather all .pkl paths for given problems and split."""
    files = []
    for prob in problems:
        d = os.path.join(data_dir, prob, split)
        if os.path.isdir(d):
            files.extend(
                os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pkl"))
    return sorted(files)


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

def train_one_epoch(model, dataset, balancer, lag_mgr,
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
            base = balancer(out, data, model)

            # Reuse pre-computed per-constraint violations from base loss
            viols = base["per_con_viol"]
            lag = lagrangian_loss(viols, lambdas, obj_norm=base["obj_loss"])

            # ── GradNorm weight update (once per batch, on first instance) ──
            if j == 0:
                balancer.step(base, model)

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
        # FIX: prevent optimizer from double-updating GradNorm weights
        if hasattr(balancer, 'log_w') and balancer.log_w.grad is not None:
            balancer.log_w.grad = None

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        batch_end = batch_start + actual_bs
        if batch_end % max(n // 5, 1) == 0 or batch_end == n:
            elapsed = time.time() - t0
            avg_total = meters["total"] / batch_end
            logging.getLogger("milp_train").info(
                f"  [{batch_end:>4d}/{n}]  total={avg_total:.4f}  "
                f"sup={meters['sup']/batch_end:.4f}  "
                f"lag={meters['lag']/batch_end:.4f}  "
                f"viol={meters['viol_mean']/batch_end:.5f}  {elapsed:.0f}s")

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
        if is_bin.any():
            best_sol = sols[objs.argmin()]
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
        obj_best = objs.min()
        obj_range = (objs.max() - objs.min()).clamp(min=1e-6)
        raw_gap = ((obj_pred - obj_best) / obj_range).item()
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
        n_probes=cfg["n_probes"],
        dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")  # before logger

    # ── loss ──
    base_loss = MILPCompositeLoss(hidden_dim=cfg["hidden_dim"]).to(device)
    balancer  = GradNormBalancer(base_loss, alpha=cfg["gradnorm_alpha"]).to(device)

    # ── Lagrangian ──
    lag_mgr = LagrangianManager(lr=cfg["lagrangian_lr"])

    # ── optimiser + scheduler (warmup + cosine) ──
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(balancer.parameters()),
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
    csv_log = CSVLogger(os.path.join(run_path, "training_log.csv"))
    logger = setup_logger(run_path)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    logger.info(f"Run dir: {run_path}")

    # ── resume ──
    start_epoch = 0
    best_val_loss = float("inf")
    if cfg["resume"]:
        start_epoch, best_val_loss, _ = load_checkpoint(
            cfg["resume"], model, optimizer, scheduler, balancer, lag_mgr)
        logger.info(f"Resumed from epoch {start_epoch}, best_val={best_val_loss:.4f}")

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
        w = balancer.weight_dict()
        for k, v in w.items():
            writer.add_scalar(f"gradnorm/{k}", v, epoch)
        writer.add_scalar("lagrangian/mean_lambda", lag_mgr.mean_lambda(), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - t0
        logger.info(f"\n  Train  total={train_m['total']:.4f}  sup={train_m['sup']:.4f}  "
              f"lag={train_m['lag']:.4f}  viol={train_m['viol_mean']:.5f}  "
              f"[{elapsed:.0f}s]")
        logger.info(f"  GradNorm weights: {w}")

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

            # ── CSV logging: one row per epoch with all metrics ──
            csv_row = {
                "epoch": epoch + 1,
                "lr": cur_lr,
                "elapsed_s": round(elapsed, 1),
                # train losses
                "train_total": round(train_m["total"], 6),
                "train_sup": round(train_m["sup"], 6),
                "train_viol": round(train_m.get("viol", 0), 6),
                "train_orth": round(train_m.get("orth", 0), 6),
                "train_recon": round(train_m.get("recon", 0), 6),
                "train_entropy": round(train_m.get("entropy", 0), 6),
                "train_lag": round(train_m["lag"], 6),
                "train_viol_mean": round(train_m["viol_mean"], 6),
                # val metrics
                "val_total": round(val_total, 6),
                "val_sup": round(val_m.get("loss/sup", 0), 6),
                "val_viol": round(val_m.get("loss/viol", 0), 6),
                "val_orth": round(val_m.get("loss/orth", 0), 6),
                "val_recon": round(val_m.get("loss/recon", 0), 6),
                "val_entropy": round(val_m.get("loss/entropy", 0), 6),
                "val_binary_acc": round(val_m.get("binary_acc", 0), 6),
                "val_feasible_rate": round(val_m.get("feasible_rate", 0), 6),
                "val_viol_rate": round(val_m.get("viol_rate", 0), 6),
                "val_viol_mag": round(val_m.get("viol_mag", 0), 6),
                "val_obj_gap": round(val_m.get("obj_gap", 0), 6),
                # GradNorm weights
                **{f"gn_w_{k}": round(v, 6) for k, v in w.items()},
                # Lagrangian
                "mean_lambda": round(lag_mgr.mean_lambda(), 6),
                "best_val_loss": round(best_val_loss, 6),
                "no_improve": no_improve,
            }
            csv_log.log(csv_row)

            # ---- early stopping / best model ----
            if val_sup < best_val_loss:
                best_val_loss = val_sup
                save_checkpoint(
                    os.path.join(run_path, "best.pt"),
                    model, optimizer, scheduler, balancer, lag_mgr,
                    epoch, best_val_loss, cfg)
                logger.info(f"  >> New best model saved (val_loss={best_val_loss:.4f})")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    logger.info(f"\n  Early stopping after {cfg['patience']} epochs "
                          f"without improvement.")
                    break
        else:
            # No validation set — still log training metrics to CSV
            csv_row = {
                "epoch": epoch + 1,
                "lr": cur_lr,
                "elapsed_s": round(elapsed, 1),
                "train_total": round(train_m["total"], 6),
                "train_sup": round(train_m["sup"], 6),
                "train_viol": round(train_m.get("viol", 0), 6),
                "train_orth": round(train_m.get("orth", 0), 6),
                "train_recon": round(train_m.get("recon", 0), 6),
                "train_entropy": round(train_m.get("entropy", 0), 6),
                "train_lag": round(train_m["lag"], 6),
                "train_viol_mean": round(train_m["viol_mean"], 6),
                **{f"gn_w_{k}": round(v, 6) for k, v in w.items()},
                "mean_lambda": round(lag_mgr.mean_lambda(), 6),
            }
            csv_log.log(csv_row)

        # periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                os.path.join(run_path, f"epoch_{epoch+1:03d}.pt"),
                model, optimizer, scheduler, balancer, lag_mgr,
                epoch, best_val_loss, cfg)

    # ── final ──
    save_checkpoint(
        os.path.join(run_path, "last.pt"),
        model, optimizer, scheduler, balancer, lag_mgr,
        epoch, best_val_loss, cfg)
    csv_log.close()
    writer.close()
    logger.info(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    logger.info(f"Logs & checkpoints: {run_path}")
    logger.info(f"CSV log: {os.path.join(run_path, 'training_log.csv')}")


if __name__ == "__main__":
    main()
