#!/usr/bin/env python3
"""
inference.py — Trust-region guided MILP solving
================================================

Uses the trained GNN model to predict variable assignments, then
converts those predictions into **trust-region constraints** for Gurobi:

1. **Binary variables (Local Branching)**:
     Σ_{i∈S₁}(1-xᵢ) + Σ_{i∈S₀}xᵢ  ≤  Δ_bin
   with confidence-weighted variant:
     Σ wᵢ·dᵢ ≤ Δ_bin   where wᵢ = |P(xᵢ=1) − 0.5|

2. **Integer variables (L1 linearisation)**:
     xᵢ − x̂ᵢ ≤ dᵢ ,   x̂ᵢ − xᵢ ≤ dᵢ ,   dᵢ ≥ 0
     Σ wᵢ·dᵢ ≤ Δ_int

3. **Confidence weighting (Anisotropic trust region)**:
     wᵢ = |P(xᵢ) − 0.5|  ∈ [0, 0.5]
   High confidence → tight constraint (hard to change).
   Low confidence  → loose (solver explores freely).

Usage
-----
    # Evaluate on test instances:
    python inference.py --problem CA --checkpoint runs/run_001/best.pt

    # Custom trust region size:
    python inference.py --problem CA --checkpoint best.pt --delta 40

    # Without confidence weighting (uniform trust region):
    python inference.py --problem CA --checkpoint best.pt --no_confidence_weight
"""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import set_seed
from model import MILPGNNModel

# ── Default Δ per problem (from baseline) ────────────────────────────────
DEFAULT_DELTA = {
    "CA": 40, "IP": 55, "IS": 40, "SC": 200, "WA": 100,
}

INSTANCE_DIR = "/home/lmh/autodl-tmp/data/l2o_milp"
SOLUTION_DIR = "/home/lmh/autodl-tmp/baseline/dataset"


# ══════════════════════════════════════════════════════════════════════════
#  Model inference
# ══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = MILPGNNModel(
        hidden_dim=cfg.get("hidden_dim", 64),
        n_probes=cfg.get("n_probes", 16),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def predict_instance(model, instance_pkl_path, device):
    """
    Run model on one processed instance.
    Returns: predictions dict, raw data dict
    """
    with open(instance_pkl_path, "rb") as f:
        data = pickle.load(f)

    vf = torch.tensor(data["var_feats"],   dtype=torch.float32, device=device)
    cf = torch.tensor(data["con_feats"],   dtype=torch.float32, device=device)
    ei = torch.tensor(data["edge_indices"], dtype=torch.long,   device=device)
    ev = torch.tensor(data["edge_values"], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(vf, cf, ei, ev)

    return {
        "pred":          out["pred"].cpu().numpy(),
        "binary_prob":   torch.sigmoid(out["binary_logits"]).cpu().numpy(),
        "integer_mu":    out["integer_mu"].cpu().numpy(),
        "var_names":     data["var_names"],
        "var_feats":     data["var_feats"],
        "n_vars":        data["n_vars"],
        "n_cons":        data["n_cons"],
    }, data


# ══════════════════════════════════════════════════════════════════════════
#  Trust region construction
# ══════════════════════════════════════════════════════════════════════════

def add_trust_region(grb_model, pred_info, delta,
                     use_confidence_weight=True,
                     min_weight=0.05):
    """
    Add trust-region constraints to a Gurobi model.

    For binary variables: confidence-weighted local branching.
    For integer variables: L1-linearised trust region with auxiliary vars.

    Parameters
    ----------
    grb_model            : gurobipy.Model  (already loaded with the instance)
    pred_info            : dict from predict_instance()
    delta                : float  trust region radius
    use_confidence_weight: bool   if True, use anisotropic (confidence-weighted) TR
    min_weight           : float  minimum weight to avoid zero weights
    """
    var_feats = pred_info["var_feats"]
    var_names = pred_info["var_names"]
    binary_prob = pred_info["binary_prob"]
    integer_mu  = pred_info["integer_mu"]
    n_vars = pred_info["n_vars"]

    # Build name → Gurobi variable map
    grb_vars = grb_model.getVars()
    var_map = {v.VarName: v for v in grb_vars}

    # Classify variables
    is_bin  = var_feats[:, 1].astype(bool)
    is_int  = (var_feats[:, 2].astype(bool)) | (var_feats[:, 3].astype(bool))

    bin_indices = np.where(is_bin)[0]
    int_indices = np.where(is_int)[0]

    grb_model.update()
    aux_vars = []
    weights  = []

    # ── Binary variables: Local Branching ──
    # Σ_{i∈S₁}(1−xᵢ) + Σ_{i∈S₀}xᵢ  ≤  Δ  (or confidence-weighted)
    if len(bin_indices) > 0:
        # Predicted assignment: round probability
        x_hat_bin = (binary_prob[bin_indices] > 0.5).astype(float)
        probs = binary_prob[bin_indices]

        # Confidence weight: |p − 0.5| ∈ [0, 0.5]
        # Normalise to [min_weight, 1.0]
        conf = np.abs(probs - 0.5)               # [0, 0.5]
        if use_confidence_weight:
            conf_w = conf * 2.0                    # [0, 1.0]
            conf_w = np.clip(conf_w, min_weight, 1.0)
        else:
            conf_w = np.ones_like(conf)

        # Build the local branching constraint using a LinExpr
        lb_expr = gp.LinExpr()
        for j, idx in enumerate(bin_indices):
            name = var_names[idx]
            if name not in var_map:
                continue
            gv = var_map[name]
            w = float(conf_w[j])

            if x_hat_bin[j] == 1.0:
                # predicted 1 → penalty for flipping to 0: w*(1 - x)
                lb_expr += w * (1.0 - gv)
            else:
                # predicted 0 → penalty for flipping to 1: w*x
                lb_expr += w * gv

        grb_model.addConstr(lb_expr <= delta, name="trust_binary")

    # ── Integer variables: L1 linearisation ──
    # For each integer var: d_i ≥ |x_i − x̂_i|  via two linear constraints
    # Global: Σ w_i · d_i ≤ Δ_int
    if len(int_indices) > 0:
        x_hat_int = np.round(integer_mu[int_indices]).astype(float)
        probs_int = np.abs(integer_mu[int_indices] -
                           np.round(integer_mu[int_indices]))
        # confidence = 1 - fractional part (closer to integer → more confident)
        conf_int = 1.0 - np.clip(probs_int, 0, 0.5) * 2
        if use_confidence_weight:
            w_int = np.clip(conf_int, min_weight, 1.0)
        else:
            w_int = np.ones(len(int_indices))

        int_expr = gp.LinExpr()
        for j, idx in enumerate(int_indices):
            name = var_names[idx]
            if name not in var_map:
                continue
            gv = var_map[name]
            x_hat = float(x_hat_int[j])
            w = float(w_int[j])

            # Auxiliary variable d_i >= 0
            d = grb_model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                 name=f"_tr_d_{name}")
            # d_i >= x_i - x̂_i
            grb_model.addConstr(d >= gv - x_hat, name=f"_tr_up_{name}")
            # d_i >= x̂_i - x_i
            grb_model.addConstr(d >= x_hat - gv, name=f"_tr_dn_{name}")

            int_expr += w * d

        grb_model.addConstr(int_expr <= delta, name="trust_integer")

    grb_model.update()
    n_bin = len(bin_indices)
    n_int = len(int_indices)
    return n_bin, n_int


# ══════════════════════════════════════════════════════════════════════════
#  Gurobi solve
# ══════════════════════════════════════════════════════════════════════════

def solve_with_trust_region(instance_path, pred_info, delta,
                            time_limit=1000, threads=1,
                            use_confidence_weight=True,
                            log_path=None,
                            adaptive=True, max_expand=3):
    """
    Load the original MILP instance, add trust-region constraints from
    the GNN prediction, and solve with Gurobi.

    If adaptive=True and the trust region is infeasible, automatically
    doubles Δ and retries (up to max_expand times).

    Returns dict with objective, solve time, gap, solution vector, etc.
    """
    attempt = 0
    current_delta = delta

    while True:
        gp.setParam("LogToConsole", 0)
        model = gp.read(instance_path)
        model.Params.TimeLimit = time_limit
        model.Params.Threads = threads
        model.Params.MIPFocus = 0            # balanced (same as baseline)
        if log_path:
            model.Params.LogFile = log_path

        # Add trust region
        n_bin, n_int = add_trust_region(
            model, pred_info, current_delta,
            use_confidence_weight=use_confidence_weight)

        # Solve
        t0 = time.time()
        model.optimize()
        solve_time = time.time() - t0

        # Check feasibility — if infeasible and adaptive, expand Δ
        if (adaptive and model.Status == GRB.INFEASIBLE
                and attempt < max_expand):
            attempt += 1
            old_delta = current_delta
            current_delta = int(current_delta * 2)
            # Reduce remaining time budget
            time_limit = max(60, time_limit - int(solve_time))
            continue  # retry with larger Δ

        break  # feasible or exhausted retries

    result = {
        "status":      model.Status,
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, f"OTHER({model.Status})"),
        "solve_time":  solve_time,
        "n_bin_trust": n_bin,
        "n_int_trust": n_int,
        "delta":       current_delta,
        "delta_init":  delta,
        "n_expansions": attempt,
    }

    if model.SolCount > 0:
        result["obj_val"]  = model.ObjVal
        result["obj_bound"] = model.ObjBound
        result["mip_gap"]  = model.MIPGap
        result["n_nodes"]  = int(model.NodeCount)
        # Extract solution
        sol = np.array([v.X for v in model.getVars()
                        if not v.VarName.startswith("_tr_")])
        result["solution"] = sol
    else:
        result["obj_val"]  = float("inf")
        result["obj_bound"] = float("-inf")
        result["mip_gap"]  = float("inf")
        result["n_nodes"]  = 0
        result["solution"] = None

    return result


# ══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════

def evaluate_problem(model, problem, checkpoint_path, args, device):
    """Evaluate on all test/val instances of a problem."""
    data_dir = os.path.join(SCRIPT_DIR, "processed", problem)
    split = args.split
    split_dir = os.path.join(data_dir, split)

    if not os.path.isdir(split_dir):
        print(f"  [SKIP] {split_dir} not found")
        return []

    # Instance directory (original .lp/.mps files)
    inst_dir = os.path.join(INSTANCE_DIR, problem)

    pkl_files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pkl")])
    delta = args.delta if args.delta > 0 else DEFAULT_DELTA.get(problem, 50)

    print(f"\n{'='*60}")
    print(f"Problem: {problem}  |  split: {split}  |  "
          f"instances: {len(pkl_files)}  |  Δ={delta}")
    print(f"{'='*60}")

    out_dir = os.path.join(args.output_dir, problem)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, pkl_name in enumerate(pkl_files):
        stem = pkl_name.replace(".pkl", "")
        pkl_path = os.path.join(split_dir, pkl_name)

        # Find the original instance file
        inst_file = None
        for ext in (".lp", ".mps"):
            cand = os.path.join(inst_dir, stem + ext)
            if os.path.exists(cand):
                inst_file = cand
                break
        if inst_file is None:
            print(f"  [{i+1}] {stem}: original instance not found, skipping")
            continue

        # Predict
        pred_info, raw_data = predict_instance(model, pkl_path, device)

        # Load Gurobi baseline objective for comparison
        sol_path = os.path.join(SOLUTION_DIR, problem, "solution", f"{stem}.sol")
        baseline_obj = None
        if os.path.exists(sol_path):
            with open(sol_path, "rb") as f:
                sol_data = pickle.load(f)
            baseline_obj = float(sol_data["objs"].min())

        # Solve
        log_path = os.path.join(out_dir, f"{stem}.log")
        res = solve_with_trust_region(
            inst_file, pred_info, delta,
            time_limit=args.time_limit,
            threads=args.threads,
            use_confidence_weight=not args.no_confidence_weight,
            log_path=log_path)

        res["instance"] = stem
        res["baseline_obj"] = baseline_obj

        # Compute improvement
        if baseline_obj is not None and res["obj_val"] < float("inf"):
            gap = (res["obj_val"] - baseline_obj) / (abs(baseline_obj) + 1e-8)
            res["vs_baseline"] = gap
        else:
            res["vs_baseline"] = None

        results.append(res)

        status = res["status_name"]
        obj = res["obj_val"] if res["obj_val"] < float("inf") else "N/A"
        vs = f"{res['vs_baseline']:+.4f}" if res["vs_baseline"] is not None else "N/A"

        print(f"  [{i+1:>3d}/{len(pkl_files)}] {stem}  "
              f"status={status:<10s}  obj={obj:<12}  "
              f"vs_baseline={vs}  "
              f"time={res['solve_time']:.1f}s  nodes={res['n_nodes']}")

    # Summary
    if results:
        feasible = [r for r in results if r["obj_val"] < float("inf")]
        print(f"\n  Summary: {len(feasible)}/{len(results)} feasible")
        if feasible:
            objs = [r["obj_val"] for r in feasible]
            times = [r["solve_time"] for r in feasible]
            vs_bl = [r["vs_baseline"] for r in feasible
                     if r["vs_baseline"] is not None]
            print(f"    Obj:  mean={np.mean(objs):.2f}  "
                  f"best={np.min(objs):.2f}")
            print(f"    Time: mean={np.mean(times):.1f}s  "
                  f"max={np.max(times):.1f}s")
            if vs_bl:
                print(f"    vs Baseline: mean={np.mean(vs_bl):+.4f}  "
                      f"best={np.min(vs_bl):+.4f}")

    # Save results
    save_path = os.path.join(out_dir, "results.json")
    serialisable = []
    for r in results:
        r2 = {k: v for k, v in r.items() if k != "solution"}
        r2 = {k: (v if not isinstance(v, float) or np.isfinite(v)
                   else str(v)) for k, v in r2.items()}
        serialisable.append(r2)
    with open(save_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Results saved to {save_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MILP GNN Trust-Region Inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--problems", nargs="+",
                        default=["CA", "IP", "IS", "SC", "WA"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--delta", type=int, default=-1,
                        help="Trust region radius (-1 = use per-problem default)")
    parser.add_argument("--time_limit", type=int, default=1000,
                        help="Gurobi time limit in seconds")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no_confidence_weight", action="store_true",
                        help="Disable confidence-based weighting (uniform TR)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir",
                        default=os.path.join(SCRIPT_DIR, "inference_results"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")
    print(f"  hidden_dim={cfg.get('hidden_dim', 64)}  "
          f"n_probes={cfg.get('n_probes', 16)}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for prob in args.problems:
        all_results[prob] = evaluate_problem(
            model, prob, args.checkpoint, args, device)

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
