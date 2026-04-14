#!/usr/bin/env python3
"""
inference.py — Trust-region MILP instance export (inference-only)
==================================================================

Uses the trained GNN model to predict variable assignments, then
adds **trust-region constraints** to each instance and exports the
modified problem as a standalone .mps file for downstream solving.

Trust-region formulation
------------------------
Binary variables (confidence-weighted local branching):
    sum_{i in S1} w_i*(1-x_i) + sum_{i in S0} w_i*x_i  <=  delta
    where w_i = clip(|P(x_i=1) - 0.5| * 2, min_weight, 1.0)

Integer variables (L1 linearisation):
    d_i >= x_i - x_hat_i,  d_i >= x_hat_i - x_i,  d_i >= 0
    sum w_i*d_i <= delta

Usage
-----
    python inference.py --checkpoint runs/run_001/best.pt --problems CA IP
    python inference.py --checkpoint best.pt --delta 40 --no_confidence_weight
    python inference.py --checkpoint best.pt --conf_threshold 0.9
    python inference.py --checkpoint best.pt --fix_vars --fix_threshold 0.97
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np
import torch
import pyscipopt as scip_io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import set_seed
from model import MILPGNNModel

DEFAULT_SEED = 42


# ══════════════════════════════════════════════════════════════════════════
#  Feature extraction (mirrors build_dataset.py, no solution required)
# ══════════════════════════════════════════════════════════════════════════

def extract_features(instance_path: str):
    """
    Extract bipartite-graph features from a raw MILP instance file.
    Returns a dict compatible with the .pkl format used in training.
    """
    import ecole

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        scip_params={
            "presolving/maxrounds": 0,
            "presolving/maxrestarts": 0,
        },
    )
    env.seed(DEFAULT_SEED)
    obs, _, _, _, _ = env.reset(instance_path)

    var_feats_ecole = obs.variable_features.astype(np.float32)   # [V, 19]
    con_feats_ecole = obs.row_features.astype(np.float32)        # [C, 5]
    edge_indices    = obs.edge_features.indices.astype(np.int64) # [2, E]
    edge_values     = obs.edge_features.values.astype(np.float32)# [E]

    n_vars = var_feats_ecole.shape[0]
    n_cons = con_feats_ecole.shape[0]

    # Raw objective coefficients and variable names via pyscipopt
    scip_model = scip_io.Model()
    scip_model.hideOutput(True)
    scip_model.readProblem(instance_path)

    scip_vars = scip_model.getVars()
    raw_obj    = np.zeros(n_vars, dtype=np.float32)
    var_names  = []
    binary_vars = np.zeros(n_vars, dtype=np.int32)
    for i, v in enumerate(scip_vars):
        var_names.append(v.name)
        raw_obj[i] = v.getObj()
        if v.vtype() == "BINARY":
            binary_vars[i] = 1
    scip_model.freeProb()

    # Degrees
    var_degree = np.zeros(n_vars, dtype=np.float32)
    con_degree = np.zeros(n_cons, dtype=np.float32)
    np.add.at(var_degree, edge_indices[1], 1)
    np.add.at(con_degree, edge_indices[0], 1)

    # Assemble: var_feats 21 cols, con_feats 6 cols
    var_feats = np.column_stack([
        var_feats_ecole,
        var_degree.reshape(-1, 1),
        raw_obj.reshape(-1, 1),
    ])
    con_feats = np.column_stack([
        con_feats_ecole,
        con_degree.reshape(-1, 1),
    ])
    np.nan_to_num(var_feats, copy=False, nan=0.0)
    np.nan_to_num(con_feats, copy=False, nan=0.0)

    return {
        "var_feats":    var_feats,
        "con_feats":    con_feats,
        "edge_indices": edge_indices,
        "edge_values":  edge_values,
        "var_names":    var_names,
        "binary_vars":  binary_vars,
        "n_vars":       n_vars,
        "n_cons":       n_cons,
    }


def load_or_extract_features(inst_file: str, cache_path: str):
    """
    Return feature dict for *inst_file*.
    If *cache_path* (.pkl) already exists, load from it.
    Otherwise extract features, save to *cache_path*, then return.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f), True   # (data, from_cache)

    data = extract_features(inst_file)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data, False

# ── Default delta per problem (from baseline) ─────────────────────────────
DEFAULT_DELTA = {
    "CA": 40, "IP": 55, "IS": 40, "SC": 200, "WA": 100,
}

INSTANCE_DIR = "/home/lmh/autodl-tmp/data/l2o_milp"


# ══════════════════════════════════════════════════════════════════════════
#  Model loading & inference
# ══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = MILPGNNModel(
        hidden_dim=cfg.get("hidden_dim", 64),
        n_gcn_layers=cfg.get("n_gcn_layers", 2),
        n_probes=cfg.get("n_probes", 16),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════
#  Trust-region constraint builder (via pyscipopt)
# ══════════════════════════════════════════════════════════════════════════

def build_trust_region_instance(inst_file, pred_info, delta,
                                use_confidence_weight=True,
                                min_weight=0.05,
                                conf_threshold=0.9,
                                fix_vars=False,
                                fix_threshold=0.97):
    """
    Load the original MILP from inst_file, add trust-region constraints
    based on pred_info, and return the modified pyscipopt Model.

    Parameters
    ----------
    inst_file             : str   path to original .lp / .mps file
    pred_info             : dict  with keys binary_prob, integer_mu,
                                  var_names, var_feats
    delta                 : float trust-region radius
    use_confidence_weight : bool  anisotropic (True) or uniform (False) TR
    min_weight            : float minimum confidence weight
    conf_threshold        : float only include a variable in the trust region
                                  if its confidence >= this value (default 0.9).
                                  Binary confidence = |P(x=1) - 0.5| * 2 in [0,1].
                                  Integer confidence = 1 - 2*frac in [0,1].
    fix_vars              : bool  if True, instead of a trust-region constraint,
                                  variables whose confidence >= fix_threshold are
                                  fixed to their predicted value (equality bound).
                                  Variables below fix_threshold are excluded entirely.
    fix_threshold         : float confidence threshold used when fix_vars=True
                                  (default 0.97).

    Returns
    -------
    model   : pyscipopt.Model  (modified, not yet solved)
    n_bin   : int  number of binary variables added to trust region / fixed
    n_int   : int  number of integer variables added to trust region / fixed
    """
    var_feats   = pred_info["var_feats"]
    var_names   = pred_info["var_names"]
    binary_prob = pred_info["binary_prob"]
    integer_mu  = pred_info["integer_mu"]

    # Classify variables by type flags (columns match build_dataset.py)
    is_bin = var_feats[:, 1].astype(bool)
    is_int = var_feats[:, 2].astype(bool) | var_feats[:, 3].astype(bool)
    bin_indices = np.where(is_bin)[0]
    int_indices = np.where(is_int)[0]

    # Load original problem
    model = scip_io.Model()
    model.hideOutput(True)
    model.readProblem(inst_file)

    scip_vars = model.getVars()
    scip_vars.sort(key=lambda v: v.name)
    var_map = {v.name: v for v in scip_vars}

    # Select the active threshold depending on mode
    active_threshold = fix_threshold if fix_vars else conf_threshold

    # ── Binary variables ──────────────────────────────────────────────────
    n_bin_active = 0
    if len(bin_indices) > 0:
        probs     = binary_prob[bin_indices]
        x_hat_bin = (probs > 0.5).astype(float)

        # Binary confidence: distance from 0.5, scaled to [0, 1]
        conf_bin = np.abs(probs - 0.5) * 2.0   # [0, 1]

        if fix_vars:
            # Fix variables whose confidence >= fix_threshold
            for j, idx in enumerate(bin_indices):
                if conf_bin[j] < active_threshold:
                    continue
                name = var_names[idx]
                if name not in var_map:
                    continue
                gv = var_map[name]
                val = float(x_hat_bin[j])
                model.chgVarLb(gv, val)
                model.chgVarUb(gv, val)
                n_bin_active += 1
        else:
            # Trust-region: only include variables above conf_threshold
            if use_confidence_weight:
                conf_w = np.clip(conf_bin, min_weight, 1.0)
            else:
                conf_w = np.ones(len(bin_indices))

            terms = []
            for j, idx in enumerate(bin_indices):
                if conf_bin[j] < active_threshold:
                    continue
                name = var_names[idx]
                if name not in var_map:
                    continue
                gv = var_map[name]
                w  = float(conf_w[j])
                if x_hat_bin[j] == 1.0:
                    terms.append(w * (1.0 - gv))
                else:
                    terms.append(w * gv)
                n_bin_active += 1

            if terms:
                model.addCons(
                    scip_io.quicksum(terms) <= delta,
                    name="trust_binary"
                )

    # ── Integer variables ─────────────────────────────────────────────────
    n_int_active = 0
    if len(int_indices) > 0:
        x_hat_int = np.round(integer_mu[int_indices]).astype(float)
        frac      = np.abs(integer_mu[int_indices] - x_hat_int)
        conf_int  = 1.0 - np.clip(frac, 0.0, 0.5) * 2.0   # [0, 1]

        if fix_vars:
            for j, idx in enumerate(int_indices):
                if conf_int[j] < active_threshold:
                    continue
                name = var_names[idx]
                if name not in var_map:
                    continue
                gv  = var_map[name]
                val = float(x_hat_int[j])
                model.chgVarLb(gv, val)
                model.chgVarUb(gv, val)
                n_int_active += 1
        else:
            if use_confidence_weight:
                w_int_arr = np.clip(conf_int, min_weight, 1.0)
            else:
                w_int_arr = np.ones(len(int_indices))

            aux_vars = []
            weights  = []
            for j, idx in enumerate(int_indices):
                if conf_int[j] < active_threshold:
                    continue
                name = var_names[idx]
                if name not in var_map:
                    continue
                gv    = var_map[name]
                x_hat = float(x_hat_int[j])
                w     = float(w_int_arr[j])

                d = model.addVar(name=f"_tr_d_{name}", vtype="C", lb=0.0)
                model.addCons(d >= gv - x_hat, name=f"_tr_up_{name}")
                model.addCons(d >= x_hat - gv, name=f"_tr_dn_{name}")
                aux_vars.append(d)
                weights.append(w)
                n_int_active += 1

            if aux_vars:
                model.addCons(
                    scip_io.quicksum(w * d for w, d in zip(weights, aux_vars)) <= delta,
                    name="trust_integer"
                )

    return model, n_bin_active, n_int_active


# ══════════════════════════════════════════════════════════════════════════
#  Per-problem export loop
# ══════════════════════════════════════════════════════════════════════════

def _iter_test_instances(inst_dir: str, cache_dir: str):
    """
    Yield (stem, inst_file, cache_pkl_path) for every raw instance in
    *inst_dir*.  Used when split == "test" (no pre-built pkl files exist).
    """
    if not os.path.isdir(inst_dir):
        return
    for fname in sorted(os.listdir(inst_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".lp", ".mps"):
            continue
        inst_file  = os.path.join(inst_dir, fname)
        cache_path = os.path.join(cache_dir, stem + ".pkl")
        yield stem, inst_file, cache_path


def _iter_split_instances(split_dir: str, inst_dir: str, cache_dir: str):
    """
    Yield (stem, inst_file, cache_pkl_path) for every pkl in *split_dir*.
    Used for train / val splits where pkl files were pre-built.
    If a pkl already exists in *cache_dir*, that path is returned;
    otherwise the original split_dir pkl is used (no re-extraction needed).
    """
    if not os.path.isdir(split_dir):
        return
    for pkl_name in sorted(f for f in os.listdir(split_dir) if f.endswith(".pkl")):
        stem = pkl_name[:-4]
        inst_file = None
        for ext in (".lp", ".mps"):
            cand = os.path.join(inst_dir, stem + ext)
            if os.path.exists(cand):
                inst_file = cand
                break
        # cache path in split_dir (pre-built); no re-extraction is needed
        cache_path = os.path.join(split_dir, pkl_name)
        yield stem, inst_file, cache_path


def export_problem(gnn_model, problem, args, device):
    """
    For each instance of `problem`:
      1. Load feature .pkl (extract & cache if missing) -> tensor inputs
      2. Run GNN inference -> pred_info
      3. Add trust-region constraints to the original MILP
      4. Export modified instance as <stem>_modified.mps

    When split == "test", instances are taken directly from the raw instance
    directory; features are extracted on-the-fly and cached as .pkl files
    under processed/<problem>/test/.
    """
    inst_dir  = os.path.join(INSTANCE_DIR, problem)
    out_dir   = os.path.join(args.output_dir, problem, "modified_instances")
    os.makedirs(out_dir, exist_ok=True)

    delta = args.delta if args.delta > 0 else DEFAULT_DELTA.get(problem, 50)

    if args.split == "test":
        cache_dir = os.path.join(SCRIPT_DIR, "processed", problem, "test")
        os.makedirs(cache_dir, exist_ok=True)
        instances = list(_iter_test_instances(inst_dir, cache_dir))
        source_label = f"raw instances in {inst_dir}"
    else:
        split_dir = os.path.join(SCRIPT_DIR, "processed", problem, args.split)
        cache_dir = split_dir          # pre-built pkls live here already
        instances = list(_iter_split_instances(split_dir, inst_dir, cache_dir))
        source_label = split_dir

    if not instances:
        print(f"  [SKIP] no instances found for {problem}/{args.split}")
        return

    print(f"\n{'='*60}")
    print(f"Problem: {problem}  |  split: {args.split}  |  "
          f"instances: {len(instances)}  |  delta={delta}")
    print(f"Source : {source_label}")
    print(f"Cache  : {cache_dir}")
    print(f"Output : {out_dir}")
    print(f"{'='*60}")

    sum_feat = sum_infer = sum_total = 0.0
    sum_bin = sum_int = 0
    count = 0

    for i, (stem, inst_file, cache_path) in enumerate(instances):
        if inst_file is None:
            print(f"  [{i+1}] {stem}: original instance not found, skipping")
            continue

        t0 = time.time()

        # ---- 1. Feature extraction (from cache or raw instance) ----
        if args.split == "test":
            data, from_cache = load_or_extract_features(inst_file, cache_path)
            cache_tag = "cache" if from_cache else "extracted"
        else:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            cache_tag = "pkl"

        vf = torch.tensor(data["var_feats"],    dtype=torch.float32, device=device)
        cf = torch.tensor(data["con_feats"],    dtype=torch.float32, device=device)
        ei = torch.tensor(data["edge_indices"], dtype=torch.long,    device=device)
        ev = torch.tensor(data["edge_values"],  dtype=torch.float32, device=device)
        t1 = time.time()
        feat_time = t1 - t0

        # ---- 2. GNN inference ----
        with torch.no_grad():
            out = gnn_model(vf, cf, ei, ev)
        pred_info = {
            "binary_prob": torch.sigmoid(out["binary_logits"]).cpu().numpy(),
            "integer_mu":  out["integer_mu"].cpu().numpy(),
            "var_names":   data["var_names"],
            "var_feats":   data["var_feats"],
            "n_vars":      data["n_vars"],
            "n_cons":      data["n_cons"],
        }
        t2 = time.time()
        infer_time = t2 - t1

        # ---- 3. Add trust-region constraints & export ----
        model, n_bin, n_int = build_trust_region_instance(
            inst_file, pred_info, delta,
            use_confidence_weight=not args.no_confidence_weight,
            conf_threshold=args.conf_threshold,
            fix_vars=args.fix_vars,
            fix_threshold=args.fix_threshold,
        )
        out_path = os.path.join(out_dir, stem + "_modified.mps")
        model.writeProblem(out_path)
        model.freeProb()

        total_time = time.time() - t0
        sum_feat  += feat_time
        sum_infer += infer_time
        sum_total += total_time
        sum_bin   += n_bin
        sum_int   += n_int
        count += 1

        print(f"  [{i+1:>3d}/{len(instances)}] {stem}  "
              f"[{cache_tag}]  "
              f"n_bin={n_bin}  n_int={n_int}  delta={delta}  "
              f"feat={feat_time:.3f}s  infer={infer_time:.3f}s  "
              f"total={total_time:.3f}s  -> {os.path.basename(out_path)}")

    if count > 0:
        thr = args.fix_threshold if args.fix_vars else args.conf_threshold
        mode_label = f"fix_threshold={thr}" if args.fix_vars else f"conf_threshold={thr}"
        print(f"\n  Summary ({count} instances):")
        print(f"    Confidence filter  : {mode_label}")
        print(f"    Vars in threshold  : "
              f"avg_bin={sum_bin/count:.1f}  avg_int={sum_int/count:.1f}  "
              f"avg_total={(sum_bin+sum_int)/count:.1f}")
        print(f"    Feature extraction : total={sum_feat:.3f}s  "
              f"avg={sum_feat/count:.3f}s")
        print(f"    GNN inference      : total={sum_infer:.3f}s  "
              f"avg={sum_infer/count:.3f}s")
        print(f"    Overall            : total={sum_total:.3f}s  "
              f"avg={sum_total/count:.3f}s")
        print(f"    Output directory   : {out_dir}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNN inference -> trust-region MILP instance export")
    parser.add_argument("--checkpoint", required=True,
                        help="Model checkpoint path (.pt)")
    parser.add_argument("--problems", nargs="+",
                        default=["CA", "IP", "IS", "SC", "WA"],
                        help="Problem types to process")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to use (test = raw instances, features auto-cached)")
    parser.add_argument("--delta", type=int, default=-1,
                        help="Trust-region radius (-1 = per-problem default)")
    parser.add_argument("--no_confidence_weight", action="store_true",
                        help="Use uniform weights instead of confidence-based")
    parser.add_argument("--conf_threshold", type=float, default=0.9,
                        help="Minimum confidence for a variable to enter the trust "
                             "region (default 0.9).  Binary: |P(x=1)-0.5|*2; "
                             "Integer: 1 - 2*frac.")
    parser.add_argument("--fix_vars", action="store_true",
                        help="Instead of a trust-region constraint, fix high-confidence "
                             "variables to their predicted value (equality bound). "
                             "Uses --fix_threshold to decide which vars to fix.")
    parser.add_argument("--fix_threshold", type=float, default=0.97,
                        help="Confidence threshold for --fix_vars mode (default 0.97).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir",
                        default=os.path.join(SCRIPT_DIR, "output"),
                        help="Root directory for exported .mps files")
    args = parser.parse_args()

    # Derive run folder name from checkpoint path and append to output_dir.
    # Expected layout: .../runs/run_XXX/best.pt  -> run_XXX
    # Walk up the checkpoint path to find a component that starts with "run_".
    ckpt_parts = os.path.normpath(os.path.abspath(args.checkpoint)).split(os.sep)
    run_name = next((p for p in reversed(ckpt_parts) if p.startswith("run_")), None)
    if run_name:
        args.output_dir = os.path.join(args.output_dir, run_name)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")
    print(f"  hidden_dim={cfg.get('hidden_dim', 64)}  "
          f"n_gcn_layers={cfg.get('n_gcn_layers', 2)}  "
          f"n_probes={cfg.get('n_probes', 16)}")

    os.makedirs(args.output_dir, exist_ok=True)

    for prob in args.problems:
        export_problem(model, prob, args, device)

    print(f"\nDone. Modified instances written to {args.output_dir}/")


if __name__ == "__main__":
    main()
