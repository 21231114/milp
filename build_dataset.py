#!/usr/bin/env python3
"""
build_dataset.py
================
Extract bipartite-graph features from MILP instances using ecole,
pair them with Gurobi baseline solutions, and split into train / val.

For each instance the script produces a dict saved as .pkl:

  var_feats        : np.float32  [n_vars, 21]   — 19 ecole + degree + raw_lb + raw_ub  (but we keep ecole 19 + 2 static = 21 total)
  con_feats        : np.float32  [n_cons, 6]    — 5 ecole + degree
  edge_indices     : np.int64    [2, nnz]       — (constraint_idx, variable_idx)
  edge_values      : np.float32  [nnz]          — coefficient (edge weight)
  var_names        : list[str]                  — variable names (original order)
  binary_vars      : np.int32    [n_vars]       — 1 if binary, else 0
  sols             : np.float32  [K, n_vars]    — up to 50 rounded Gurobi solutions
  objs             : np.float32  [K]            — objective values
  n_vars / n_cons  : int

Variable feature columns (21):
  0  objective               (ecole, normalised)
  1  is_type_binary
  2  is_type_integer
  3  is_type_implicit_integer
  4  is_type_continuous
  5  has_lower_bound
  6  has_upper_bound
  7  normed_reduced_cost     (dynamic-initial)
  8  solution_value          (dynamic-initial, LP relaxation)
  9  solution_frac           (dynamic-initial)
 10  is_solution_at_lb       (dynamic-initial)
 11  is_solution_at_ub       (dynamic-initial)
 12  scaled_age              (dynamic-initial)
 13  incumbent_value         (dynamic-initial)
 14  average_incumbent_value (dynamic-initial)
 15  is_basis_lower          (dynamic-initial)
 16  is_basis_basic          (dynamic-initial)
 17  is_basis_upper          (dynamic-initial)
 18  is_basis_zero           (dynamic-initial)
 19  degree                  (static, # constraints the variable appears in)
 20  raw_obj_coefficient     (static, un-normalised objective coeff from SCIP)

Constraint feature columns (6):
  0  bias                         (ecole, normalised RHS)
  1  objective_cosine_similarity
  2  is_tight                     (dynamic-initial)
  3  dual_solution_value          (dynamic-initial)
  4  scaled_age                   (dynamic-initial)
  5  degree                       (static, # variables in the constraint)

Usage:
    python build_dataset.py                           # process all 5 problems
    python build_dataset.py --problems CA IS          # only specific problems
    python build_dataset.py --seed 42                 # (default seed)
"""

import os
import sys
import argparse
import pickle
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths  (edit if your layout differs)
# ---------------------------------------------------------------------------
INSTANCE_DIR = "/home/lmh/autodl-tmp/data/l2o_milp"
SOLUTION_DIR = "/home/lmh/autodl-tmp/baseline/dataset"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed")

ALL_PROBLEMS = ["CA", "IP", "IS", "SC", "WA"]
TRAIN_COUNT = 240
VAL_COUNT = 60
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Lazy imports (so --help is fast even if libs are slow to load)
# ---------------------------------------------------------------------------
ecole = None
SCIPModel = None
torch = None


def _lazy_imports():
    global ecole, SCIPModel, torch
    if ecole is None:
        import ecole as _ecole
        from pyscipopt import Model as _SCIPModel
        import torch as _torch
        ecole = _ecole
        SCIPModel = _SCIPModel
        torch = _torch


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(instance_path: str):
    """
    Return (var_feats, con_feats, edge_indices, edge_values, var_names,
            binary_vars, raw_obj_coeffs) by combining ecole observations
    with raw SCIP model information.

    Presolving is disabled so the variable / constraint count matches the
    original formulation (and hence the Gurobi solution files).
    """
    _lazy_imports()

    # ---- ecole: initial LP relaxation observation --------------------------
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
    edge_indices = obs.edge_features.indices.astype(np.int64)    # [2, E]
    edge_values = obs.edge_features.values.astype(np.float32)    # [E]

    n_vars = var_feats_ecole.shape[0]
    n_cons = con_feats_ecole.shape[0]

    # ---- pyscipopt: raw bounds & objective --------------------------------
    model = SCIPModel()
    model.hideOutput()
    model.readProblem(instance_path)

    scip_vars = model.getVars()
    raw_obj = np.zeros(n_vars, dtype=np.float32)
    var_names = []
    binary_vars = np.zeros(n_vars, dtype=np.int32)
    for i, v in enumerate(scip_vars):
        var_names.append(v.name)
        raw_obj[i] = v.getObj()
        if v.vtype() == "BINARY":
            binary_vars[i] = 1

    # ---- degrees -----------------------------------------------------------
    var_degree = np.zeros(n_vars, dtype=np.float32)
    con_degree = np.zeros(n_cons, dtype=np.float32)
    np.add.at(var_degree, edge_indices[1], 1)
    np.add.at(con_degree, edge_indices[0], 1)

    # ---- assemble ----------------------------------------------------------
    # var_feats: 19 (ecole) + degree + raw_obj = 21
    var_feats = np.column_stack([
        var_feats_ecole,
        var_degree.reshape(-1, 1),
        raw_obj.reshape(-1, 1),
    ])

    # con_feats: 5 (ecole) + degree = 6
    con_feats = np.column_stack([
        con_feats_ecole,
        con_degree.reshape(-1, 1),
    ])

    # NaN → 0
    np.nan_to_num(var_feats, copy=False, nan=0.0)
    np.nan_to_num(con_feats, copy=False, nan=0.0)

    model.freeProb()

    return var_feats, con_feats, edge_indices, edge_values, var_names, binary_vars, raw_obj


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_instance(problem: str, inst_file: str, inst_path: str):
    """Process one MILP instance and return a result dict (or None)."""
    stem = os.path.splitext(inst_file)[0]
    sol_path = os.path.join(SOLUTION_DIR, problem, "solution", f"{stem}.sol")

    if not os.path.exists(sol_path):
        print(f"  [WARN] no solution for {stem}, skipping")
        return None

    # 1. Feature extraction
    (var_feats, con_feats, edge_indices, edge_values,
     var_names, binary_vars, raw_obj) = extract_features(inst_path)

    # 2. Load Gurobi solutions
    with open(sol_path, "rb") as f:
        sol_data = pickle.load(f)

    sols = np.round(sol_data["sols"][:50], 0).astype(np.float32)
    objs = sol_data["objs"][:50].astype(np.float32)

    # 3. Sanity check: variable count must match
    if var_feats.shape[0] != len(sol_data["var_names"]):
        print(f"  [WARN] var count mismatch for {stem}: "
              f"ecole={var_feats.shape[0]}, sol={len(sol_data['var_names'])}")
        return None

    return {
        "var_feats": var_feats,
        "con_feats": con_feats,
        "edge_indices": edge_indices,
        "edge_values": edge_values,
        "var_names": var_names,
        "binary_vars": binary_vars,
        "sols": sols,
        "objs": objs,
        "n_vars": var_feats.shape[0],
        "n_cons": con_feats.shape[0],
    }


# ---------------------------------------------------------------------------
# Train / Val split
# ---------------------------------------------------------------------------

def split_instances(instance_files: list, seed: int):
    """
    Deterministically split *instance_files* into 240 train + 60 val.
    Returns (train_files, val_files).
    """
    indices = list(range(len(instance_files)))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    val_indices = sorted(indices[:VAL_COUNT])
    train_indices = sorted(indices[VAL_COUNT:VAL_COUNT + TRAIN_COUNT])

    train_files = [instance_files[i] for i in train_indices]
    val_files = [instance_files[i] for i in val_indices]
    return train_files, val_files


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_problem(problem: str, seed: int):
    """Process all instances for one problem type."""
    inst_dir = os.path.join(INSTANCE_DIR, problem)
    if not os.path.isdir(inst_dir):
        print(f"[SKIP] instance dir not found: {inst_dir}")
        return

    # Collect and sort instance files
    inst_files = sorted(os.listdir(inst_dir))
    print(f"\n{'='*60}")
    print(f"Problem: {problem}  |  instances: {len(inst_files)}")
    print(f"{'='*60}")

    if len(inst_files) != TRAIN_COUNT + VAL_COUNT:
        print(f"  [INFO] expected {TRAIN_COUNT + VAL_COUNT} instances, "
              f"got {len(inst_files)}; adjusting split proportionally")

    # Split
    train_files, val_files = split_instances(inst_files, seed)

    # Output dirs
    for split_name in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_DIR, problem, split_name), exist_ok=True)

    # Save split info
    split_info = {
        "train": [os.path.splitext(f)[0] for f in train_files],
        "val": [os.path.splitext(f)[0] for f in val_files],
    }
    with open(os.path.join(OUTPUT_DIR, problem, "split.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Process each split
    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        print(f"\n--- {split_name}: {len(file_list)} instances ---")
        success = 0
        for idx, inst_file in enumerate(file_list):
            stem = os.path.splitext(inst_file)[0]
            out_path = os.path.join(OUTPUT_DIR, problem, split_name, f"{stem}.pkl")

            if os.path.exists(out_path):
                success += 1
                continue

            t0 = time.time()
            inst_path = os.path.join(inst_dir, inst_file)
            try:
                result = process_instance(problem, inst_file, inst_path)
            except Exception as e:
                print(f"  [{idx+1}/{len(file_list)}] {stem}  ERROR: {e}")
                continue

            if result is None:
                continue

            with open(out_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            elapsed = time.time() - t0
            success += 1
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(file_list)}] {stem}  "
                      f"V={result['n_vars']} C={result['n_cons']} "
                      f"E={result['edge_indices'].shape[1]}  "
                      f"{elapsed:.1f}s")

        print(f"  => {success}/{len(file_list)} saved")


def main():
    parser = argparse.ArgumentParser(description="Build MILP bipartite-graph dataset")
    parser.add_argument("--problems", nargs="+", default=ALL_PROBLEMS,
                        choices=ALL_PROBLEMS, help="Which problems to process")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # Reproducibility
    sys.path.insert(0, SCRIPT_DIR)
    from utils import set_seed
    set_seed(args.seed)

    _lazy_imports()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Instance dir : {INSTANCE_DIR}")
    print(f"Solution dir : {SOLUTION_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Seed         : {args.seed}")
    print(f"Problems     : {args.problems}")

    for prob in args.problems:
        process_problem(prob, args.seed)

    # Save global feature description
    desc_path = os.path.join(OUTPUT_DIR, "feature_description.json")
    desc = {
        "variable_features (21 cols)": {
            "0  objective": "normalised objective coefficient",
            "1  is_type_binary": "1 if binary variable",
            "2  is_type_integer": "1 if general integer variable",
            "3  is_type_implicit_integer": "1 if implicit integer",
            "4  is_type_continuous": "1 if continuous",
            "5  has_lower_bound": "1 if finite lower bound",
            "6  has_upper_bound": "1 if finite upper bound",
            "7  normed_reduced_cost": "LP reduced cost (initial, normalised)",
            "8  solution_value": "LP relaxation solution value (initial)",
            "9  solution_frac": "fractionality of LP solution (initial)",
            "10 is_solution_at_lb": "1 if LP sol at lower bound",
            "11 is_solution_at_ub": "1 if LP sol at upper bound",
            "12 scaled_age": "scaled age (initial, = 0)",
            "13 incumbent_value": "incumbent value (initial)",
            "14 average_incumbent_value": "average incumbent (initial)",
            "15 is_basis_lower": "1 if basis status = lower",
            "16 is_basis_basic": "1 if basis status = basic",
            "17 is_basis_upper": "1 if basis status = upper",
            "18 is_basis_zero": "1 if basis status = zero/free",
            "19 degree": "number of constraints the variable appears in",
            "20 raw_obj_coefficient": "un-normalised objective coefficient",
        },
        "constraint_features (6 cols)": {
            "0  bias": "normalised RHS",
            "1  objective_cosine_similarity": "cosine similarity with objective",
            "2  is_tight": "1 if tight at LP solution (initial)",
            "3  dual_solution_value": "LP dual value (initial)",
            "4  scaled_age": "scaled age (initial, = 0)",
            "5  degree": "number of variables in the constraint",
        },
        "edge_values": "coefficient of the variable in the constraint (normalised)",
    }
    with open(desc_path, "w") as f:
        json.dump(desc, f, indent=2)

    print(f"\nDone. Feature description saved to {desc_path}")


if __name__ == "__main__":
    main()
