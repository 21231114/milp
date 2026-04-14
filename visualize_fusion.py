#!/usr/bin/env python3
"""
visualize_fusion.py — 可视化模型融合参数
==========================================

展示三路特征对最终预测的贡献比例：
  - 原始动态特征 (dynamic_out):   alpha * (1 - c)
  - 宏观反馈动态特征 (H_feedback): alpha * c
  - 静态特征 (static_out):         1 - alpha

其中：
  alpha = StaticDynamicFusion 的门控（alpha → 1 表示更依赖动态）
  c     = MicroMacroFusion 的门控（c → 1 表示更依赖宏观反馈）

Usage
-----
    python visualize_fusion.py --checkpoint runs/run_013/best.pt
    python visualize_fusion.py --checkpoint runs/run_013/best.pt --problems CA IP --split val
    python visualize_fusion.py --checkpoint runs/run_013/best.pt --max_instances 5
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import MILPGNNModel, STATIC_VAR_IDX, DYNAMIC_VAR_IDX
from model import _COL_BIN, _COL_INT, _COL_IMP, _COL_CON


# ══════════════════════════════════════════════════════════════════════════
#  带中间量返回的前向传播（不修改模型，仅 hooks）
# ══════════════════════════════════════════════════════════════════════════

def forward_with_gates(model: MILPGNNModel, vf, cf, ei, ev):
    """
    运行一次前向传播，返回：
      alpha : [N, d]   StaticDynamicFusion 的门控
      c     : [N, 1]   MicroMacroFusion 的门控
      以及三路贡献比例（对 d 维取均值后得到 [N] 标量）
    """
    model.eval()
    with torch.no_grad():
        # ── 复现 _forward_single 的步骤，手动提取中间量 ──
        var_types = vf[:, [_COL_BIN, _COL_INT, _COL_IMP, _COL_CON]]
        lp_sol = vf[:, 8]

        var_feats = model.normalizer.normalize_var(vf)
        con_feats = model.normalizer.normalize_con(cf)
        edge_val  = model.normalizer.normalize_edge(ev)

        static_f  = var_feats[:, STATIC_VAR_IDX]
        dynamic_f = var_feats[:, DYNAMIC_VAR_IDX]

        # GCN
        static_out,  _ = model.static_gcn(static_f, con_feats, ei, edge_val)
        dynamic_out, _ = model.dynamic_gcn(dynamic_f, con_feats, ei, edge_val)

        # MacroMicro Attention
        H_fb, A_fwd, H_macro = model.attention(static_out, dynamic_out)

        # MicroMacro Fusion → 提取 c
        w = H_macro.norm(dim=-1)
        w = w / (w.sum() + 1e-8)
        S_i = (w.unsqueeze(-1) * A_fwd).sum(dim=0)  # [N]
        S_i = S_i * S_i.size(0)
        c = torch.sigmoid(model.mm_fusion.gamma * S_i + model.mm_fusion.beta)  # [N]
        c_expand = c.unsqueeze(-1)                                               # [N,1]
        V_updated = c_expand * H_fb + (1.0 - c_expand) * dynamic_out

        # StaticDynamic Fusion → 提取 alpha
        alpha = torch.sigmoid(
            model.sd_fusion.W_gate(
                torch.cat([static_out, V_updated], dim=-1)))  # [N, d]

        Feature_final = alpha * V_updated + (1.0 - alpha) * static_out

        # 三路贡献（按维度平均后得到 per-variable 标量）
        alpha_mean = alpha.mean(dim=-1)           # [N]
        contrib_static   = 1.0 - alpha_mean       # [N]
        contrib_dyn_orig = alpha_mean * (1.0 - c) # [N]
        contrib_dyn_macro= alpha_mean * c          # [N]

    return {
        "alpha":           alpha_mean.cpu().numpy(),    # [N]
        "c":               c.cpu().numpy(),              # [N]
        "contrib_static":  contrib_static.cpu().numpy(),
        "contrib_dyn_orig":contrib_dyn_orig.cpu().numpy(),
        "contrib_dyn_macro":contrib_dyn_macro.cpu().numpy(),
        "var_types":       var_types.cpu().numpy(),      # [N, 4]
        "gamma":           model.mm_fusion.gamma.item(),
        "beta":            model.mm_fusion.beta.item(),
    }


# ══════════════════════════════════════════════════════════════════════════
#  数据加载
# ══════════════════════════════════════════════════════════════════════════

def iter_instances(problem, split, max_n):
    split_dir = os.path.join(SCRIPT_DIR, "processed", problem, split)
    if not os.path.isdir(split_dir):
        print(f"  [SKIP] {split_dir} not found")
        return
    pkls = sorted(f for f in os.listdir(split_dir) if f.endswith(".pkl"))[:max_n]
    for pkl_name in pkls:
        path = os.path.join(split_dir, pkl_name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        yield pkl_name[:-4], data


# ══════════════════════════════════════════════════════════════════════════
#  绘图
# ══════════════════════════════════════════════════════════════════════════

COLORS = {
    "static":    "#4C72B0",
    "dyn_orig":  "#DD8452",
    "dyn_macro": "#55A868",
}
LABELS = {
    "static":    "Static features (1-alpha)",
    "dyn_orig":  "Orig dynamic alpha*(1-c)",
    "dyn_macro": "Macro-feedback alpha*c",
}


def plot_problem_summary(all_stats, problem, out_path):
    """
    为一个 problem 画汇总图：
      左上: 三路贡献均值柱状图（全部变量 + 按类型分组）
      右上: alpha 和 c 的分布直方图
      下  : 逐实例贡献堆叠条形图
    """
    n_inst = len(all_stats)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Feature Fusion Weights — {problem}  ({n_inst} instances)", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax_bar   = fig.add_subplot(gs[0, 0])
    ax_hist  = fig.add_subplot(gs[0, 1])
    ax_gate  = fig.add_subplot(gs[0, 2])
    ax_stack = fig.add_subplot(gs[1, :])

    # ── 汇总所有变量的贡献 ──────────────────────────────────────────────
    all_cs = np.concatenate([s["contrib_static"]   for s in all_stats])
    all_do = np.concatenate([s["contrib_dyn_orig"]  for s in all_stats])
    all_dm = np.concatenate([s["contrib_dyn_macro"] for s in all_stats])
    all_vt = np.concatenate([s["var_types"]         for s in all_stats], axis=0)  # [total, 4]

    # 全局均值
    mean_cs = all_cs.mean()
    mean_do = all_do.mean()
    mean_dm = all_dm.mean()

    # 按变量类型分组
    type_names = ["binary", "integer", "implicit_int", "continuous"]
    groups = {"all": (mean_cs, mean_do, mean_dm)}
    for ti, tname in enumerate(type_names):
        mask = all_vt[:, ti].astype(bool)
        if mask.sum() == 0:
            continue
        groups[tname] = (all_cs[mask].mean(), all_do[mask].mean(), all_dm[mask].mean())

    # 柱状图
    group_keys = list(groups.keys())
    x = np.arange(len(group_keys))
    w = 0.25
    for ki, key in enumerate(["static", "dyn_orig", "dyn_macro"]):
        vals = [groups[g][["static","dyn_orig","dyn_macro"].index(key)] for g in group_keys]
        ax_bar.bar(x + (ki-1)*w, vals, w, label=LABELS[key], color=COLORS[key], alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(group_keys, rotation=15, ha="right", fontsize=8)
    ax_bar.set_ylabel("Mean contribution ratio")
    ax_bar.set_title("Three-way contribution (by var type)")
    ax_bar.legend(fontsize=7)
    ax_bar.set_ylim(0, 1)
    ax_bar.axhline(0.333, color="gray", lw=0.8, ls="--", alpha=0.5)

    # ── alpha / c 分布 ──────────────────────────────────────────────────
    ax_hist.hist(all_cs, bins=40, color=COLORS["static"],    alpha=0.6, label="静态 1−α", density=True)
    ax_hist.hist(all_do, bins=40, color=COLORS["dyn_orig"],  alpha=0.6, label="原始动态 α(1−c)", density=True)
    ax_hist.hist(all_dm, bins=40, color=COLORS["dyn_macro"], alpha=0.6, label="宏观反馈 αc", density=True)
    ax_hist.set_xlabel("Contribution")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Three-way contribution distribution")
    ax_hist.legend(fontsize=7)

    # ── alpha & c 门控本身的分布 ────────────────────────────────────────
    all_alpha = np.concatenate([s["alpha"] for s in all_stats])
    all_c     = np.concatenate([s["c"]     for s in all_stats])
    ax_gate.hist(all_alpha, bins=40, color="#9467bd", alpha=0.7, label="alpha (dyn gate)", density=True)
    ax_gate.hist(all_c,     bins=40, color="#e377c2", alpha=0.7, label="c (macro gate)", density=True)
    ax_gate.set_xlabel("Gate value")
    ax_gate.set_ylabel("Density")
    ax_gate.set_title("Gate params alpha & c distribution")
    gamma_val = all_stats[0]["gamma"]
    beta_val  = all_stats[0]["beta"]
    ax_gate.text(0.02, 0.95, f"γ={gamma_val:.3f}  β={beta_val:.3f}",
                 transform=ax_gate.transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.6))
    ax_gate.legend(fontsize=7)

    # ── 逐实例堆叠条形图 ───────────────────────────────────────────────
    inst_names = [s["name"] for s in all_stats]
    inst_cs = [s["contrib_static"].mean()    for s in all_stats]
    inst_do = [s["contrib_dyn_orig"].mean()  for s in all_stats]
    inst_dm = [s["contrib_dyn_macro"].mean() for s in all_stats]

    xi = np.arange(n_inst)
    ax_stack.bar(xi, inst_cs, label=LABELS["static"],    color=COLORS["static"],    alpha=0.85)
    ax_stack.bar(xi, inst_do, bottom=inst_cs,             label=LABELS["dyn_orig"],  color=COLORS["dyn_orig"],  alpha=0.85)
    bot2 = [a+b for a,b in zip(inst_cs, inst_do)]
    ax_stack.bar(xi, inst_dm, bottom=bot2,                label=LABELS["dyn_macro"], color=COLORS["dyn_macro"], alpha=0.85)
    ax_stack.set_xticks(xi)
    ax_stack.set_xticklabels(inst_names, rotation=45, ha="right", fontsize=7)
    ax_stack.set_ylabel("Mean contribution ratio")
    ax_stack.set_title("Per-instance three-way feature contribution (stacked)")
    ax_stack.set_ylim(0, 1.05)
    ax_stack.legend(fontsize=8, loc="upper right")
    ax_stack.axhline(1.0, color="gray", lw=0.8, ls="--")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存图像 -> {out_path}")


def print_summary(all_stats, problem):
    all_cs = np.concatenate([s["contrib_static"]    for s in all_stats])
    all_do = np.concatenate([s["contrib_dyn_orig"]  for s in all_stats])
    all_dm = np.concatenate([s["contrib_dyn_macro"] for s in all_stats])
    all_alpha = np.concatenate([s["alpha"] for s in all_stats])
    all_c     = np.concatenate([s["c"]     for s in all_stats])
    gamma = all_stats[0]["gamma"]
    beta  = all_stats[0]["beta"]

    total = all_stats[0]["contrib_static"].shape[0]  # per-instance size varies
    print(f"\n{'='*55}")
    print(f"  {problem}  |  {len(all_stats)} instances")
    print(f"{'='*55}")
    print(f"  可学习参数:  γ={gamma:.4f}  β={beta:.4f}")
    print(f"{'─'*55}")
    print(f"  {'':25s}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}")
    print(f"  {'静态特征 (1-α)':25s}  {all_cs.mean():7.4f}  {all_cs.std():7.4f}  {all_cs.min():7.4f}  {all_cs.max():7.4f}")
    print(f"  {'原始动态 α·(1-c)':25s}  {all_do.mean():7.4f}  {all_do.std():7.4f}  {all_do.min():7.4f}  {all_do.max():7.4f}")
    print(f"  {'宏观反馈动态 α·c':25s}  {all_dm.mean():7.4f}  {all_dm.std():7.4f}  {all_dm.min():7.4f}  {all_dm.max():7.4f}")
    print(f"{'─'*55}")
    print(f"  {'α (动态门控)':25s}  {all_alpha.mean():7.4f}  {all_alpha.std():7.4f}  {all_alpha.min():7.4f}  {all_alpha.max():7.4f}")
    print(f"  {'c (宏观门控)':25s}  {all_c.mean():7.4f}  {all_c.std():7.4f}  {all_c.min():7.4f}  {all_c.max():7.4f}")
    print(f"{'='*55}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="可视化模型融合门控参数")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint (.pt)")
    parser.add_argument("--problems",   nargs="+", default=["CA", "IP", "SC", "WA"])
    parser.add_argument("--split",      default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_instances", type=int, default=20, help="每个 problem 最多处理几个实例")
    parser.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "output", "fusion_vis"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    model = MILPGNNModel(
        hidden_dim=cfg.get("hidden_dim", 64),
        n_probes=cfg.get("n_probes", 16),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"已加载模型: {args.checkpoint}")
    print(f"  hidden_dim={cfg.get('hidden_dim',64)}  n_probes={cfg.get('n_probes',16)}")
    print(f"  γ={model.mm_fusion.gamma.item():.4f}  β={model.mm_fusion.beta.item():.4f}")

    for problem in args.problems:
        print(f"\n处理 {problem}/{args.split} ...")
        all_stats = []

        for stem, data in iter_instances(problem, args.split, args.max_instances):
            vf = torch.tensor(data["var_feats"],    dtype=torch.float32, device=device)
            cf = torch.tensor(data["con_feats"],    dtype=torch.float32, device=device)
            ei = torch.tensor(data["edge_indices"], dtype=torch.long,    device=device)
            ev = torch.tensor(data["edge_values"],  dtype=torch.float32, device=device)

            stats = forward_with_gates(model, vf, cf, ei, ev)
            stats["name"] = stem
            all_stats.append(stats)
            print(f"  {stem}: n_vars={vf.shape[0]}  "
                  f"α_mean={stats['alpha'].mean():.3f}  "
                  f"c_mean={stats['c'].mean():.3f}  "
                  f"static={stats['contrib_static'].mean():.3f}  "
                  f"dyn_orig={stats['contrib_dyn_orig'].mean():.3f}  "
                  f"dyn_macro={stats['contrib_dyn_macro'].mean():.3f}")

        if not all_stats:
            continue

        print_summary(all_stats, problem)
        out_png = os.path.join(args.output_dir, f"{problem}_{args.split}_fusion.png")
        plot_problem_summary(all_stats, problem, out_png)

    print(f"\n完成。图像保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
