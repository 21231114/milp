#!/usr/bin/env python3
"""
visualize_attention.py — 可视化 Q_macro 切片权重矩阵 (Transolver W)
===================================================================

新架构说明 (Transolver slice/deslice):
  W[M, N] = softmax(Q_macro @ K^T / sqrt(d), dim=0)
  每列 W[:, i] 表示节点 i 在 M 个探针上的归属分布。
  与旧架构不同，softmax 沿 M 维度 (dim=0) 而非 N 维度做，
  因此每行（探针）的颜色差异将非常清晰，避免稀释问题。

用法:
    # 单个实例
    python visualize_attention.py --checkpoint runs/run_001/best.pt --instances processed/CA/val/instance_1.pkl

    # 多个实例 (支持 .pkl 处理后文件 或 .mps/.lp 原始文件)
    python visualize_attention.py --checkpoint best.pt --instances inst1.pkl inst2.pkl --output avg_attn.png
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import MILPGNNModel
# 尝试导入 inference 中的 extract_features，以便支持原始 .mps / .lp 文件直接读取
try:
    from inference import extract_features
except ImportError:
    extract_features = None

def load_instance(path):
    """根据文件后缀加载实例数据 (.pkl 缓存，或调用 extract_features 直接处理 .mps)"""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    elif path.endswith(".mps") or path.endswith(".lp"):
        if extract_features is None:
            raise RuntimeError("处理原始实例需要从 inference.py 导入 extract_features 失败。")
        return extract_features(path)
    else:
        raise ValueError(f"不支持的实例文件格式 (仅支持 .pkl / .mps / .lp): {path}")

def main():
    parser = argparse.ArgumentParser(description="可视化 Q_macro 切片权重热力图 (Transolver W)")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径 (.pt)")
    parser.add_argument("--instances", nargs="+", required=True, help="要测试的实例文件路径列表 (.pkl 或 .mps)")
    parser.add_argument("--output", default="attention_heatmap.png", help="输出图片路径 (默认: attention_heatmap.png)")
    parser.add_argument("--align_size", type=int, default=100, help="实例大小不同时，用来对齐的变量维度大小 (默认: 100)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = MILPGNNModel(
        hidden_dim=cfg.get("hidden_dim", 64),
        n_gcn_layers=cfg.get("n_gcn_layers", 2),
        n_probes=cfg.get("n_probes", 16),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"成功加载模型，隐藏维度: {cfg.get('hidden_dim', 64)}, 宏观探针数(M): {cfg.get('n_probes', 16)}")
    print("注意: 新架构使用 softmax(dim=0) 的切片权重 W，每列对 M 个探针归一化")

    slice_weight_matrices = []
    original_sizes = []

    # 2. 遍历推理所有传入的实例
    for inst_path in args.instances:
        print(f"正在处理: {inst_path}")
        data = load_instance(inst_path)
        vf = torch.tensor(data["var_feats"], dtype=torch.float32, device=device)
        cf = torch.tensor(data["con_feats"], dtype=torch.float32, device=device)
        ei = torch.tensor(data["edge_indices"], dtype=torch.long, device=device)
        ev = torch.tensor(data["edge_values"], dtype=torch.float32, device=device)

        with torch.no_grad():
            # 开启 return_aux=True 以获取切片权重 W
            out = model(vf, cf, ei, ev, return_aux=True)
            # 新架构: aux 中包含 W_slice [M, N]，softmax over M-dim
            W = out["aux"]["W_slice"]    # [M, N]
            N = W.size(1)

            print(f"  N={N}  W.max()={W.max().item():.4f}  W.mean()={W.mean().item():.4f}")
            print(f"  每列和 (应≈1): {W.sum(0).mean().item():.4f} ± {W.sum(0).std().item():.4f}")

            slice_weight_matrices.append(W.cpu().numpy())
            original_sizes.append(N)

    # 3. 处理多个实例的求平均逻辑
    unique_sizes = set(original_sizes)
    if len(unique_sizes) > 1:
        print(f"\n检测到传入的实例包含不同的变量数量 (N={unique_sizes})。")
        print(f"正在使用双线性插值将特征轴对齐到 N={args.align_size} 以计算平均矩阵...")
        aligned_matrices = []
        for W_np in slice_weight_matrices:
            W_t = torch.tensor(W_np).unsqueeze(0).unsqueeze(0)
            W_aligned = F.interpolate(W_t, size=(W_np.shape[0], args.align_size), mode='bilinear', align_corners=False)
            aligned_matrices.append(W_aligned.squeeze().numpy())
        avg_W = np.mean(aligned_matrices, axis=0)
        x_label = f"变量节点 (N 被对齐插值到 {args.align_size} bins)"
    else:
        avg_W = np.mean(slice_weight_matrices, axis=0)
        x_label = f"变量节点 (N = {original_sizes[0]})"

    # 4. 绘制热力图并保存
    M = avg_W.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 原始切片权重热力图
    im = axes[0].imshow(avg_W, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=axes[0], label='切片权重 W (softmax over M)')
    axes[0].set_xlabel(x_label, fontsize=11)
    axes[0].set_ylabel(f"宏观探针 (索引 0 到 {M-1})", fontsize=11)
    title_prefix = "平均切片权重矩阵" if len(args.instances) > 1 else "切片权重矩阵"
    axes[0].set_title(f"{title_prefix} W [M×N]\n(每列和≈1，列内高对比度代表清晰的探针归属)", fontsize=12)

    # 右图: 每个探针的覆盖强度 (行均值)
    row_means = avg_W.mean(axis=1)      # [M]  probe coverage intensity
    row_maxes = avg_W.max(axis=1)       # [M]  max weight per probe
    axes[1].barh(np.arange(M), row_means, color='steelblue', alpha=0.7, label='行均值')
    axes[1].barh(np.arange(M), row_maxes, color='orange', alpha=0.5, label='行最大值')
    axes[1].set_xlabel("权重值", fontsize=11)
    axes[1].set_ylabel(f"宏观探针 (0 到 {M-1})", fontsize=11)
    axes[1].set_title("各探针的权重强度\n(均匀分布→探针过于相似；差异大→探针专一)", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].axvline(1.0 / M, color='red', linestyle='--', alpha=0.7, label=f'均匀基准 1/M={1/M:.3f}')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n完成！可视化热力图已保存至: {args.output}")

if __name__ == "__main__":
    main()