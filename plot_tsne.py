"""
本地 t-SNE 可视化脚本

读取 embeddings_for_tsne.npz → t-SNE 降维 → 画 1×3 子图 → 保存 PDF/PNG

使用方法:
    pip install numpy matplotlib scikit-learn
    python plot_tsne.py --input embeddings_for_tsne.npz --output figures/tsne.pdf
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── 配色与标记 ──
LABEL_NAMES = {0: "Normal", 1: "RCE", 2: "Non-RCE"}
COLORS = {"Normal": "#4472C4", "RCE": "#C00000", "Non-RCE": "#808080"}
MARKERS = {"Normal": "o", "RCE": "o", "Non-RCE": "^"}
SIZES = {"Normal": 10, "RCE": 12, "Non-RCE": 14}
DRAW_ORDER = ["Normal", "RCE", "Non-RCE"]


def run_tsne(features, perplexity=30, seed=42):
    """标准化 + t-SNE"""
    scaled = StandardScaler().fit_transform(features)
    tsne = TSNE(
        n_components=2, random_state=seed,
        perplexity=min(perplexity, len(features) - 1),
        n_iter=1000, learning_rate=200.0, init="random",
    )
    return tsne.fit_transform(scaled)


def plot_panel(ax, emb, label_names, title, show_proto=False):
    """绘制单个子图"""
    for cls in DRAW_ORDER:
        mask = label_names == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=COLORS[cls], marker=MARKERS[cls],
            s=SIZES[cls], alpha=0.55, edgecolors="none",
            label=cls, zorder=2 if cls == "Non-RCE" else 1,
        )

    if show_proto:
        for cls in ["Normal", "RCE"]:
            mask = label_names == cls
            if mask.sum() == 0:
                continue
            pts = emb[mask]
            center = pts.mean(axis=0)
            ax.scatter(
                center[0], center[1], marker="*", s=220,
                c=COLORS[cls], edgecolors="black", linewidths=1.2, zorder=4,
            )
            dists = np.linalg.norm(pts - center, axis=1)
            radius = np.median(dists) + 1.4826 * np.median(np.abs(dists - np.median(dists)))
            circle = plt.Circle(
                center, radius, fill=False, edgecolor=COLORS[cls],
                linewidth=1.2, linestyle="--", alpha=0.6, zorder=3,
            )
            ax.add_patch(circle)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description="t-SNE 可视化")
    parser.add_argument("--input", default="embeddings_for_tsne.npz", help=".npz 文件路径")
    parser.add_argument("--output", default="tsne_visualization.pdf", help="输出图片路径")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=800,
                        help="最多采样多少个点 (太多会很慢且密集)")
    args = parser.parse_args()

    print(f"加载 {args.input} ...")
    data = np.load(args.input)
    emb_weak = data["embeddings_weak"]
    emb_full = data["embeddings_full"]
    emb_mid = data["embeddings_noproto"]
    labels_raw = data["labels"]

    label_names = np.array([LABEL_NAMES.get(l, f"Unk_{l}") for l in labels_raw])
    print(f"总样本数: {len(labels_raw)}")
    for cls in DRAW_ORDER:
        print(f"  {cls}: {(label_names == cls).sum()}")

    # 采样（如果太多）
    if len(labels_raw) > args.max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(labels_raw), args.max_samples, replace=False)
        emb_weak = emb_weak[idx]
        emb_full = emb_full[idx]
        emb_mid = emb_mid[idx]
        label_names = label_names[idx]
        print(f"采样到 {args.max_samples} 个点")

    panels = [
        (emb_weak, "(a) 适应前", False),
        (emb_full, "(b) PCKD适应后", True),
        (emb_mid, "(c) DREAM适应后", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (emb, title, show_proto) in zip(axes, panels):
        print(f"  t-SNE: {title} ...")
        emb_2d = run_tsne(emb, perplexity=args.perplexity)
        plot_panel(ax, emb_2d, label_names, title, show_proto=show_proto)

    handles, legs = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legs, loc="upper center", ncol=3,
        fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, 1.01), markerscale=1.8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"已保存: {args.output}")

    # 同时保存 PNG
    png_path = args.output.rsplit(".", 1)[0] + ".png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"已保存: {png_path}")

    plt.show()


if __name__ == "__main__":
    main()
