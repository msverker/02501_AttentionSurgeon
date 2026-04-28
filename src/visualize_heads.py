"""
visualize_heads.py — Analyse and visualise attention-head profiles.

Produces:
  figures/periodic_table.png    — 12×12 "periodic table" coloured by cluster
  figures/heatmaps_all.png      — all metrics in one dark heatmap figure (kept)
  figures/cluster_scatter.png   — PCA scatter coloured by cluster
  figures/metric_correlation.png
  (stdout)                      — cluster statistics and per-layer summary

Usage:
  python visualize_heads.py
  python visualize_heads.py --input head_profiles.npz --n_clusters 4
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd


# ---------------------------------------------------------------------- #
#  Theme                                                                   #
# ---------------------------------------------------------------------- #
sns.set_theme(style="ticks", context="paper", font_scale=1.1)

CANONICAL_GROUPS = [
    "Local-Edge",
    "Global-Semantic",
    "CLS-Aggregator",
    "Positional",
    "Dead/Redundant",
]

GROUP_COLORS = {
    "Local-Edge": "#E74C3C",
    "Global-Semantic": "#27AE60",
    "CLS-Aggregator": "#2980B9",
    "Positional": "#E67E22",
    "Dead/Redundant": "#95A5A6",
}

METRIC_LABELS = {
    "entropy": "Attention Entropy",
    "distance": "Attention Distance",
    "activation_mag": "Activation Magnitude",
    "importance": "Head Importance",
    "cls_query_entropy": "CLS Query Entropy",
}

# Dark colormap for heatmaps_all
VIRIDIS_DARK = LinearSegmentedColormap.from_list(
    "viridis_dark",
    ["#0F1117", "#1B4F8A", "#2E8B7A", "#7EC850", "#F5D000"],
)


# ---------------------------------------------------------------------- #
#  I/O                                                                     #
# ---------------------------------------------------------------------- #
def load_profiles(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------- #
#  Clustering                                                              #
# ---------------------------------------------------------------------- #
def cluster_heads(profiles, n_clusters=4):
    vecs, names = [], []
    for key in [
        "entropy",
        "distance",
        "activation_mag",
        "cls_query_entropy",
        "importance",
    ]:
        if key in profiles:
            vecs.append(profiles[key].flatten())
            names.append(key)
    X = np.column_stack(vecs)
    X_s = StandardScaler().fit_transform(X)
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_s)
    return labels, X_s, names


def label_clusters(labels, X_scaled, feature_names):
    n_clusters = len(set(labels))
    centroids = np.zeros((n_clusters, X_scaled.shape[1]))
    for c in range(n_clusters):
        centroids[c] = X_scaled[labels == c].mean(axis=0)

    feat = {n: i for i, n in enumerate(feature_names)}
    assignments, used = {}, set()

    def _pick(order, role):
        for c in order:
            if c not in used:
                assignments[c] = role
                used.add(c)
                return

    if "activation_mag" in feat:
        _pick(
            sorted(
                range(n_clusters), key=lambda c: centroids[c, feat["activation_mag"]]
            ),
            "Dead/Redundant",
        )
    if "cls_query_entropy" in feat:
        _pick(
            sorted(
                range(n_clusters), key=lambda c: centroids[c, feat["cls_query_entropy"]]
            ),
            "CLS-Aggregator",
        )
    if "distance" in feat:
        _pick(
            sorted(range(n_clusters), key=lambda c: centroids[c, feat["distance"]]),
            "Local-Edge",
        )
        _pick(
            sorted(range(n_clusters), key=lambda c: -centroids[c, feat["distance"]]),
            "Global-Semantic",
        )
    for c in range(n_clusters):
        if c not in assignments:
            assignments[c] = "Positional"
    return assignments


# ---------------------------------------------------------------------- #
#  Periodic table  (clean white)                                           #
# ---------------------------------------------------------------------- #
def plot_periodic_table(profiles, labels, name_map, output_dir):
    NL, NH = 12, 12
    grid = labels.reshape(NL, NH)

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
    ax.set_facecolor("white")

    for layer in range(NL):
        for h in range(NH):
            gname = name_map[grid[layer, h]]
            x, y = h, NL - 1 - layer
            color = GROUP_COLORS.get(gname, "#95A5A6")

            rect = plt.Rectangle(
                (x + 0.05, y + 0.05),
                0.9,
                0.9,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
                alpha=0.12,
                zorder=1,
            )
            border = plt.Rectangle(
                (x + 0.05, y + 0.05),
                0.9,
                0.9,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.add_patch(border)

            ax.text(
                x + 0.5,
                y + 0.78,
                f"L{layer}·H{h}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
                fontweight="bold",
                zorder=3,
            )

            lines = []
            if "entropy" in profiles:
                lines.append(f"E {profiles['entropy'][layer, h]:.2f}")
            if "distance" in profiles:
                lines.append(f"D {profiles['distance'][layer, h]:.2f}")
            if "importance" in profiles:
                lines.append(f"I {profiles['importance'][layer, h]:.2f}")

            ax.text(
                x + 0.5,
                y + 0.38,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=4.0,
                color="#555555",
                zorder=3,
                linespacing=1.7,
            )

    ax.set_xlim(0, NH)
    ax.set_ylim(0, NL)
    ax.set_xticks([i + 0.5 for i in range(NH)])
    ax.set_xticklabels([f"H{i}" for i in range(NH)], fontsize=8, color="#666666")
    ax.set_yticks([i + 0.5 for i in range(NL)])
    ax.set_yticklabels(
        [f"L{i}" for i in reversed(range(NL))], fontsize=8, color="#666666"
    )
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    sns.despine(ax=ax, left=True, bottom=True)

    ax.set_title(
        "Periodic Table of Attention Heads  ·  DINOv2 ViT-B/16",
        fontsize=14,
        fontweight="bold",
        color="#222222",
        pad=16,
    )

    unique_groups = sorted(
        set(name_map.values()),
        key=lambda g: CANONICAL_GROUPS.index(g) if g in CANONICAL_GROUPS else 99,
    )
    patches = [
        mpatches.Patch(
            facecolor=GROUP_COLORS.get(g, "#95A5A6"),
            edgecolor="none",
            label=g,
            alpha=0.8,
        )
        for g in unique_groups
    ]
    ax.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Functional Group",
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#DDDDDD",
    )

    plt.tight_layout()
    fig.savefig(
        output_dir / "periodic_table.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  → periodic_table.png")


# ---------------------------------------------------------------------- #
#  Heatmaps  (dark — kept as requested)                                   #
# ---------------------------------------------------------------------- #
def plot_heatmaps(profiles, output_dir):
    keys = list(profiles.keys())
    n = len(keys)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), facecolor="#0F1117")
    axes = np.array(axes).flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        ax.set_facecolor("#0F1117")
        arr = profiles[key]

        im = ax.imshow(arr, aspect="auto", cmap=VIRIDIS_DARK, interpolation="nearest")

        for layer in range(12):
            for h in range(12):
                val = arr[layer, h]
                brightness = (val - arr.min()) / (arr.max() - arr.min() + 1e-8)
                txt_color = "#FFFFFF" if brightness < 0.6 else "#0F1117"
                ax.text(
                    h,
                    layer,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color=txt_color,
                )

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#6B7280", labelsize=7)
        cbar.outline.set_edgecolor("#2A2D3A")

        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xticklabels([f"H{i}" for i in range(12)], fontsize=7, color="#6B7280")
        ax.set_yticklabels([f"L{i}" for i in range(12)], fontsize=7, color="#6B7280")
        ax.set_xlabel("Head", fontsize=9, color="#A0A8B8")
        ax.set_ylabel("Layer", fontsize=9, color="#A0A8B8")
        ax.set_title(
            METRIC_LABELS.get(key, key.replace("_", " ").title()),
            fontsize=11,
            color="#FFFFFF",
            pad=10,
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2D3A")

    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Attention Head Metrics  ·  DINOv2 ViT-B/16",
        fontsize=14,
        fontweight="bold",
        color="#FFFFFF",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(
        output_dir / "heatmaps_all.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0F1117",
    )
    plt.close(fig)
    print("  → heatmaps_all.png")


# ---------------------------------------------------------------------- #
#  Cluster scatter  (clean white + seaborn)                               #
# ---------------------------------------------------------------------- #
def plot_cluster_scatter(X_scaled, labels, name_map, output_dir):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X_scaled)

    rows = []
    for idx, (x, y) in enumerate(X2):
        layer, h = divmod(idx, 12)
        rows.append(
            {
                "PC1": x,
                "PC2": y,
                "layer": layer,
                "head": h,
                "group": name_map[labels[idx]],
                "label": f"{layer}·{h}",
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 9), facecolor="white")
    ax.set_facecolor("white")

    for group in df["group"].unique():
        sub = df[df["group"] == group]
        color = GROUP_COLORS.get(group, "#95A5A6")
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            c=color,
            label=group,
            s=100,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["label"],
                (row["PC1"], row["PC2"]),
                fontsize=5,
                ha="center",
                va="bottom",
                color="#888888",
                xytext=(0, 5),
                textcoords="offset points",
            )

    ax.axhline(0, color="#EEEEEE", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#EEEEEE", linewidth=0.8, zorder=1)
    ax.grid(True, color="#F0F0F0", linewidth=0.5, zorder=0)

    ax.set_xlabel(
        f"PC1  ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=10
    )
    ax.set_ylabel(
        f"PC2  ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=10
    )
    ax.set_title("Head Cluster Map  ·  PCA Projection", fontsize=13, pad=14)
    ax.legend(
        title="Functional Group",
        title_fontsize=9,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#DDDDDD",
    )

    sns.despine(ax=ax, offset=8, trim=True)
    plt.tight_layout()
    fig.savefig(
        output_dir / "cluster_scatter.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  → cluster_scatter.png")


# ---------------------------------------------------------------------- #
#  Correlation matrix  (clean white + seaborn)                            #
# ---------------------------------------------------------------------- #
def plot_correlation_matrix(profiles, output_dir):
    keys = [
        k
        for k in [
            "entropy",
            "distance",
            "activation_mag",
            "cls_query_entropy",
            "importance",
        ]
        if k in profiles
    ]
    if len(keys) < 2:
        return

    mat = np.column_stack([profiles[k].flatten() for k in keys])
    corr = np.corrcoef(mat.T)
    tick_labels = [METRIC_LABELS.get(k, k) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        square=True,
        linewidths=0.5,
        linecolor="#EEEEEE",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
        annot_kws={"fontsize": 9},
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("Metric Correlations", fontsize=13, pad=14)
    ax.tick_params(length=0)

    plt.tight_layout()
    fig.savefig(
        output_dir / "metric_correlation.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  → metric_correlation.png")


# ---------------------------------------------------------------------- #
#  Summary                                                                 #
# ---------------------------------------------------------------------- #
def print_summary(labels, name_map):
    NL, NH = 12, 12
    grid = labels.reshape(NL, NH)

    print("\nCluster sizes:")
    for c, gname in sorted(name_map.items(), key=lambda x: x[1]):
        print(f"  {gname:20s}: {(labels == c).sum()} heads")

    print("\nPer-layer distribution:")
    for layer in range(NL):
        row_names = [name_map[c] for c in grid[layer]]
        dist = Counter(row_names)
        summary = "  ".join(f"{g}: {n}" for g, n in sorted(dist.items()))
        print(f"  Layer {layer:2d} │ {summary}")


# ---------------------------------------------------------------------- #
#  CLI                                                                     #
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="head_profiles.npz")
    ap.add_argument("--output_dir", type=str, default="figures")
    ap.add_argument("--n_clusters", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

    profiles = load_profiles(args.input)
    print(f"Loaded metrics: {list(profiles.keys())}")

    labels, X_s, feat_names = cluster_heads(profiles, args.n_clusters)
    name_map = label_clusters(labels, X_s, feat_names)

    print(f"\nSaving figures to {out}/")
    plot_periodic_table(profiles, labels, name_map, out)
    plot_heatmaps(profiles, out)
    plot_cluster_scatter(X_s, labels, name_map, out)
    plot_correlation_matrix(profiles, out)

    print_summary(labels, name_map)


if __name__ == "__main__":
    main()
