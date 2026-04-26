"""
visualize_heads.py — Analyse and visualise attention-head profiles.

Produces:
  figures/periodic_table.png   — 12×12 "periodic table" of heads coloured by cluster
  figures/heatmap_*.png        — one heatmap per metric
  figures/cluster_scatter.png  — PCA scatter of heads coloured by cluster
  (stdout)                     — cluster statistics and per-layer summary

Usage:
  python visualize_heads.py                          # defaults
  python visualize_heads.py --input head_profiles.npz --n_clusters 5
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------- #
#  Colour palette for the five canonical functional groups                 #
# ---------------------------------------------------------------------- #
CANONICAL_GROUPS = [
    "Local-Edge",
    "Global-Semantic",
    "CLS-Aggregator",
    "Positional",
    "Dead/Redundant",
]

GROUP_COLORS = {
    "Local-Edge":       "#E74C3C",
    "Global-Semantic":  "#2ECC71",
    "CLS-Aggregator":   "#3498DB",
    "Positional":       "#F39C12",
    "Dead/Redundant":   "#95A5A6",
}


# ---------------------------------------------------------------------- #
#  I/O                                                                     #
# ---------------------------------------------------------------------- #
def load_profiles(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------- #
#  Clustering                                                              #
# ---------------------------------------------------------------------- #
def cluster_heads(profiles, n_clusters=5):
    """
    Build a feature matrix from all available metrics, standardise, cluster.
    Returns (labels, X_scaled, feature_names).
    """
    vecs, names = [], []
    for key in ["entropy", "distance", "activation_mag",
                "cls_query_entropy", "importance"]:
        if key in profiles:
            vecs.append(profiles[key].flatten())
            names.append(key)

    X = np.column_stack(vecs)  # (144, num_features)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_s)
    return labels, X_s, names


def label_clusters(labels, X_scaled, feature_names):
    """
    Heuristically assign a canonical name to each cluster based on its
    centroid position in feature space.
    """
    n_clusters = len(set(labels))
    centroids = np.zeros((n_clusters, X_scaled.shape[1]))
    for c in range(n_clusters):
        centroids[c] = X_scaled[labels == c].mean(axis=0)

    feat = {n: i for i, n in enumerate(feature_names)}

    # Score each cluster for each canonical role
    assignments = {}
    used_clusters = set()

    def _pick(scores, role):
        """Pick the cluster with the best score that hasn't been used yet."""
        for c in scores:
            if c not in used_clusters:
                assignments[c] = role
                used_clusters.add(c)
                return
    # Dead / Redundant — lowest activation magnitude
    if "activation_mag" in feat:
        order = sorted(range(n_clusters),
                       key=lambda c: centroids[c, feat["activation_mag"]])
        _pick(order, "Dead/Redundant")

    # CLS-Aggregator — lowest cls_query_entropy (focused CLS attention)
    if "cls_query_entropy" in feat:
        order = sorted(range(n_clusters),
                       key=lambda c: centroids[c, feat["cls_query_entropy"]])
        _pick(order, "CLS-Aggregator")

    # Local-Edge — lowest distance among remaining
    if "distance" in feat:
        order = sorted(range(n_clusters),
                       key=lambda c: centroids[c, feat["distance"]])
        _pick(order, "Local-Edge")

    # Global-Semantic — highest distance among remaining
    if "distance" in feat:
        order = sorted(range(n_clusters),
                       key=lambda c: -centroids[c, feat["distance"]])
        _pick(order, "Global-Semantic")

    # Everything else → Positional
    for c in range(n_clusters):
        if c not in assignments:
            assignments[c] = "Positional"

    return assignments


# ---------------------------------------------------------------------- #
#  Visualisations                                                          #
# ---------------------------------------------------------------------- #
def _group_color(name):
    return GROUP_COLORS.get(name, "#BDC3C7")


def plot_periodic_table(profiles, labels, name_map, output_dir):
    """12-layer × 12-head 'periodic table' coloured by functional group."""
    NL, NH = 12, 12
    grid = labels.reshape(NL, NH)

    fig, ax = plt.subplots(figsize=(16, 10))

    for l in range(NL):
        for h in range(NH):
            gname = name_map[grid[l, h]]
            x, y = h, NL - 1 - l

            rect = plt.Rectangle(
                (x, y), 1, 1,
                facecolor=_group_color(gname),
                edgecolor="white", linewidth=2,
            )
            ax.add_patch(rect)

            # Annotate with compact metrics
            lines = [f"L{l}H{h}"]
            if "entropy" in profiles:
                lines.append(f"E:{profiles['entropy'][l, h]:.2f}")
            if "distance" in profiles:
                lines.append(f"D:{profiles['distance'][l, h]:.1f}")
            ax.text(
                x + 0.5, y + 0.5, "\n".join(lines),
                ha="center", va="center", fontsize=5.5, fontweight="bold",
            )

    ax.set_xlim(-0.2, NH + 0.2)
    ax.set_ylim(-0.2, NL + 0.2)
    ax.set_xticks([i + 0.5 for i in range(NH)])
    ax.set_xticklabels(range(NH))
    ax.set_yticks([i + 0.5 for i in range(NL)])
    ax.set_yticklabels(list(reversed(range(NL))))
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        "Periodic Table of Attention Heads — DINOv2 ViT-B/16",
        fontsize=14, fontweight="bold",
    )
    ax.set_aspect("equal")

    unique_groups = sorted(set(name_map.values()), key=lambda g: CANONICAL_GROUPS.index(g)
                           if g in CANONICAL_GROUPS else 99)
    patches = [mpatches.Patch(color=_group_color(g), label=g) for g in unique_groups]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "periodic_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → periodic_table.png")


def plot_heatmaps(profiles, output_dir):
    """One heatmap per metric (12 layers × 12 heads)."""
    for key, arr in profiles.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Head {key.replace('_', ' ').title()}", fontsize=13)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fname = f"heatmap_{key}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {fname}")


def plot_cluster_scatter(X_scaled, labels, name_map, output_dir):
    """2-D PCA projection of heads coloured by cluster."""
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    for c in sorted(set(labels)):
        mask = labels == c
        gname = name_map[c]
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=_group_color(gname), label=gname,
            s=60, alpha=0.8, edgecolors="black", linewidth=0.5,
        )
        for idx in np.where(mask)[0]:
            l, h = divmod(idx, 12)
            ax.annotate(
                f"{l}.{h}", (X2[idx, 0], X2[idx, 1]),
                fontsize=5, ha="center", va="bottom",
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Head Clusters (PCA)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → cluster_scatter.png")


def plot_correlation_matrix(profiles, output_dir):
    """Correlation between the different head-level metrics."""
    keys = [k for k in ["entropy", "distance", "activation_mag",
                         "cls_query_entropy", "importance"] if k in profiles]
    if len(keys) < 2:
        return
    mat = np.column_stack([profiles[k].flatten() for k in keys])
    corr = np.corrcoef(mat.T)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(keys, fontsize=9)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Metric Correlation")
    plt.tight_layout()
    fig.savefig(output_dir / "metric_correlation.png", dpi=150, bbox_inches="tight")
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
    for l in range(NL):
        row_names = [name_map[c] for c in grid[l]]
        dist = Counter(row_names)
        summary = "  ".join(f"{g}: {n}" for g, n in sorted(dist.items()))
        print(f"  Layer {l:2d} │ {summary}")


# ---------------------------------------------------------------------- #
#  CLI                                                                     #
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Visualise attention head profiles")
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
