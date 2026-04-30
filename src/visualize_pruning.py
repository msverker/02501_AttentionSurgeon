import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", context="paper", font_scale=1.2)

STRATEGY_STYLES = {
    "Random": {"color": "#95A5A6", "linestyle": "--", "zorder": 1},
    "Magnitude": {"color": "#E67E22", "linestyle": "-", "zorder": 2},
    "Importance": {"color": "#2980B9", "linestyle": "-", "zorder": 3},
    "RL Agent": {"color": "#E74C3C", "linestyle": "-", "zorder": 4},
}


def load_results(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def plot_accuracy_vs_pruned(results, output_dir, baseline_acc, dataset_name, metric_name="Top-1 Accuracy"):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    # n_steps = len(results["random_means"])
    n_steps = len(results['rl_means'])
    x = np.arange(1, n_steps + 1)
    pct = x / 144 * 100  # pruning percentage

    # # random — with std band
    # rand_acc = results["random_means"][:, 0]
    # rand_std = results["random_stds"][:, 0]
    # ax.plot(
    #     pct,
    #     rand_acc,
    #     label="Random",
    #     **{k: v for k, v in STRATEGY_STYLES["Random"].items() if k != "zorder"},
    #     zorder=STRATEGY_STYLES["Random"]["zorder"],
    #     linewidth=1.5,
    # )
    # ax.fill_between(
    #     pct,
    #     rand_acc - rand_std,
    #     rand_acc + rand_std,
    #     color=STRATEGY_STYLES["Random"]["color"],
    #     alpha=0.15,
    # )

    # # magnitude
    # mag_acc = results["magnitude_means"][:, 0]
    # ax.plot(
    #     pct,
    #     mag_acc,
    #     label="Magnitude",
    #     **{k: v for k, v in STRATEGY_STYLES["Magnitude"].items() if k != "zorder"},
    #     zorder=STRATEGY_STYLES["Magnitude"]["zorder"],
    #     linewidth=1.5,
    # )

    # # importance
    # imp_acc = results["importance_means"][:, 0]
    # ax.plot(
    #     pct,
    #     imp_acc,
    #     label="Importance",
    #     **{k: v for k, v in STRATEGY_STYLES["Importance"].items() if k != "zorder"},
    #     zorder=STRATEGY_STYLES["Importance"]["zorder"],
    #     linewidth=1.5,
    # )

    # RL agent placeholder — add when available
    if "rl_means" in results:
        rl_acc = results["rl_means"][:, 0]
        rl_std = results["rl_stds"][:, 0]
        ax.plot(
            pct,
            rl_acc,
            label="RL Agent",
            **{k: v for k, v in STRATEGY_STYLES["RL Agent"].items() if k != "zorder"},
            zorder=STRATEGY_STYLES["RL Agent"]["zorder"],
            linewidth=2,
        )
        ax.fill_between(
            pct,
            rl_acc - rl_std,
            rl_acc + rl_std,
            color=STRATEGY_STYLES["RL Agent"]["color"],
            alpha=0.15,
        )

    # baseline accuracy
    ax.axhline(
        baseline_acc,
        color="#2ECC71",
        linestyle=":",
        linewidth=1.2,
        label=f"No pruning ({baseline_acc:.3f})",
        zorder=0,
    )

    # 50% pruning marker
    ax.axvline(50, color="#DDDDDD", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(50.5, ax.get_ylim()[0] + 0.01, "50% pruned", fontsize=8, color="#AAAAAA")

    ax.set_xlabel("Heads pruned (%)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    
    title_dataset = dataset_name.upper() if dataset_name else "Model"
    ax.set_title(
        f"Pruning Strategy Comparison  ·  DINOv2 ViT-B/16  ·  {title_dataset}",
        fontsize=12,
        pad=14,
    )
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor="#DDDDDD")
    ax.set_xlim(0, 50)
    ax.set_ylim(bottom=None)

    sns.despine(ax=ax, offset=8, trim=True)
    plt.tight_layout()
    suffix = f"_{dataset_name.lower()}" if dataset_name else ""
    fig.savefig(
        output_dir / f"pruning_comparison{suffix}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  → pruning_comparison{suffix}.png")


def plot_reward_vs_pruned(results, output_dir, baseline_acc, dataset_name, metric_name="Top-1 Accuracy"):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    # n_steps = len(results["random_means"])
    n_steps = len(results['rl_means'])
    x = np.arange(1, n_steps + 1)
    pct = x / 144 * 100

    # rand_reward = results["random_means"][:, 2]
    # rand_std = results["random_stds"][:, 2]
    # ax.plot(
    #     pct,
    #     rand_reward,
    #     label="Random",
    #     **{k: v for k, v in STRATEGY_STYLES["Random"].items() if k != "zorder"},
    #     zorder=1,
    #     linewidth=1.5,
    # )
    # ax.fill_between(
    #     pct,
    #     rand_reward - rand_std,
    #     rand_reward + rand_std,
    #     color=STRATEGY_STYLES["Random"]["color"],
    #     alpha=0.15,
    # )

    # ax.plot(
    #     pct,
    #     results["magnitude_means"][:, 2],
    #     label="Magnitude",
    #     **{k: v for k, v in STRATEGY_STYLES["Magnitude"].items() if k != "zorder"},
    #     zorder=2,
    #     linewidth=1.5,
    # )

    # ax.plot(
    #     pct,
    #     results["importance_means"][:, 2],
    #     label="Importance",
    #     **{k: v for k, v in STRATEGY_STYLES["Importance"].items() if k != "zorder"},
    #     zorder=3,
    #     linewidth=1.5,
    # )

    if "rl_means" in results:
        rl_reward = results["rl_means"][:, 2]
        rl_std = results["rl_stds"][:, 2]
        ax.plot(
            pct,
            rl_reward,
            label="RL Agent",
            **{k: v for k, v in STRATEGY_STYLES["RL Agent"].items() if k != "zorder"},
            zorder=4,
            linewidth=2,
        )
        ax.fill_between(
            pct,
            rl_reward - rl_std,
            rl_reward + rl_std,
            color=STRATEGY_STYLES["RL Agent"]["color"],
            alpha=0.15,
        )

    ax.set_xlabel("Heads pruned (%)", fontsize=11)
    ax.set_ylabel(f"Reward  ({metric_name.lower()} × FLOPs ratio)", fontsize=11)
    title_dataset = dataset_name.upper() if dataset_name else "Model"
    ax.set_title(
        f"Pruning Reward Comparison  ·  DINOv2 ViT-B/16  ·  {title_dataset}",
        fontsize=12,
        pad=14,
    )
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor="#DDDDDD")
    ax.set_xlim(0, 50)

    sns.despine(ax=ax, offset=8, trim=True)
    plt.tight_layout()
    suffix = f"_{dataset_name.lower()}" if dataset_name else ""
    fig.savefig(
        output_dir / f"pruning_reward{suffix}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"  → pruning_reward{suffix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=str, default="results/baseline_pruning_imagenet_results.npz"
    )
    ap.add_argument("--output", type=str, default="figures")
    ap.add_argument("--dataset-name", type=str, default="ImageNet", help="Name of the dataset to include in plot titles and filenames")
    ap.add_argument("--metric-name", type=str, default="Top-1 Accuracy", help="Name of the metric for the y-axis")
    ap.add_argument("--no-rl", action="store_true", help="Do not plot RL Agent performance")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(exist_ok=True)

    results = load_results(args.input)
    
    if args.no_rl:
        results.pop("rl_means", None)
        results.pop("rl_stds", None)

    baseline_acc = float(results["baseline_acc"].item())
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    print(f"Saving figures to {out}/")
    plot_accuracy_vs_pruned(results, out, baseline_acc, args.dataset_name, args.metric_name)
    plot_reward_vs_pruned(results, out, baseline_acc, args.dataset_name, args.metric_name)


if __name__ == "__main__":
    main()