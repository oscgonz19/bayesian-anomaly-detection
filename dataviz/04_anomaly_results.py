#!/usr/bin/env python3
"""
Anomaly Detection Results Visualizations

Visualize anomaly scores, rankings, and detection performance.

Usage:
    python dataviz/04_anomaly_results.py --scores outputs/scores.parquet --output outputs/figures/results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")


def plot_score_distribution(scored_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot anomaly score distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    benign_scores = scored_df[~scored_df["has_attack"]]["anomaly_score"]
    attack_scores = scored_df[scored_df["has_attack"]]["anomaly_score"]

    # Overlapping histograms
    ax1 = axes[0, 0]
    ax1.hist(benign_scores, bins=50, alpha=0.7, label=f"Benign (n={len(benign_scores):,})",
             density=True, color="steelblue", edgecolor="white")
    ax1.hist(attack_scores, bins=50, alpha=0.7, label=f"Attack (n={len(attack_scores):,})",
             density=True, color="crimson", edgecolor="white")
    ax1.set_xlabel("Anomaly Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Anomaly Score Distribution by Class")
    ax1.legend()

    # Cumulative distribution
    ax2 = axes[0, 1]
    ax2.hist(benign_scores, bins=100, alpha=0.7, label="Benign", density=True, cumulative=True, histtype="step", lw=2)
    ax2.hist(attack_scores, bins=100, alpha=0.7, label="Attack", density=True, cumulative=True, histtype="step", lw=2, color="crimson")
    ax2.set_xlabel("Anomaly Score")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Distribution")
    ax2.legend()

    # Box plot
    ax3 = axes[1, 0]
    data_to_plot = [benign_scores, attack_scores]
    bp = ax3.boxplot(data_to_plot, labels=["Benign", "Attack"], patch_artist=True, notch=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("crimson")
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax3.set_ylabel("Anomaly Score")
    ax3.set_title("Score Distribution (Box Plot)")

    # Add statistics
    stats_text = f"Benign: μ={benign_scores.mean():.2f}, med={benign_scores.median():.2f}\n"
    stats_text += f"Attack: μ={attack_scores.mean():.2f}, med={attack_scores.median():.2f}\n"
    stats_text += f"Separation: {attack_scores.mean() - benign_scores.mean():.2f}"
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=9)

    # Violin plot
    ax4 = axes[1, 1]
    plot_data = pd.DataFrame({
        "Score": pd.concat([benign_scores, attack_scores]),
        "Class": ["Benign"] * len(benign_scores) + ["Attack"] * len(attack_scores)
    })
    sns.violinplot(data=plot_data, x="Class", y="Score", ax=ax4, palette=["steelblue", "crimson"])
    ax4.set_ylabel("Anomaly Score")
    ax4.set_title("Score Distribution (Violin Plot)")

    plt.tight_layout()
    fig.savefig(output_dir / "01_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_score_distribution.png")


def plot_top_anomalies(scored_df: pd.DataFrame, output_dir: Path, n: int = 30) -> None:
    """Plot top anomalies with uncertainty."""
    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.35)))

    top_df = scored_df.head(n).copy()
    top_df = top_df.sort_values("anomaly_score", ascending=True)

    y_pos = np.arange(len(top_df))
    colors = ["crimson" if attack else "steelblue" for attack in top_df["has_attack"]]

    # Error bars
    xerr_lower = top_df["anomaly_score"] - top_df["score_lower"]
    xerr_upper = top_df["score_upper"] - top_df["anomaly_score"]

    ax.barh(y_pos, top_df["anomaly_score"], xerr=[xerr_lower, xerr_upper],
            color=colors, alpha=0.7, capsize=3, ecolor="gray")

    # Labels
    labels = []
    for _, row in top_df.iterrows():
        entity = row.get("user_id", f"Entity {row.get('entity_idx', '?')}")
        window = str(row["window"])[:10] if hasattr(row["window"], "__str__") else ""
        count = int(row["event_count"])
        labels.append(f"{entity} | {window} | n={count}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Anomaly Score (90% CI)")
    ax.set_title(f"Top {n} Anomalies with Uncertainty Intervals")

    # Legend
    legend_elements = [
        Patch(facecolor="crimson", alpha=0.7, label="Attack (ground truth)"),
        Patch(facecolor="steelblue", alpha=0.7, label="Benign (ground truth)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_dir / "02_top_anomalies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_top_anomalies.png")


def plot_score_vs_count(scored_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot anomaly score vs event count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1 = axes[0]
    benign = scored_df[~scored_df["has_attack"]]
    attack = scored_df[scored_df["has_attack"]]

    ax1.scatter(benign["event_count"], benign["anomaly_score"], alpha=0.3, s=10, label="Benign")
    ax1.scatter(attack["event_count"], attack["anomaly_score"], alpha=0.7, s=30, color="crimson", label="Attack", marker="x")
    ax1.set_xlabel("Event Count")
    ax1.set_ylabel("Anomaly Score")
    ax1.set_title("Anomaly Score vs Event Count")
    ax1.legend()

    # Hexbin for density
    ax2 = axes[1]
    hb = ax2.hexbin(scored_df["event_count"], scored_df["anomaly_score"], gridsize=30, cmap="Blues", mincnt=1)
    ax2.scatter(attack["event_count"], attack["anomaly_score"], alpha=0.8, s=30, color="crimson", marker="x", label="Attack")
    plt.colorbar(hb, ax=ax2, label="Count")
    ax2.set_xlabel("Event Count")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("Score vs Count Density (Attacks highlighted)")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "03_score_vs_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_score_vs_count.png")


def plot_attack_type_analysis(scored_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze scores by attack type."""
    attack_df = scored_df[scored_df["has_attack"]]

    if len(attack_df) == 0 or "attack_type" not in attack_df.columns:
        print("  Skipping attack type analysis (no data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Score distribution by attack type
    ax1 = axes[0, 0]
    attack_types = attack_df["attack_type"].unique()
    data_by_type = [attack_df[attack_df["attack_type"] == t]["anomaly_score"] for t in attack_types]

    bp = ax1.boxplot(data_by_type, labels=attack_types, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(attack_types)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel("Attack Type")
    ax1.set_ylabel("Anomaly Score")
    ax1.set_title("Score Distribution by Attack Type")
    ax1.tick_params(axis="x", rotation=45)

    # Mean score by attack type
    ax2 = axes[0, 1]
    mean_scores = attack_df.groupby("attack_type")["anomaly_score"].mean().sort_values(ascending=False)
    ax2.barh(range(len(mean_scores)), mean_scores.values, color=colors[:len(mean_scores)], alpha=0.7)
    ax2.set_yticks(range(len(mean_scores)))
    ax2.set_yticklabels(mean_scores.index)
    ax2.set_xlabel("Mean Anomaly Score")
    ax2.set_title("Mean Score by Attack Type")

    # Event count by attack type
    ax3 = axes[1, 0]
    count_by_type = [attack_df[attack_df["attack_type"] == t]["event_count"] for t in attack_types]
    bp2 = ax3.boxplot(count_by_type, labels=attack_types, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xlabel("Attack Type")
    ax3.set_ylabel("Event Count")
    ax3.set_title("Event Count by Attack Type")
    ax3.tick_params(axis="x", rotation=45)

    # Detection rate in top K by attack type
    ax4 = axes[1, 1]
    k_values = [25, 50, 100]
    detection_rates = {}

    for attack_type in attack_types:
        type_df = scored_df[scored_df["attack_type"] == attack_type]
        rates = []
        for k in k_values:
            top_k = scored_df.head(k)
            detected = len(top_k[top_k["attack_type"] == attack_type])
            total = len(type_df)
            rates.append(detected / total if total > 0 else 0)
        detection_rates[attack_type] = rates

    x = np.arange(len(k_values))
    width = 0.8 / len(attack_types)
    for i, (attack_type, rates) in enumerate(detection_rates.items()):
        ax4.bar(x + i * width, rates, width, label=attack_type, alpha=0.7)

    ax4.set_xticks(x + width * (len(attack_types) - 1) / 2)
    ax4.set_xticklabels([f"Top {k}" for k in k_values])
    ax4.set_ylabel("Detection Rate")
    ax4.set_title("Detection Rate by Attack Type")
    ax4.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "04_attack_type_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 04_attack_type_analysis.png")


def plot_prediction_intervals(scored_df: pd.DataFrame, output_dir: Path, n_examples: int = 12) -> None:
    """Plot prediction intervals for example entities."""
    if "predicted_mean" not in scored_df.columns:
        print("  Skipping prediction intervals (no data)")
        return

    # Select diverse examples
    attacks = scored_df[scored_df["has_attack"]].nlargest(n_examples // 2, "anomaly_score")
    benign_high = scored_df[~scored_df["has_attack"]].nlargest(n_examples // 4, "anomaly_score")
    benign_low = scored_df[~scored_df["has_attack"]].nsmallest(n_examples // 4, "anomaly_score")
    examples = pd.concat([attacks, benign_high, benign_low]).head(n_examples)

    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(examples.iterrows()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        pred_lower = row["predicted_lower"]
        pred_upper = row["predicted_upper"]
        pred_mean = row["predicted_mean"]
        actual = row["event_count"]

        # Plot interval
        ax.barh(0, pred_upper - pred_lower, left=pred_lower, height=0.4,
                color="lightblue", alpha=0.7, label="90% CI")
        ax.plot(pred_mean, 0, "b|", markersize=25, markeredgewidth=3, label="Predicted")

        # Actual observation
        color = "crimson" if row["has_attack"] else "green"
        marker = "X" if row["has_attack"] else "o"
        ax.plot(actual, 0, marker, color=color, markersize=15, label=f"Observed ({int(actual)})")

        entity_id = row.get("user_id", f"Entity {idx}")
        status = "ATTACK" if row["has_attack"] else "Benign"
        ax.set_title(f"{entity_id}\n{status} | Score: {row['anomaly_score']:.1f}", fontsize=10)
        ax.set_xlim([0, max(pred_upper * 1.3, actual * 1.3, 10)])
        ax.set_yticks([])
        ax.set_xlabel("Event Count")

        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    # Hide unused subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Posterior Predictive Intervals vs Observations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "05_prediction_intervals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_prediction_intervals.png")


def plot_ranking_analysis(scored_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze anomaly rankings."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cumulative attacks in top K
    ax1 = axes[0, 0]
    total_attacks = scored_df["has_attack"].sum()
    k_values = range(1, min(len(scored_df), 500) + 1)
    cumulative_attacks = [scored_df.head(k)["has_attack"].sum() for k in k_values]
    recall_at_k = [c / total_attacks for c in cumulative_attacks]

    ax1.plot(k_values, recall_at_k, lw=2, color="steelblue")
    ax1.fill_between(k_values, recall_at_k, alpha=0.3)
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.5)

    # Mark specific K values
    for k in [25, 50, 100, 200]:
        if k < len(k_values):
            ax1.axvline(k, color="gray", linestyle=":", alpha=0.5)
            ax1.text(k, recall_at_k[k-1], f"  K={k}: {recall_at_k[k-1]:.1%}", fontsize=8, va="bottom")

    ax1.set_xlabel("K (Top K)")
    ax1.set_ylabel("Recall")
    ax1.set_title("Cumulative Recall Curve")

    # Precision at K
    ax2 = axes[0, 1]
    precision_at_k = [scored_df.head(k)["has_attack"].sum() / k for k in k_values]
    ax2.plot(k_values, precision_at_k, lw=2, color="steelblue")
    ax2.fill_between(k_values, precision_at_k, alpha=0.3)
    ax2.axhline(scored_df["has_attack"].mean(), color="red", linestyle="--",
                label=f"Baseline: {scored_df['has_attack'].mean():.1%}")
    ax2.set_xlabel("K (Top K)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision at K")
    ax2.legend()

    # Rank distribution of attacks
    ax3 = axes[1, 0]
    attack_ranks = scored_df[scored_df["has_attack"]]["anomaly_rank"]
    ax3.hist(attack_ranks, bins=50, color="crimson", alpha=0.7, edgecolor="white")
    ax3.axvline(attack_ranks.median(), color="black", linestyle="--", lw=2,
                label=f"Median rank: {attack_ranks.median():.0f}")
    ax3.set_xlabel("Rank (1 = most anomalous)")
    ax3.set_ylabel("Number of Attacks")
    ax3.set_title("Rank Distribution of True Attacks")
    ax3.legend()

    # Score uncertainty vs rank
    ax4 = axes[1, 1]
    if "score_std" in scored_df.columns:
        ax4.scatter(scored_df["anomaly_rank"], scored_df["score_std"], alpha=0.3, s=10)
        ax4.scatter(scored_df[scored_df["has_attack"]]["anomaly_rank"],
                    scored_df[scored_df["has_attack"]]["score_std"],
                    alpha=0.7, s=30, color="crimson", label="Attack", marker="x")
        ax4.set_xlabel("Rank")
        ax4.set_ylabel("Score Uncertainty (std)")
        ax4.set_title("Uncertainty vs Rank")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No uncertainty data", ha="center", va="center", transform=ax4.transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / "06_ranking_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_ranking_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="Anomaly Results Visualizations")
    parser.add_argument("--scores", "-s", type=str, default="outputs/scores.parquet", help="Scores file")
    parser.add_argument("--output", "-o", type=str, default="outputs/figures/results", help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ANOMALY DETECTION RESULTS VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading scores from {args.scores}...")
    scored_df = pd.read_parquet(args.scores)
    print(f"  Loaded {len(scored_df):,} scored windows")
    print(f"  Attacks: {scored_df['has_attack'].sum():,} ({scored_df['has_attack'].mean():.2%})\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # Generate all plots
    plot_score_distribution(scored_df, output_dir)
    plot_top_anomalies(scored_df, output_dir)
    plot_score_vs_count(scored_df, output_dir)
    plot_attack_type_analysis(scored_df, output_dir)
    plot_prediction_intervals(scored_df, output_dir)
    plot_ranking_analysis(scored_df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
