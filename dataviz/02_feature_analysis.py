#!/usr/bin/env python3
"""
Feature Engineering Visualizations

Visualize the feature engineering process and resulting modeling table.

Usage:
    python dataviz/02_feature_analysis.py --input data/events.parquet --output outputs/figures/features
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

plt.style.use("seaborn-v0_8-whitegrid")


def plot_event_count_distribution(modeling_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot distribution of event counts (target variable)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    y = modeling_df["event_count"]

    # Histogram
    ax1 = axes[0, 0]
    ax1.hist(y, bins=50, color="steelblue", alpha=0.7, edgecolor="white", density=True)
    ax1.set_xlabel("Event Count")
    ax1.set_ylabel("Density")
    ax1.set_title("Event Count Distribution")

    # Add theoretical Poisson and NegBinom fits
    x_range = np.arange(0, y.max() + 1)
    poisson_fit = stats.poisson.pmf(x_range, y.mean())
    ax1.plot(x_range, poisson_fit, "r-", lw=2, label=f"Poisson(λ={y.mean():.1f})")
    ax1.legend()

    # Log scale histogram
    ax2 = axes[0, 1]
    ax2.hist(y, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Event Count")
    ax2.set_ylabel("Frequency (log)")
    ax2.set_title("Event Count Distribution (Log Scale)")
    ax2.set_yscale("log")

    # By class
    ax3 = axes[1, 0]
    benign_counts = modeling_df[~modeling_df["has_attack"]]["event_count"]
    attack_counts = modeling_df[modeling_df["has_attack"]]["event_count"]

    ax3.hist(benign_counts, bins=50, alpha=0.7, label=f"Benign (n={len(benign_counts)})", density=True)
    ax3.hist(attack_counts, bins=50, alpha=0.7, label=f"Attack (n={len(attack_counts)})", color="crimson", density=True)
    ax3.set_xlabel("Event Count")
    ax3.set_ylabel("Density")
    ax3.set_title("Event Count by Class")
    ax3.legend()

    # Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = [benign_counts, attack_counts]
    bp = ax4.boxplot(data_to_plot, labels=["Benign", "Attack"], patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("crimson")
    ax4.set_ylabel("Event Count")
    ax4.set_title("Event Count by Class (Box Plot)")

    # Add statistics
    stats_text = f"Benign: μ={benign_counts.mean():.1f}, σ={benign_counts.std():.1f}\n"
    stats_text += f"Attack: μ={attack_counts.mean():.1f}, σ={attack_counts.std():.1f}"
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "01_event_count_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_event_count_distribution.png")


def plot_overdispersion_analysis(modeling_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze overdispersion in the data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Entity-level mean vs variance
    entity_stats = modeling_df.groupby("entity_idx")["event_count"].agg(["mean", "var"])
    entity_stats = entity_stats[entity_stats["var"].notna()]  # Remove entities with single observation

    ax1 = axes[0]
    ax1.scatter(entity_stats["mean"], entity_stats["var"], alpha=0.5, s=30)

    # Add Poisson line (var = mean)
    max_val = max(entity_stats["mean"].max(), entity_stats["var"].max())
    ax1.plot([0, max_val], [0, max_val], "r--", lw=2, label="Poisson (var = mean)")

    # Add linear fit
    slope, intercept, r_value, _, _ = stats.linregress(entity_stats["mean"], entity_stats["var"])
    x_fit = np.linspace(0, entity_stats["mean"].max(), 100)
    ax1.plot(x_fit, slope * x_fit + intercept, "g-", lw=2,
             label=f"Fit: var = {slope:.1f}×mean + {intercept:.1f} (R²={r_value**2:.2f})")

    ax1.set_xlabel("Mean Event Count")
    ax1.set_ylabel("Variance")
    ax1.set_title("Mean-Variance Relationship by Entity\n(Overdispersion Check)")
    ax1.legend()

    # Variance-to-mean ratio distribution
    ax2 = axes[1]
    vmr = entity_stats["var"] / entity_stats["mean"]
    vmr = vmr[vmr.notna() & np.isfinite(vmr)]

    ax2.hist(vmr, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(1, color="red", linestyle="--", lw=2, label="Poisson (VMR=1)")
    ax2.axvline(vmr.median(), color="green", linestyle="-", lw=2, label=f"Median VMR={vmr.median():.1f}")
    ax2.set_xlabel("Variance-to-Mean Ratio")
    ax2.set_ylabel("Number of Entities")
    ax2.set_title("Overdispersion: Variance-to-Mean Ratio\n(VMR > 1 indicates overdispersion)")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "02_overdispersion_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_overdispersion_analysis.png")


def plot_temporal_features(modeling_df: pd.DataFrame, output_dir: Path) -> None:
    """Visualize temporal features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hour distribution
    if "hour" in modeling_df.columns:
        ax1 = axes[0, 0]
        hour_attack_rate = modeling_df.groupby("hour")["has_attack"].mean()
        hour_counts = modeling_df.groupby("hour").size()

        ax1_twin = ax1.twinx()
        ax1.bar(hour_counts.index, hour_counts.values, alpha=0.5, color="steelblue", label="Count")
        ax1_twin.plot(hour_attack_rate.index, hour_attack_rate.values, "ro-", lw=2, label="Attack Rate")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Window Count", color="steelblue")
        ax1_twin.set_ylabel("Attack Rate", color="red")
        ax1.set_title("Hourly Distribution and Attack Rate")
        ax1.set_xticks(range(0, 24, 2))

    # Day of week distribution
    if "day_of_week" in modeling_df.columns:
        ax2 = axes[0, 1]
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_attack_rate = modeling_df.groupby("day_of_week")["has_attack"].mean()
        dow_counts = modeling_df.groupby("day_of_week").size()

        ax2_twin = ax2.twinx()
        ax2.bar(dow_counts.index, dow_counts.values, alpha=0.5, color="steelblue")
        ax2_twin.plot(dow_attack_rate.index, dow_attack_rate.values, "ro-", lw=2)
        ax2.set_xlabel("Day of Week")
        ax2.set_ylabel("Window Count", color="steelblue")
        ax2_twin.set_ylabel("Attack Rate", color="red")
        ax2.set_title("Day of Week Distribution and Attack Rate")
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(dow_names)

    # Business hours vs non-business
    if "is_business_hours" in modeling_df.columns:
        ax3 = axes[1, 0]
        bh_stats = modeling_df.groupby("is_business_hours").agg({
            "event_count": "mean",
            "has_attack": "mean"
        })

        x = [0, 1]
        width = 0.35
        ax3.bar([i - width/2 for i in x], bh_stats["event_count"], width, label="Avg Event Count", alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.bar([i + width/2 for i in x], bh_stats["has_attack"], width, label="Attack Rate", color="crimson", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(["Non-Business", "Business Hours"])
        ax3.set_ylabel("Avg Event Count", color="steelblue")
        ax3_twin.set_ylabel("Attack Rate", color="crimson")
        ax3.set_title("Business Hours vs Non-Business Hours")

    # Weekend vs weekday
    if "is_weekend" in modeling_df.columns:
        ax4 = axes[1, 1]
        weekend_stats = modeling_df.groupby("is_weekend").agg({
            "event_count": ["mean", "std"],
            "has_attack": "mean"
        })

        x = [0, 1]
        ax4.bar(x, weekend_stats[("event_count", "mean")], yerr=weekend_stats[("event_count", "std")],
                alpha=0.7, capsize=5, color="steelblue")
        ax4.set_xticks(x)
        ax4.set_xticklabels(["Weekday", "Weekend"])
        ax4.set_ylabel("Mean Event Count (±std)")
        ax4.set_title("Weekday vs Weekend Event Counts")

        # Add attack rates as text
        for i, rate in enumerate(weekend_stats[("has_attack", "mean")]):
            ax4.text(i, weekend_stats[("event_count", "mean")].iloc[i] + weekend_stats[("event_count", "std")].iloc[i],
                     f"Attack: {rate:.1%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "03_temporal_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_temporal_features.png")


def plot_feature_correlations(modeling_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot feature correlation matrix."""
    feature_cols = [
        "event_count", "unique_ips", "unique_endpoints", "unique_devices",
        "unique_locations", "failed_count", "bytes_total", "has_attack"
    ]
    feature_cols = [c for c in feature_cols if c in modeling_df.columns]

    fig, ax = plt.subplots(figsize=(10, 8))

    corr_matrix = modeling_df[feature_cols].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    fig.savefig(output_dir / "04_feature_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 04_feature_correlations.png")


def plot_entity_statistics(modeling_df: pd.DataFrame, output_dir: Path) -> None:
    """Visualize entity-level statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Entity mean distribution
    if "entity_mean_count" in modeling_df.columns:
        entity_means = modeling_df.groupby("entity_idx")["entity_mean_count"].first()

        ax1 = axes[0, 0]
        ax1.hist(entity_means, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax1.axvline(entity_means.mean(), color="red", linestyle="--", label=f"Mean: {entity_means.mean():.1f}")
        ax1.set_xlabel("Entity Mean Event Count")
        ax1.set_ylabel("Number of Entities")
        ax1.set_title("Distribution of Entity Baseline Rates")
        ax1.legend()

    # Z-score distribution
    if "count_zscore" in modeling_df.columns:
        ax2 = axes[0, 1]
        benign_z = modeling_df[~modeling_df["has_attack"]]["count_zscore"]
        attack_z = modeling_df[modeling_df["has_attack"]]["count_zscore"]

        ax2.hist(benign_z, bins=50, alpha=0.6, label="Benign", density=True)
        ax2.hist(attack_z, bins=50, alpha=0.6, label="Attack", color="crimson", density=True)
        ax2.axvline(0, color="black", linestyle="--", lw=1)
        ax2.set_xlabel("Z-Score (vs Entity Mean)")
        ax2.set_ylabel("Density")
        ax2.set_title("Z-Score Distribution by Class")
        ax2.legend()

    # Windows per entity
    windows_per_entity = modeling_df.groupby("entity_idx").size()

    ax3 = axes[1, 0]
    ax3.hist(windows_per_entity, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax3.set_xlabel("Number of Windows")
    ax3.set_ylabel("Number of Entities")
    ax3.set_title("Windows per Entity Distribution")

    # Entity heterogeneity
    ax4 = axes[1, 1]
    entity_cv = modeling_df.groupby("entity_idx")["event_count"].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
    entity_cv = entity_cv[entity_cv > 0]

    ax4.hist(entity_cv, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax4.axvline(entity_cv.median(), color="red", linestyle="--", label=f"Median CV: {entity_cv.median():.2f}")
    ax4.set_xlabel("Coefficient of Variation (CV)")
    ax4.set_ylabel("Number of Entities")
    ax4.set_title("Entity Variability (CV = σ/μ)")
    ax4.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "05_entity_statistics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_entity_statistics.png")


def plot_attack_feature_comparison(modeling_df: pd.DataFrame, output_dir: Path) -> None:
    """Compare features between attack and benign windows."""
    feature_cols = ["event_count", "unique_ips", "unique_endpoints", "unique_devices", "failed_count"]
    feature_cols = [c for c in feature_cols if c in modeling_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]

        benign_vals = modeling_df[~modeling_df["has_attack"]][col]
        attack_vals = modeling_df[modeling_df["has_attack"]][col]

        # Violin plot
        data = [benign_vals, attack_vals]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)

        parts["bodies"][0].set_facecolor("steelblue")
        parts["bodies"][1].set_facecolor("crimson")
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Attack"])
        ax.set_ylabel(col)
        ax.set_title(f"{col}\nBenign μ={benign_vals.mean():.1f}, Attack μ={attack_vals.mean():.1f}")

    # Hide empty subplot
    if len(feature_cols) < 6:
        axes[-1].axis("off")

    plt.suptitle("Feature Distributions: Attack vs Benign Windows", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "06_attack_feature_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_attack_feature_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Visualizations")
    parser.add_argument("--input", "-i", type=str, default="data/events.parquet", help="Input events file")
    parser.add_argument("--output", "-o", type=str, default="outputs/figures/features", help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Load data and build modeling table
    print(f"Loading data from {args.input}...")
    events_df = pd.read_parquet(args.input)

    print("Building modeling table...")
    from bsad.features import build_modeling_table
    modeling_df, metadata = build_modeling_table(events_df)

    print(f"  Entities: {metadata['n_entities']}")
    print(f"  Windows: {metadata['n_windows']}")
    print(f"  Attack rate: {metadata['attack_rate']:.2%}\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # Generate all plots
    plot_event_count_distribution(modeling_df, output_dir)
    plot_overdispersion_analysis(modeling_df, output_dir)
    plot_temporal_features(modeling_df, output_dir)
    plot_feature_correlations(modeling_df, output_dir)
    plot_entity_statistics(modeling_df, output_dir)
    plot_attack_feature_comparison(modeling_df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
