#!/usr/bin/env python3
"""
Data Exploration Visualizations

Visualize raw security event data to understand patterns and distributions.

Usage:
    python dataviz/01_data_exploration.py --input data/events.parquet --output outputs/figures/exploration
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_events_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot event count over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Daily event counts
    daily_counts = df.groupby(df["timestamp"].dt.date).size()

    ax1 = axes[0]
    ax1.bar(range(len(daily_counts)), daily_counts.values, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Event Count")
    ax1.set_title("Daily Event Volume")

    # Separate by attack type
    ax2 = axes[1]
    daily_attacks = df[df["is_attack"]].groupby(df[df["is_attack"]]["timestamp"].dt.date).size()
    daily_benign = df[~df["is_attack"]].groupby(df[~df["is_attack"]]["timestamp"].dt.date).size()

    x = range(len(daily_counts))
    ax2.bar(x, daily_benign.reindex(daily_counts.index, fill_value=0), label="Benign", alpha=0.7)
    ax2.bar(x, daily_attacks.reindex(daily_counts.index, fill_value=0),
            bottom=daily_benign.reindex(daily_counts.index, fill_value=0), label="Attack", alpha=0.7, color="crimson")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Event Count")
    ax2.set_title("Daily Events by Class")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "01_events_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_events_timeline.png")


def plot_hourly_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot hourly distribution of events."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df["hour"] = df["timestamp"].dt.hour

    # Overall hourly distribution
    ax1 = axes[0]
    hourly_counts = df.groupby("hour").size()
    ax1.bar(hourly_counts.index, hourly_counts.values, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Event Count")
    ax1.set_title("Hourly Event Distribution (All Events)")
    ax1.set_xticks(range(0, 24, 2))

    # By class
    ax2 = axes[1]
    hourly_benign = df[~df["is_attack"]].groupby("hour").size()
    hourly_attack = df[df["is_attack"]].groupby("hour").size()

    width = 0.35
    x = np.arange(24)
    ax2.bar(x - width/2, hourly_benign.reindex(range(24), fill_value=0), width, label="Benign", alpha=0.7)
    ax2.bar(x + width/2, hourly_attack.reindex(range(24), fill_value=0), width, label="Attack", alpha=0.7, color="crimson")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Event Count")
    ax2.set_title("Hourly Distribution by Class")
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "02_hourly_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_hourly_distribution.png")


def plot_user_activity(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot user activity distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Events per user
    user_counts = df.groupby("user_id").size()

    ax1 = axes[0, 0]
    ax1.hist(user_counts, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Events per User")
    ax1.set_ylabel("Number of Users")
    ax1.set_title("Distribution of Events per User")
    ax1.axvline(user_counts.mean(), color="red", linestyle="--", label=f"Mean: {user_counts.mean():.1f}")
    ax1.legend()

    # Log scale
    ax2 = axes[0, 1]
    ax2.hist(user_counts, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Events per User")
    ax2.set_ylabel("Number of Users (log)")
    ax2.set_title("Distribution of Events per User (Log Scale)")
    ax2.set_yscale("log")

    # Top 20 users
    ax3 = axes[1, 0]
    top_users = user_counts.nlargest(20)
    colors = ["crimson" if df[df["user_id"] == u]["is_attack"].any() else "steelblue" for u in top_users.index]
    ax3.barh(range(len(top_users)), top_users.values, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_users)))
    ax3.set_yticklabels(top_users.index, fontsize=8)
    ax3.set_xlabel("Event Count")
    ax3.set_title("Top 20 Users by Event Count (Red = Has Attack)")
    ax3.invert_yaxis()

    # Attack rate per user
    ax4 = axes[1, 1]
    user_attack_rate = df.groupby("user_id")["is_attack"].mean()
    ax4.hist(user_attack_rate[user_attack_rate > 0], bins=30, color="crimson", alpha=0.7, edgecolor="white")
    ax4.set_xlabel("Attack Rate (fraction of events)")
    ax4.set_ylabel("Number of Users")
    ax4.set_title("Attack Rate Distribution (Users with Attacks Only)")

    plt.tight_layout()
    fig.savefig(output_dir / "03_user_activity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_user_activity.png")


def plot_attack_patterns(df: pd.DataFrame, output_dir: Path) -> None:
    """Visualize attack pattern characteristics."""
    attack_df = df[df["is_attack"]]

    if len(attack_df) == 0:
        print("  No attacks found, skipping attack patterns plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Attack type distribution
    ax1 = axes[0, 0]
    attack_types = attack_df["attack_type"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(attack_types)))
    wedges, texts, autotexts = ax1.pie(
        attack_types.values,
        labels=attack_types.index,
        autopct="%1.1f%%",
        colors=colors,
        explode=[0.02] * len(attack_types),
    )
    ax1.set_title("Attack Type Distribution")

    # Events per attack type
    ax2 = axes[0, 1]
    ax2.bar(attack_types.index, attack_types.values, color=colors, alpha=0.8)
    ax2.set_xlabel("Attack Type")
    ax2.set_ylabel("Number of Events")
    ax2.set_title("Event Count by Attack Type")
    ax2.tick_params(axis="x", rotation=45)

    # Status code distribution in attacks
    ax3 = axes[1, 0]
    status_counts = attack_df["status_code"].value_counts().sort_index()
    colors_status = ["green" if s == 200 else "orange" if s < 500 else "red" for s in status_counts.index]
    ax3.bar(status_counts.index.astype(str), status_counts.values, color=colors_status, alpha=0.7)
    ax3.set_xlabel("HTTP Status Code")
    ax3.set_ylabel("Count")
    ax3.set_title("Status Code Distribution in Attack Events")

    # Location distribution in attacks
    ax4 = axes[1, 1]
    location_counts = attack_df["location"].value_counts()
    ax4.barh(range(len(location_counts)), location_counts.values, color="crimson", alpha=0.7)
    ax4.set_yticks(range(len(location_counts)))
    ax4.set_yticklabels(location_counts.index, fontsize=9)
    ax4.set_xlabel("Event Count")
    ax4.set_title("Attack Events by Location")
    ax4.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "04_attack_patterns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 04_attack_patterns.png")


def plot_ip_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze IP address patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Events per IP
    ip_counts = df.groupby("ip_address").size()

    ax1 = axes[0, 0]
    ax1.hist(ip_counts, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Events per IP")
    ax1.set_ylabel("Number of IPs")
    ax1.set_title("Distribution of Events per IP Address")
    ax1.set_yscale("log")

    # Users per IP
    users_per_ip = df.groupby("ip_address")["user_id"].nunique()

    ax2 = axes[0, 1]
    ax2.hist(users_per_ip, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Unique Users per IP")
    ax2.set_ylabel("Number of IPs")
    ax2.set_title("User Count Distribution per IP")

    # Top attack IPs
    ax3 = axes[1, 0]
    attack_ip_counts = df[df["is_attack"]].groupby("ip_address").size().nlargest(15)
    if len(attack_ip_counts) > 0:
        ax3.barh(range(len(attack_ip_counts)), attack_ip_counts.values, color="crimson", alpha=0.7)
        ax3.set_yticks(range(len(attack_ip_counts)))
        ax3.set_yticklabels(attack_ip_counts.index, fontsize=8)
        ax3.set_xlabel("Attack Event Count")
        ax3.set_title("Top 15 IPs by Attack Events")
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, "No Attack IPs", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Top Attack IPs")

    # IP-User relationship (attack vs benign)
    ax4 = axes[1, 1]
    ip_user_attack = df[df["is_attack"]].groupby("ip_address")["user_id"].nunique()
    ip_user_benign = df[~df["is_attack"]].groupby("ip_address")["user_id"].nunique()

    ax4.scatter(
        ip_user_benign.values,
        ip_counts.reindex(ip_user_benign.index, fill_value=0).values,
        alpha=0.5, label="Benign IPs", s=20
    )
    if len(ip_user_attack) > 0:
        ax4.scatter(
            ip_user_attack.values,
            ip_counts.reindex(ip_user_attack.index, fill_value=0).values,
            alpha=0.7, color="crimson", label="Attack IPs", s=50, marker="x"
        )
    ax4.set_xlabel("Unique Users Accessed")
    ax4.set_ylabel("Total Events")
    ax4.set_title("IP Activity: Users vs Events")
    ax4.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "05_ip_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_ip_analysis.png")


def plot_endpoint_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze endpoint access patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Endpoint distribution
    endpoint_counts = df["endpoint"].value_counts().head(15)

    ax1 = axes[0]
    colors = ["crimson" if df[df["endpoint"] == e]["is_attack"].any() else "steelblue" for e in endpoint_counts.index]
    ax1.barh(range(len(endpoint_counts)), endpoint_counts.values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(endpoint_counts)))
    ax1.set_yticklabels(endpoint_counts.index, fontsize=9)
    ax1.set_xlabel("Event Count")
    ax1.set_title("Top 15 Endpoints (Red = Contains Attacks)")
    ax1.invert_yaxis()

    # Attack rate by endpoint
    ax2 = axes[1]
    endpoint_attack_rate = df.groupby("endpoint")["is_attack"].mean().sort_values(ascending=False).head(15)
    ax2.barh(range(len(endpoint_attack_rate)), endpoint_attack_rate.values, color="crimson", alpha=0.7)
    ax2.set_yticks(range(len(endpoint_attack_rate)))
    ax2.set_yticklabels(endpoint_attack_rate.index, fontsize=9)
    ax2.set_xlabel("Attack Rate (fraction)")
    ax2.set_title("Top 15 Endpoints by Attack Rate")
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "06_endpoint_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_endpoint_analysis.png")


def plot_summary_stats(df: pd.DataFrame, output_dir: Path) -> None:
    """Create summary statistics visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    stats = {
        "Total Events": f"{len(df):,}",
        "Unique Users": f"{df['user_id'].nunique():,}",
        "Unique IPs": f"{df['ip_address'].nunique():,}",
        "Unique Endpoints": f"{df['endpoint'].nunique():,}",
        "Date Range": f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
        "": "",
        "Attack Events": f"{df['is_attack'].sum():,}",
        "Attack Rate": f"{df['is_attack'].mean():.2%}",
        "Attack Types": f"{df[df['is_attack']]['attack_type'].nunique()}",
        " ": "",
        "Avg Events/User": f"{len(df) / df['user_id'].nunique():.1f}",
        "Avg Events/IP": f"{len(df) / df['ip_address'].nunique():.1f}",
        "Avg Events/Day": f"{len(df) / df['timestamp'].dt.date.nunique():.1f}",
    }

    ax.axis("off")
    table_data = [[k, v] for k, v in stats.items()]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.4, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Data Summary Statistics", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / "00_summary_stats.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: 00_summary_stats.png")


def main():
    parser = argparse.ArgumentParser(description="Data Exploration Visualizations")
    parser.add_argument("--input", "-i", type=str, default="data/events.parquet", help="Input events file")
    parser.add_argument("--output", "-o", type=str, default="outputs/figures/exploration", help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("DATA EXPLORATION VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Loaded {len(df):,} events\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # Generate all plots
    plot_summary_stats(df, output_dir)
    plot_events_timeline(df, output_dir)
    plot_hourly_distribution(df, output_dir)
    plot_user_activity(df, output_dir)
    plot_attack_patterns(df, output_dir)
    plot_ip_analysis(df, output_dir)
    plot_endpoint_analysis(df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
