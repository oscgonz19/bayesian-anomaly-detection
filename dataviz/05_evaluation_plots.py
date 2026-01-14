#!/usr/bin/env python3
"""
Evaluation Metrics Visualizations

Visualize model evaluation metrics: PR-AUC, ROC-AUC, Recall@K, etc.

Usage:
    python dataviz/05_evaluation_plots.py --scores outputs/scores.parquet --output outputs/figures/evaluation
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score,
)

plt.style.use("seaborn-v0_8-whitegrid")


def plot_precision_recall_curve(y_true: np.ndarray, scores: np.ndarray, output_dir: Path) -> dict:
    """Plot precision-recall curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    # PR curve
    ax1 = axes[0]
    ax1.plot(recall, precision, lw=2, color="steelblue", label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax1.fill_between(recall, precision, alpha=0.3, color="steelblue")

    # Baseline (random classifier)
    baseline = y_true.mean()
    ax1.axhline(baseline, color="red", linestyle="--", lw=1.5, label=f"Baseline = {baseline:.3f}")

    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper right")

    # F1 score curve
    ax2 = axes[1]
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    valid_idx = np.where(thresholds < thresholds.max())[0]

    ax2.plot(thresholds[valid_idx], precision[valid_idx], lw=2, label="Precision", color="steelblue")
    ax2.plot(thresholds[valid_idx], recall[valid_idx], lw=2, label="Recall", color="green")
    ax2.plot(thresholds[valid_idx], f1_scores[valid_idx], lw=2, label="F1", color="purple")

    # Best F1 threshold
    best_f1_idx = np.argmax(f1_scores[valid_idx])
    best_threshold = thresholds[valid_idx][best_f1_idx]
    ax2.axvline(best_threshold, color="red", linestyle="--", lw=1.5,
                label=f"Best F1 @ {best_threshold:.2f}")

    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision, Recall, F1 vs Threshold")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "01_precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_precision_recall_curve.png")

    return {"pr_auc": pr_auc, "best_f1_threshold": float(best_threshold)}


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, output_dir: Path) -> dict:
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 8))

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.3, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Random Classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_dir / "02_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_roc_curve.png")

    return {"roc_auc": roc_auc}


def plot_recall_precision_at_k(y_true: np.ndarray, scores: np.ndarray, output_dir: Path) -> dict:
    """Plot recall and precision at various K values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_positives = y_true.sum()
    sorted_idx = np.argsort(scores)[::-1]
    y_sorted = y_true[sorted_idx]

    k_values = np.arange(1, min(len(y_true), 500) + 1)
    cumsum = np.cumsum(y_sorted)

    recall_at_k = cumsum[:len(k_values)] / n_positives
    precision_at_k = cumsum[:len(k_values)] / k_values

    # Recall@K
    ax1 = axes[0]
    ax1.plot(k_values, recall_at_k, lw=2, color="steelblue")
    ax1.fill_between(k_values, recall_at_k, alpha=0.3, color="steelblue")

    # Mark key points
    for k in [10, 25, 50, 100, 200]:
        if k <= len(k_values):
            ax1.plot(k, recall_at_k[k-1], "ro", markersize=8)
            ax1.annotate(f"K={k}: {recall_at_k[k-1]:.1%}", (k, recall_at_k[k-1]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_xlabel("K")
    ax1.set_ylabel("Recall")
    ax1.set_title("Recall@K: Fraction of Attacks in Top K")
    ax1.axhline(1.0, color="red", linestyle="--", alpha=0.5)

    # Precision@K
    ax2 = axes[1]
    ax2.plot(k_values, precision_at_k, lw=2, color="green")
    ax2.fill_between(k_values, precision_at_k, alpha=0.3, color="green")

    baseline = y_true.mean()
    ax2.axhline(baseline, color="red", linestyle="--", lw=1.5, label=f"Baseline = {baseline:.3f}")

    ax2.set_xlabel("K")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision@K: Fraction of Top K that are Attacks")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "03_recall_precision_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_recall_precision_at_k.png")

    # Return metrics
    metrics = {}
    for k in [10, 25, 50, 100]:
        if k <= len(k_values):
            metrics[f"recall_at_{k}"] = float(recall_at_k[k-1])
            metrics[f"precision_at_{k}"] = float(precision_at_k[k-1])

    return metrics


def plot_lift_curve(y_true: np.ndarray, scores: np.ndarray, output_dir: Path) -> None:
    """Plot lift curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n = len(y_true)
    n_positives = y_true.sum()
    baseline_rate = n_positives / n

    sorted_idx = np.argsort(scores)[::-1]
    y_sorted = y_true[sorted_idx]

    percentiles = np.arange(1, 101)
    lifts = []

    for p in percentiles:
        k = int(n * p / 100)
        if k == 0:
            k = 1
        precision_at_k = y_sorted[:k].sum() / k
        lift = precision_at_k / baseline_rate
        lifts.append(lift)

    ax.plot(percentiles, lifts, lw=2, color="steelblue")
    ax.fill_between(percentiles, lifts, 1, alpha=0.3, color="steelblue")
    ax.axhline(1, color="red", linestyle="--", lw=1.5, label="Baseline (Lift = 1)")

    ax.set_xlabel("Percentile of Predictions")
    ax.set_ylabel("Lift")
    ax.set_title("Lift Curve: How much better than random?")
    ax.legend()

    # Annotate max lift
    max_lift = max(lifts)
    max_lift_pct = percentiles[lifts.index(max_lift)]
    ax.annotate(f"Max Lift: {max_lift:.1f}x at {max_lift_pct}%",
                (max_lift_pct, max_lift), textcoords="offset points",
                xytext=(20, 10), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    fig.savefig(output_dir / "04_lift_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 04_lift_curve.png")


def plot_metrics_summary(metrics: dict, output_dir: Path) -> None:
    """Create summary visualization of all metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Primary metrics bar chart
    ax1 = axes[0]
    primary_metrics = {
        "PR-AUC": metrics.get("pr_auc", 0),
        "ROC-AUC": metrics.get("roc_auc", 0),
    }
    colors = ["steelblue", "green"]
    bars = ax1.bar(primary_metrics.keys(), primary_metrics.values(), color=colors, alpha=0.7)
    ax1.axhline(0.5, color="red", linestyle="--", lw=1.5, label="Random (0.5)")
    ax1.set_ylabel("Score")
    ax1.set_title("Primary Metrics")
    ax1.set_ylim([0, 1])
    ax1.legend()

    # Add value labels
    for bar, val in zip(bars, primary_metrics.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Recall@K bar chart
    ax2 = axes[1]
    recall_metrics = {k: v for k, v in metrics.items() if k.startswith("recall_at_")}
    if recall_metrics:
        k_labels = [k.replace("recall_at_", "K=") for k in recall_metrics.keys()]
        bars = ax2.bar(k_labels, recall_metrics.values(), color="purple", alpha=0.7)
        ax2.set_ylabel("Recall")
        ax2.set_title("Recall@K")
        ax2.set_ylim([0, 1])

        for bar, val in zip(bars, recall_metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "05_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_metrics_summary.png")


def plot_comparison_with_baselines(y_true: np.ndarray, scores: np.ndarray,
                                   event_counts: np.ndarray, output_dir: Path) -> None:
    """Compare our model with simple baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = {
        "BSAD (Our Model)": scores,
        "Event Count (Baseline)": event_counts,
        "Random": np.random.rand(len(y_true)),
    }

    # PR-AUC comparison
    ax1 = axes[0]
    pr_aucs = []
    for name, method_scores in methods.items():
        pr_auc = average_precision_score(y_true, method_scores)
        pr_aucs.append(pr_auc)

    colors = ["steelblue", "orange", "gray"]
    bars = ax1.bar(methods.keys(), pr_aucs, color=colors, alpha=0.7)
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("PR-AUC: Model Comparison")

    for bar, val in zip(bars, pr_aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    # Recall@100 comparison
    ax2 = axes[1]
    recalls = []
    n_positives = y_true.sum()
    for name, method_scores in methods.items():
        sorted_idx = np.argsort(method_scores)[::-1][:100]
        recall = y_true[sorted_idx].sum() / n_positives
        recalls.append(recall)

    bars = ax2.bar(methods.keys(), recalls, color=colors, alpha=0.7)
    ax2.set_ylabel("Recall@100")
    ax2.set_title("Recall@100: Model Comparison")

    for bar, val in zip(bars, recalls):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / "06_baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_baseline_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluation Metrics Visualizations")
    parser.add_argument("--scores", "-s", type=str, default="outputs/scores.parquet", help="Scores file")
    parser.add_argument("--output", "-o", type=str, default="outputs/figures/evaluation", help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EVALUATION METRICS VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Load data
    print(f"Loading scores from {args.scores}...")
    scored_df = pd.read_parquet(args.scores)

    y_true = scored_df["has_attack"].astype(int).values
    scores = scored_df["anomaly_score"].values
    event_counts = scored_df["event_count"].values

    print(f"  Total observations: {len(y_true):,}")
    print(f"  Attacks: {y_true.sum():,} ({y_true.mean():.2%})\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # Generate all plots and collect metrics
    all_metrics = {}
    all_metrics.update(plot_precision_recall_curve(y_true, scores, output_dir))
    all_metrics.update(plot_roc_curve(y_true, scores, output_dir))
    all_metrics.update(plot_recall_precision_at_k(y_true, scores, output_dir))
    plot_lift_curve(y_true, scores, output_dir)
    plot_metrics_summary(all_metrics, output_dir)
    plot_comparison_with_baselines(y_true, scores, event_counts, output_dir)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved metrics to: {metrics_path}")

    print(f"\nAll visualizations saved to: {output_dir}/")

    # Print summary
    print(f"\n{'='*40}")
    print("EVALUATION SUMMARY")
    print(f"{'='*40}")
    print(f"PR-AUC:      {all_metrics.get('pr_auc', 0):.3f}")
    print(f"ROC-AUC:     {all_metrics.get('roc_auc', 0):.3f}")
    print(f"Recall@50:   {all_metrics.get('recall_at_50', 0):.1%}")
    print(f"Recall@100:  {all_metrics.get('recall_at_100', 0):.1%}")


if __name__ == "__main__":
    main()
