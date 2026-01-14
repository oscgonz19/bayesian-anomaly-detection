#!/usr/bin/env python3
"""
Alert Prioritization: From Detection to Decision

This is the "star script" that demonstrates BSAD's operational value.
It transforms raw anomaly scores into actionable SOC workflows.

Key outputs:
1. Risk-ranked alert queue
2. Alert budget curves (recall vs workload)
3. Entity-enriched tickets
4. Comparison: BSAD vs baseline workload

Usage:
    python scripts/alert_prioritization.py                    # Use UNSW data
    python scripts/alert_prioritization.py --dataset cse-cic  # Use CSE-CIC data
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage import (
    compute_risk_score,
    RiskScorer,
    calibrate_threshold,
    AlertBudget,
    build_alert_budget_curve,
    precision_at_k,
    recall_at_k,
    fpr_at_fixed_recall,
    alerts_per_k_windows,
    workload_reduction,
    ranking_report,
    build_entity_history,
    enrich_alerts,
)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "triage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_scored_data(dataset: str = "unsw") -> pd.DataFrame:
    """Load pre-scored data from outputs."""
    if dataset == "unsw":
        path = PROJECT_ROOT / "outputs" / "datasets" / "unsw-nb15" / "rare-attack" / "scored_df_2pct.parquet"
    elif dataset == "cse-cic":
        # Use multi-regime results
        path = PROJECT_ROOT / "outputs" / "datasets" / "cse-cic-ids2018" / "dev" / "comparison_results.json"
        # For CSE-CIC, we need to reconstruct from the analysis
        print("Note: CSE-CIC data requires running cse_cic_ids2018_analysis.py first")
        return None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not path.exists():
        print(f"Data not found: {path}")
        print("Run the appropriate training script first.")
        return None

    return pd.read_parquet(path)


def create_triage_dashboard(
    df: pd.DataFrame,
    output_path: Path,
    dataset_name: str = "UNSW-NB15"
):
    """Create comprehensive triage dashboard."""

    y_true = df["has_attack"].astype(int).values
    scores = df["anomaly_score"].values

    # Compute risk scores
    risk_scores = compute_risk_score(
        df,
        score_col="anomaly_score",
        std_col="score_std" if "score_std" in df.columns else None,
        entity_col="entity" if "entity" in df.columns else None,
    )
    df["risk_score"] = risk_scores

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Alert Prioritization Dashboard: {dataset_name}\n' +
                 'From Detection to Decision',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Alert Budget Curve
    ax1 = fig.add_subplot(gs[0, 0])
    budget_curve = build_alert_budget_curve(scores, y_true)
    ax1.plot(budget_curve["actual_recall"] * 100, budget_curve["alerts"], 'o-',
             color='#2ecc71', linewidth=2, markersize=8)
    ax1.set_xlabel("Recall (%)")
    ax1.set_ylabel("Total Alerts")
    ax1.set_title("Alert Budget Curve\n(Alerts needed at each recall level)", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Mark key points
    for _, row in budget_curve.iterrows():
        if row["recall_target"] in [0.3, 0.5]:
            ax1.annotate(f'{int(row["alerts"])} alerts\n@{int(row["actual_recall"]*100)}% recall',
                        xy=(row["actual_recall"]*100, row["alerts"]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, ha='left')

    # 2. Precision@k and Recall@k
    ax2 = fig.add_subplot(gs[0, 1])
    ks = [10, 25, 50, 100, 200]
    prec_at_k = [precision_at_k(y_true, scores, k) for k in ks]
    rec_at_k = [recall_at_k(y_true, scores, k) for k in ks]

    x = np.arange(len(ks))
    width = 0.35
    ax2.bar(x - width/2, prec_at_k, width, label='Precision@k', color='#3498db')
    ax2.bar(x + width/2, rec_at_k, width, label='Recall@k', color='#e74c3c')
    ax2.set_xlabel('k (top alerts)')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision & Recall at Top-k\n(Quality of ranked list)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ks)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # 3. FPR at Fixed Recall
    ax3 = fig.add_subplot(gs[0, 2])
    recall_targets = [0.1, 0.2, 0.3, 0.4, 0.5]
    fprs = [fpr_at_fixed_recall(y_true, scores, r) for r in recall_targets]
    alerts = [alerts_per_k_windows(y_true, scores, r) for r in recall_targets]

    ax3.bar([f"{int(r*100)}%" for r in recall_targets], fprs, color='#9b59b6', alpha=0.7)
    ax3.set_xlabel('Target Recall')
    ax3.set_ylabel('False Positive Rate')
    ax3.set_title('FPR at Fixed Recall\n(Cost of catching attacks)', fontweight='bold')

    # Add alerts as text
    for i, (r, fpr, alert) in enumerate(zip(recall_targets, fprs, alerts)):
        ax3.annotate(f'{alert:.0f}\nalerts/1k',
                    xy=(i, fpr), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=7)

    # 4. Risk Score Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    attacks = df[df['has_attack'] == 1]['risk_score']
    benign = df[df['has_attack'] == 0]['risk_score']
    ax4.hist(benign, bins=30, alpha=0.6, label=f'Normal (n={len(benign)})',
             color='steelblue', density=True)
    ax4.hist(attacks, bins=30, alpha=0.6, label=f'Attack (n={len(attacks)})',
             color='crimson', density=True)
    ax4.set_xlabel('Risk Score')
    ax4.set_ylabel('Density')
    ax4.set_title('Risk Score Distribution\n(Separation quality)', fontweight='bold')
    ax4.legend()

    # 5. Top Entities by Alert Density
    ax5 = fig.add_subplot(gs[1, 1])
    if 'entity' in df.columns:
        entity_alerts = df.groupby('entity').agg({
            'anomaly_score': 'mean',
            'has_attack': 'sum',
        }).sort_values('anomaly_score', ascending=False).head(15)

        colors = ['crimson' if x > 0 else 'steelblue' for x in entity_alerts['has_attack']]
        ax5.barh(range(len(entity_alerts)), entity_alerts['anomaly_score'], color=colors, alpha=0.7)
        ax5.set_yticks(range(len(entity_alerts)))
        ax5.set_yticklabels([str(e)[:20] for e in entity_alerts.index], fontsize=8)
        ax5.set_xlabel('Mean Anomaly Score')
        ax5.set_title('Top 15 Entities by Alert Score\n(Red = has real attack)', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No entity data available', ha='center', va='center')
        ax5.set_title('Entity Analysis')

    # 6. Operational Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Compute key metrics
    attack_rate = y_true.mean()
    recall_30_fpr = fpr_at_fixed_recall(y_true, scores, 0.3)
    recall_30_alerts = alerts_per_k_windows(y_true, scores, 0.3)
    prec_10 = precision_at_k(y_true, scores, 10)
    rec_50 = recall_at_k(y_true, scores, 50)

    summary_text = f"""
OPERATIONAL SUMMARY
════════════════════════════════════

Dataset:        {dataset_name}
Attack Rate:    {attack_rate:.2%}
Observations:   {len(df):,}

AT 30% RECALL TARGET:
────────────────────
FPR:            {recall_30_fpr:.3f}
Alerts/1k:      {recall_30_alerts:.1f}

TOP-K PERFORMANCE:
────────────────────
Precision@10:   {prec_10:.2%}
Recall@50:      {rec_50:.2%}

RECOMMENDATION:
────────────────────
With {recall_30_alerts:.0f} alerts per 1,000 windows,
a SOC processing 10k windows/day would
review ~{recall_30_alerts*10:.0f} alerts to catch
30% of attacks.
    """
    ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 7. Sample Alert Tickets
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.axis('off')

    # Build entity history and enrich top alerts
    if 'entity' in df.columns and 'event_count' in df.columns:
        history = build_entity_history(df, entity_col='entity', value_col='event_count')
        enriched = enrich_alerts(df, history, top_k=5)

        ticket_text = "SAMPLE ALERT TICKETS (Top 5)\n" + "═" * 60 + "\n\n"
        for i, alert in enumerate(enriched[:5], 1):
            ticket_text += f"[Ticket #{i}] Entity: {alert['entity_id']}\n"
            ticket_text += f"  Score: {alert['anomaly_score']:.2f} | "
            ticket_text += f"Deviation: {alert['sigma_deviation']:.1f}σ | "
            ticket_text += f"Confidence: {alert['confidence']}\n"
            ticket_text += f"  History: {alert['historical_alerts']} prior alerts\n"
            ticket_text += "─" * 50 + "\n"
    else:
        ticket_text = "Entity context not available (missing entity/event_count columns)"

    ax7.text(0.02, 0.95, ticket_text, transform=ax7.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # 8. Key Takeaway
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    takeaway = f"""
KEY TAKEAWAY
════════════════════════════════════

This dashboard transforms raw
anomaly scores into actionable
SOC metrics.

Instead of asking:
  "Is this anomalous?"

We now ask:
  "How many alerts can I handle
   to catch X% of attacks?"

At 30% recall:
  → {recall_30_alerts:.0f} alerts/1k windows
  → {recall_30_fpr*100:.1f}% false positive rate

This is OPERATIONAL value,
not benchmark performance.
    """
    ax8.text(0.05, 0.5, takeaway, transform=ax8.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Alert Prioritization: From Detection to Decision")
    parser.add_argument("--dataset", choices=["unsw", "cse-cic"], default="unsw",
                       help="Dataset to use (default: unsw)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ALERT PRIORITIZATION")
    print("From Detection to Decision")
    print("=" * 60)

    # Load data
    print(f"\nLoading {args.dataset} data...")
    df = load_scored_data(args.dataset)

    if df is None:
        print("No data available. Please run training scripts first.")
        return

    print(f"  Observations: {len(df):,}")
    print(f"  Attack rate: {df['has_attack'].mean():.2%}")

    # Generate ranking report
    print("\nGenerating ranking metrics...")
    y_true = df["has_attack"].astype(int).values
    scores = df["anomaly_score"].values

    report = ranking_report(y_true, scores)
    print("\n" + report.to_string(index=False))

    # Save report
    report.to_csv(OUTPUT_DIR / "ranking_metrics.csv", index=False)

    # Build alert budget curve
    print("\nBuilding alert budget curve...")
    budget_curve = build_alert_budget_curve(scores, y_true)
    budget_curve.to_json(OUTPUT_DIR / "alert_budget_curve.json", orient="records", indent=2)

    # Generate dashboard
    print("\nGenerating triage dashboard...")
    dataset_name = "UNSW-NB15 (2% Attack Rate)" if args.dataset == "unsw" else "CSE-CIC-IDS2018"
    create_triage_dashboard(df, FIGURES_DIR / "triage_dashboard.png", dataset_name)

    # Build entity history and save enriched alerts
    if 'entity' in df.columns and 'event_count' in df.columns:
        print("\nEnriching top alerts with entity context...")
        history = build_entity_history(df)
        enriched = enrich_alerts(df, history, top_k=100)

        with open(OUTPUT_DIR / "enriched_alerts.json", "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"  Saved {len(enriched)} enriched alerts")

    print("\n" + "=" * 60)
    print("OUTPUTS")
    print("=" * 60)
    print(f"  Ranking metrics: {OUTPUT_DIR / 'ranking_metrics.csv'}")
    print(f"  Alert budget: {OUTPUT_DIR / 'alert_budget_curve.json'}")
    print(f"  Dashboard: {FIGURES_DIR / 'triage_dashboard.png'}")
    if 'entity' in df.columns:
        print(f"  Enriched alerts: {OUTPUT_DIR / 'enriched_alerts.json'}")


if __name__ == "__main__":
    main()
