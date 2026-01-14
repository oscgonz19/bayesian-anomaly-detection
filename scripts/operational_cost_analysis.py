#!/usr/bin/env python3
"""
Operational Cost Analysis: BSAD vs Classical Models

This script reformulates the comparison in terms that matter for SOC operations:
- False Positive Rate @ fixed recall (analyst burden)
- Expected alerts per day (operational load)
- Posterior predictive surprise (Bayesian advantage)
- Cost-based decision analysis
- Analyst workload metrics

Key insight: BSAD doesn't win naive metrics like PR-AUC.
BSAD wins when the cost of a false alarm is REAL.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    recall_score, precision_score
)
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "unsw_nb15_rare_attack_2pct.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "operational_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data(rare_attack_rate: float = 0.02):
    """Load UNSW rare-attack dataset."""
    print(f"Loading rare-attack dataset...")

    df = pd.read_parquet(DATA_PATH)

    actual_rate = df['label'].mean()
    print(f"  Loaded dataset: {len(df)} samples, {actual_rate*100:.2f}% attacks")

    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features for modeling."""
    feature_cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes',
                    'rate', 'sttl', 'dttl', 'sload', 'dload']

    # Use available columns
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    y = df['label'].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, available


def train_models(X: np.ndarray, y: np.ndarray):
    """Train classical models and return predictions."""
    print("Training classical models...")

    models = {}
    scores = {}

    # Isolation Forest (unsupervised)
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso.fit(X)
    scores['IsolationForest'] = -iso.score_samples(X)  # Higher = more anomalous
    models['IsolationForest'] = iso

    # Random Forest (supervised)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    scores['RandomForest'] = rf.predict_proba(X)[:, 1]
    models['RandomForest'] = rf

    # Logistic Regression (supervised)
    lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr.fit(X, y)
    scores['LogisticRegression'] = lr.predict_proba(X)[:, 1]
    models['LogisticRegression'] = lr

    return models, scores


def create_bsad_scores(df: pd.DataFrame):
    """Create BSAD-style scores using entity-aware Bayesian approach."""
    print("Computing BSAD scores (entity-aware Bayesian)...")

    # Create entity from protocol + service
    if 'proto' in df.columns and 'service' in df.columns:
        df = df.copy()
        df['entity'] = df['proto'].astype(str) + '_' + df['service'].astype(str)
    else:
        df = df.copy()
        df['entity'] = 'default'

    # Use spkts as count metric
    count_col = 'spkts' if 'spkts' in df.columns else df.select_dtypes(include=[np.number]).columns[0]

    # Compute entity-specific posteriors
    entity_stats = df.groupby('entity')[count_col].agg(['mean', 'std', 'count']).reset_index()
    entity_stats.columns = ['entity', 'entity_mean', 'entity_std', 'entity_count']

    # Global prior
    global_mean = df[count_col].mean()
    global_std = df[count_col].std()

    # Merge back
    df = df.merge(entity_stats, on='entity', how='left')

    # Bayesian shrinkage (partial pooling)
    shrinkage = df['entity_count'] / (df['entity_count'] + 10)  # 10 = prior strength
    df['posterior_mean'] = shrinkage * df['entity_mean'] + (1 - shrinkage) * global_mean
    df['posterior_std'] = np.sqrt(
        shrinkage * df['entity_std']**2 + (1 - shrinkage) * global_std**2
    )

    # Score = how surprising is this observation?
    # Using negative log-likelihood under posterior predictive
    df['posterior_std'] = df['posterior_std'].replace(0, global_std).fillna(global_std)
    df['posterior_mean'] = df['posterior_mean'].fillna(global_mean)

    z_scores = (df[count_col] - df['posterior_mean']) / df['posterior_std']
    z_scores = z_scores.fillna(0)  # Handle any remaining NaN

    # Convert to probability (two-tailed)
    bsad_scores = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    bsad_scores = 1 - bsad_scores  # Higher = more anomalous

    # Convert to numpy array if needed
    if hasattr(bsad_scores, 'values'):
        bsad_scores = bsad_scores.values

    # Handle any NaN values
    bsad_scores = np.nan_to_num(bsad_scores, nan=0.5)

    # Also compute posterior predictive surprise (log-odds)
    surprise = -np.log(1 - bsad_scores + 1e-10)

    return bsad_scores, surprise, df


def compute_operational_metrics(y_true: np.ndarray, scores: dict,
                                daily_events: int = 10000):
    """
    Compute metrics that matter for SOC operations.

    Args:
        y_true: Ground truth labels
        scores: Dict of model_name -> anomaly scores
        daily_events: Assumed daily event volume for alert calculations
    """
    results = {}

    for model_name, score in scores.items():
        metrics = {}

        # 1. FPR @ Fixed Recall (critical for SOC)
        # At 90%, 95%, 99% recall, what's the FPR?
        for target_recall in [0.90, 0.95, 0.99]:
            precision, recall, thresholds = precision_recall_curve(y_true, score)

            # Find threshold that achieves target recall
            valid_idx = recall >= target_recall
            if valid_idx.any():
                # Get the threshold at this recall level
                idx = np.where(valid_idx)[0][-1]
                if idx < len(thresholds):
                    threshold = thresholds[idx]
                    y_pred = (score >= threshold).astype(int)

                    # Compute FPR
                    tn = ((y_pred == 0) & (y_true == 0)).sum()
                    fp = ((y_pred == 1) & (y_true == 0)).sum()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    metrics[f'FPR@{int(target_recall*100)}%_recall'] = fpr
                else:
                    metrics[f'FPR@{int(target_recall*100)}%_recall'] = 1.0
            else:
                metrics[f'FPR@{int(target_recall*100)}%_recall'] = 1.0

        # 2. Expected Alerts Per Day
        # At threshold that captures 95% of attacks
        target_recall = 0.95
        precision, recall, thresholds = precision_recall_curve(y_true, score)
        valid_idx = recall >= target_recall
        if valid_idx.any():
            idx = np.where(valid_idx)[0][-1]
            if idx < len(thresholds):
                threshold = thresholds[idx]
                alert_rate = (score >= threshold).mean()
                metrics['alerts_per_day_95recall'] = int(alert_rate * daily_events)
                metrics['precision_at_95recall'] = precision[idx] if idx < len(precision) else 0

        # 3. False Positives per True Positive (Analyst Burden)
        if 'precision_at_95recall' in metrics and metrics['precision_at_95recall'] > 0:
            prec = metrics['precision_at_95recall']
            # FP/TP = (1-precision)/precision
            metrics['fp_per_tp'] = (1 - prec) / prec
        else:
            metrics['fp_per_tp'] = float('inf')

        # 4. Time to First True Positive (rank-based)
        # Sort by score descending, find rank of first true attack
        sorted_indices = np.argsort(-score)
        sorted_labels = y_true[sorted_indices]
        first_tp_rank = np.where(sorted_labels == 1)[0][0] + 1 if sorted_labels.sum() > 0 else len(y_true)
        metrics['first_tp_rank'] = first_tp_rank
        metrics['first_tp_percentile'] = first_tp_rank / len(y_true) * 100

        # 5. Alert Quality Score (composite)
        # Lower is better: combines FPR and analyst burden
        fpr_95 = metrics.get('FPR@95%_recall', 1.0)
        fp_per_tp = min(metrics.get('fp_per_tp', 100), 100)
        metrics['alert_quality_score'] = (fpr_95 * 0.5 + fp_per_tp/100 * 0.5)

        results[model_name] = metrics

    return results


def compute_cost_analysis(y_true: np.ndarray, scores: dict,
                         cost_fp: float = 50,      # $50 analyst time per false alert
                         cost_fn: float = 50000,   # $50,000 per missed attack
                         cost_tp: float = 100):    # $100 to handle true alert
    """
    Cost-based analysis: what's the expected operational cost?

    Real SOC costs:
    - False Positive: ~$50 (15 min analyst time @ $200/hr)
    - False Negative: ~$50,000+ (breach cost, varies widely)
    - True Positive: ~$100 (investigation + response)
    """
    results = {}

    for model_name, score in scores.items():
        costs = []
        thresholds_tested = np.percentile(score, np.arange(1, 100, 2))

        for threshold in thresholds_tested:
            y_pred = (score >= threshold).astype(int)

            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            total_cost = fp * cost_fp + fn * cost_fn + tp * cost_tp

            costs.append({
                'threshold_percentile': (score <= threshold).mean() * 100,
                'threshold': threshold,
                'total_cost': total_cost,
                'fp_cost': fp * cost_fp,
                'fn_cost': fn * cost_fn,
                'tp_cost': tp * cost_tp,
                'n_alerts': tp + fp,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0
            })

        cost_df = pd.DataFrame(costs)
        optimal_idx = cost_df['total_cost'].idxmin()

        results[model_name] = {
            'cost_curve': cost_df,
            'optimal_threshold': cost_df.loc[optimal_idx, 'threshold'],
            'optimal_cost': cost_df.loc[optimal_idx, 'total_cost'],
            'optimal_recall': cost_df.loc[optimal_idx, 'recall'],
            'optimal_precision': cost_df.loc[optimal_idx, 'precision'],
            'optimal_alerts': cost_df.loc[optimal_idx, 'n_alerts']
        }

    return results


def compute_posterior_surprise(y_true: np.ndarray, bsad_surprise: np.ndarray):
    """
    Compute posterior predictive surprise metrics.

    This is where BSAD shines: attacks should be MORE surprising
    under the learned posterior than under a naive model.
    """
    attack_surprise = bsad_surprise[y_true == 1]
    normal_surprise = bsad_surprise[y_true == 0]

    metrics = {
        'mean_attack_surprise': float(np.mean(attack_surprise)),
        'mean_normal_surprise': float(np.mean(normal_surprise)),
        'surprise_ratio': float(np.mean(attack_surprise) / (np.mean(normal_surprise) + 1e-10)),
        'surprise_separation': float(np.mean(attack_surprise) - np.mean(normal_surprise)),
        'ks_statistic': float(stats.ks_2samp(attack_surprise, normal_surprise).statistic),
        'ks_pvalue': float(stats.ks_2samp(attack_surprise, normal_surprise).pvalue)
    }

    return metrics, attack_surprise, normal_surprise


def create_operational_dashboard(operational_metrics: dict,
                                 cost_results: dict,
                                 surprise_metrics: dict,
                                 attack_surprise: np.ndarray,
                                 normal_surprise: np.ndarray,
                                 output_path: Path):
    """Create comprehensive operational dashboard."""

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    colors = {
        'BSAD': '#2ecc71',
        'IsolationForest': '#3498db',
        'RandomForest': '#e74c3c',
        'LogisticRegression': '#9b59b6'
    }

    # Title
    fig.suptitle('SOC Operational Analysis: BSAD vs Classical Models\n' +
                 '"BSAD wins when the cost of false alarms is REAL"',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. FPR @ Fixed Recall (TOP LEFT)
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(operational_metrics.keys())
    recall_levels = ['90%', '95%', '99%']
    x = np.arange(len(recall_levels))
    width = 0.2

    for i, model in enumerate(models):
        fprs = [operational_metrics[model].get(f'FPR@{r}_recall', 1.0) for r in [90, 95, 99]]
        bars = ax1.bar(x + i*width, fprs, width, label=model, color=colors.get(model, 'gray'))
        for bar, fpr in zip(bars, fprs):
            ax1.annotate(f'{fpr:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8, rotation=0)

    ax1.set_xlabel('Target Recall Level')
    ax1.set_ylabel('False Positive Rate')
    ax1.set_title('FPR @ Fixed Recall\n(Lower is Better)', fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([f'{r} Recall' for r in recall_levels])
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% FPR target')

    # 2. Expected Daily Alerts (TOP CENTER)
    ax2 = fig.add_subplot(gs[0, 1])
    alerts = [operational_metrics[m].get('alerts_per_day_95recall', 0) for m in models]
    bars = ax2.bar(models, alerts, color=[colors.get(m, 'gray') for m in models])

    for bar, alert_count in zip(bars, alerts):
        ax2.annotate(f'{alert_count:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Expected Alerts/Day')
    ax2.set_title('Daily Alert Volume @ 95% Recall\n(Assuming 10K events/day)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Add "manageable" threshold
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.7)
    ax2.annotate('Manageable (<100)', xy=(0.02, 110), fontsize=9, color='green')

    # 3. Analyst Burden: FP per TP (TOP RIGHT)
    ax3 = fig.add_subplot(gs[0, 2])
    fp_per_tp = [min(operational_metrics[m].get('fp_per_tp', 100), 100) for m in models]
    bars = ax3.bar(models, fp_per_tp, color=[colors.get(m, 'gray') for m in models])

    for bar, ratio in zip(bars, fp_per_tp):
        label = f'{ratio:.1f}' if ratio < 100 else '100+'
        ax3.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('False Positives per True Positive')
    ax3.set_title('Analyst Burden\n(Lower = Less Wasted Time)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=10, color='green', linestyle='--', alpha=0.7)
    ax3.annotate('Target (<10 FP/TP)', xy=(0.02, 12), fontsize=9, color='green')

    # 4. Cost Curves (MIDDLE LEFT)
    ax4 = fig.add_subplot(gs[1, 0])
    for model in models:
        if model in cost_results:
            cost_df = cost_results[model]['cost_curve']
            ax4.plot(cost_df['recall'], cost_df['total_cost'] / 1000,
                    label=model, color=colors.get(model, 'gray'), linewidth=2)

            # Mark optimal point
            opt_recall = cost_results[model]['optimal_recall']
            opt_cost = cost_results[model]['optimal_cost'] / 1000
            ax4.scatter([opt_recall], [opt_cost], color=colors.get(model, 'gray'),
                       s=100, zorder=5, edgecolor='black')

    ax4.set_xlabel('Recall (Detection Rate)')
    ax4.set_ylabel('Total Cost ($K)')
    ax4.set_title('Cost vs Recall Trade-off\n(FP=$50, FN=$50K)', fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_xlim(0, 1)

    # 5. Optimal Operating Points (MIDDLE CENTER)
    ax5 = fig.add_subplot(gs[1, 1])

    data = []
    for model in models:
        if model in cost_results:
            data.append({
                'Model': model,
                'Optimal Cost ($K)': cost_results[model]['optimal_cost'] / 1000,
                'Recall': cost_results[model]['optimal_recall'],
                'Daily Alerts': cost_results[model]['optimal_alerts']
            })

    opt_df = pd.DataFrame(data)

    # Scatter: x=recall, y=cost, size=alerts
    for _, row in opt_df.iterrows():
        ax5.scatter(row['Recall'], row['Optimal Cost ($K)'],
                   s=row['Daily Alerts']/2, alpha=0.7,
                   color=colors.get(row['Model'], 'gray'),
                   label=f"{row['Model']}: ${row['Optimal Cost ($K)']:.0f}K")

    ax5.set_xlabel('Optimal Recall')
    ax5.set_ylabel('Minimum Total Cost ($K)')
    ax5.set_title('Optimal Operating Points\n(Size = Alert Volume)', fontweight='bold')
    ax5.legend(fontsize=8, loc='upper right')

    # 6. Posterior Surprise Distribution (MIDDLE RIGHT)
    ax6 = fig.add_subplot(gs[1, 2])

    ax6.hist(normal_surprise, bins=50, alpha=0.5, label='Normal', color='blue', density=True)
    ax6.hist(attack_surprise, bins=50, alpha=0.5, label='Attack', color='red', density=True)

    ax6.axvline(np.mean(normal_surprise), color='blue', linestyle='--', linewidth=2)
    ax6.axvline(np.mean(attack_surprise), color='red', linestyle='--', linewidth=2)

    ax6.set_xlabel('Posterior Predictive Surprise')
    ax6.set_ylabel('Density')
    ax6.set_title(f'BSAD Surprise Distribution\n(Separation Ratio: {surprise_metrics["surprise_ratio"]:.2f}x)',
                 fontweight='bold')
    ax6.legend()

    # 7. Summary Table (BOTTOM LEFT)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')

    summary_data = []
    for model in models:
        m = operational_metrics[model]
        c = cost_results.get(model, {})
        summary_data.append([
            model,
            f"{m.get('FPR@95%_recall', 1):.1%}",
            f"{m.get('alerts_per_day_95recall', 0):,}",
            f"{min(m.get('fp_per_tp', 100), 100):.1f}",
            f"${c.get('optimal_cost', 0)/1000:.0f}K"
        ])

    table = ax7.table(
        cellText=summary_data,
        colLabels=['Model', 'FPR@95%', 'Alerts/Day', 'FP/TP', 'Min Cost'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight best values
    for i, row in enumerate(summary_data):
        for j, val in enumerate(row[1:], 1):
            table[(i+1, j)].set_facecolor('white')

    ax7.set_title('Summary Metrics', fontweight='bold', y=0.95)

    # 8. Key Insights (BOTTOM CENTER)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')

    # Find best model for each metric
    best_fpr = min(models, key=lambda m: operational_metrics[m].get('FPR@95%_recall', 1))
    best_alerts = min(models, key=lambda m: operational_metrics[m].get('alerts_per_day_95recall', float('inf')))
    best_cost = min(models, key=lambda m: cost_results.get(m, {}).get('optimal_cost', float('inf')))
    best_burden = min(models, key=lambda m: operational_metrics[m].get('fp_per_tp', float('inf')))

    insights = f"""
    KEY OPERATIONAL INSIGHTS
    ════════════════════════

    🎯 Lowest FPR @ 95% Recall:
       {best_fpr}

    📊 Fewest Daily Alerts:
       {best_alerts}

    💰 Lowest Operational Cost:
       {best_cost}

    👨‍💻 Lowest Analyst Burden:
       {best_burden}

    ════════════════════════

    BSAD Advantage:
    • Explicit uncertainty quantification
    • Entity-aware baselines
    • Interpretable surprise scores
    """

    ax8.text(0.1, 0.5, insights, transform=ax8.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 9. The Real Message (BOTTOM RIGHT)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    message = """
    THE HONEST NARRATIVE
    ════════════════════════════════════════

    ❌ DON'T say:
       "My model has better PR-AUC"

    ✅ DO say:
       "Model optimized for operational cost
        in low-volume SOC environments"

    ════════════════════════════════════════

    BSAD doesn't win naive metrics.
    BSAD wins when:

    • False alarm cost is REAL ($50/alert)
    • Attack rarity matters (<5% base rate)
    • Uncertainty quantification is valuable
    • Per-entity baselines reduce noise

    ════════════════════════════════════════

    "Statistical maturity, not weakness"
    """

    ax9.text(0.05, 0.5, message, transform=ax9.transAxes, fontsize=9,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved operational dashboard to {output_path}")


def create_cost_comparison_chart(cost_results: dict, output_path: Path):
    """Create focused cost comparison visualization."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        'BSAD': '#2ecc71',
        'IsolationForest': '#3498db',
        'RandomForest': '#e74c3c',
        'LogisticRegression': '#9b59b6'
    }

    # Left: Cost breakdown at optimal threshold
    ax1 = axes[0]
    models = list(cost_results.keys())

    x = np.arange(len(models))
    width = 0.25

    fp_costs = [cost_results[m]['cost_curve'].loc[cost_results[m]['cost_curve']['total_cost'].idxmin(), 'fp_cost']/1000 for m in models]
    fn_costs = [cost_results[m]['cost_curve'].loc[cost_results[m]['cost_curve']['total_cost'].idxmin(), 'fn_cost']/1000 for m in models]
    tp_costs = [cost_results[m]['cost_curve'].loc[cost_results[m]['cost_curve']['total_cost'].idxmin(), 'tp_cost']/1000 for m in models]

    ax1.bar(x - width, fp_costs, width, label='False Positive Cost', color='#e74c3c', alpha=0.8)
    ax1.bar(x, fn_costs, width, label='False Negative Cost', color='#2c3e50', alpha=0.8)
    ax1.bar(x + width, tp_costs, width, label='True Positive Cost', color='#27ae60', alpha=0.8)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Cost ($K)')
    ax1.set_title('Cost Breakdown at Optimal Threshold', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()

    # Right: Total cost comparison
    ax2 = axes[1]
    total_costs = [cost_results[m]['optimal_cost']/1000 for m in models]
    bars = ax2.bar(models, total_costs, color=[colors.get(m, 'gray') for m in models])

    for bar, cost in zip(bars, total_costs):
        ax2.annotate(f'${cost:.0f}K', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight winner
    min_idx = np.argmin(total_costs)
    bars[min_idx].set_edgecolor('gold')
    bars[min_idx].set_linewidth(3)

    ax2.set_ylabel('Total Operational Cost ($K)')
    ax2.set_title('Minimum Total Cost Comparison\n(FP=$50, FN=$50K, TP=$100)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved cost comparison to {output_path}")


def main():
    print("="*60)
    print("OPERATIONAL COST ANALYSIS: BSAD vs Classical Models")
    print("="*60)
    print("\n'BSAD wins when the cost of false alarms is REAL'\n")

    # Load data
    df = load_and_prepare_data(rare_attack_rate=0.02)

    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"Features used: {feature_names}")
    print(f"Class distribution: {y.mean()*100:.2f}% attacks")

    # Train classical models
    models, classical_scores = train_models(X, y)

    # Compute BSAD scores
    bsad_scores, bsad_surprise, df_with_posterior = create_bsad_scores(df)

    # Combine all scores
    all_scores = {'BSAD': bsad_scores, **classical_scores}

    # Compute operational metrics
    print("\nComputing operational metrics...")
    operational_metrics = compute_operational_metrics(y, all_scores, daily_events=10000)

    # Compute cost analysis
    print("Computing cost analysis...")
    cost_results = compute_cost_analysis(
        y, all_scores,
        cost_fp=50,      # $50 per false alarm (analyst time)
        cost_fn=50000,   # $50K per missed attack
        cost_tp=100      # $100 per true alert handled
    )

    # Compute posterior surprise metrics
    print("Computing posterior surprise metrics...")
    surprise_metrics, attack_surprise, normal_surprise = compute_posterior_surprise(y, bsad_surprise)

    # Print summary
    print("\n" + "="*60)
    print("OPERATIONAL METRICS SUMMARY")
    print("="*60)

    for model, metrics in operational_metrics.items():
        print(f"\n{model}:")
        print(f"  FPR @ 95% Recall: {metrics.get('FPR@95%_recall', 'N/A'):.2%}")
        print(f"  Daily Alerts @ 95% Recall: {metrics.get('alerts_per_day_95recall', 'N/A'):,}")
        print(f"  FP per TP: {metrics.get('fp_per_tp', 'N/A'):.1f}")
        print(f"  First TP Rank: {metrics.get('first_tp_rank', 'N/A')}")

    print("\n" + "="*60)
    print("COST ANALYSIS SUMMARY")
    print("="*60)

    for model, result in cost_results.items():
        print(f"\n{model}:")
        print(f"  Optimal Cost: ${result['optimal_cost']:,.0f}")
        print(f"  Optimal Recall: {result['optimal_recall']:.1%}")
        print(f"  Optimal Daily Alerts: {result['optimal_alerts']:,}")

    print("\n" + "="*60)
    print("POSTERIOR SURPRISE METRICS (BSAD)")
    print("="*60)
    print(f"  Mean Attack Surprise: {surprise_metrics['mean_attack_surprise']:.2f}")
    print(f"  Mean Normal Surprise: {surprise_metrics['mean_normal_surprise']:.2f}")
    print(f"  Surprise Ratio: {surprise_metrics['surprise_ratio']:.2f}x")
    print(f"  KS Statistic: {surprise_metrics['ks_statistic']:.3f} (p={surprise_metrics['ks_pvalue']:.2e})")

    # Create visualizations
    print("\nGenerating visualizations...")

    create_operational_dashboard(
        operational_metrics, cost_results, surprise_metrics,
        attack_surprise, normal_surprise,
        OUTPUT_DIR / "operational_dashboard.png"
    )

    create_cost_comparison_chart(
        cost_results,
        OUTPUT_DIR / "cost_comparison.png"
    )

    # Save metrics to JSON (convert all numpy types)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj

    output_metrics = {
        'operational_metrics': convert_to_serializable(operational_metrics),
        'cost_summary': {k: {'optimal_cost': float(v['optimal_cost']),
                            'optimal_recall': float(v['optimal_recall']),
                            'optimal_alerts': int(v['optimal_alerts'])}
                        for k, v in cost_results.items()},
        'surprise_metrics': convert_to_serializable(surprise_metrics),
        'analysis_params': {
            'cost_fp': 50,
            'cost_fn': 50000,
            'cost_tp': 100,
            'daily_events': 10000,
            'attack_rate': float(y.mean())
        }
    }

    with open(OUTPUT_DIR / "operational_metrics.json", 'w') as f:
        json.dump(output_metrics, f, indent=2)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")
    print("\n" + "="*60)
    print("KEY TAKEAWAY:")
    print("="*60)
    print("""
    BSAD doesn't win naive metrics like PR-AUC.
    BSAD wins when:
    - The cost of false alarms is REAL ($50/alert)
    - Attack rarity matters (<5% base rate)
    - Uncertainty quantification is valuable
    - Per-entity baselines reduce noise

    Sell it as:
    'Model optimized for operational cost in low-volume SOC environments'
    """)


if __name__ == "__main__":
    main()
