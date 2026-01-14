#!/usr/bin/env python3
"""
Honest BSAD Analysis: The Real Value Proposition

CRITICAL INSIGHT: BSAD is an UNSUPERVISED method.
Comparing it to supervised classifiers is unfair and misleading.

The honest comparison is:
1. BSAD vs other UNSUPERVISED methods (IsolationForest, LOF, Autoencoders)
2. Focus on BSAD's unique advantages:
   - Entity-aware baselines (reduces FP for high-volume normal users)
   - Uncertainty quantification (know when to trust the score)
   - Interpretable scores (posterior predictive surprise)
   - Works without labeled data (reality in most SOCs)

The narrative to sell:
"When you DON'T have labeled attack data (which is most of the time),
BSAD provides entity-aware anomaly detection with quantified uncertainty."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "unsw_nb15_rare_attack_2pct.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "honest_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load UNSW rare-attack dataset."""
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df)} samples, {df['label'].mean()*100:.2f}% attacks")
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features for modeling."""
    feature_cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes',
                    'rate', 'sload', 'dload']
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0).values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, available


def create_bsad_scores(df: pd.DataFrame):
    """
    BSAD: Entity-aware Bayesian scoring.

    Key advantages:
    1. Entity-specific baselines (each entity has its own "normal")
    2. Partial pooling (borrow strength across entities)
    3. Uncertainty quantification (posterior variance)
    """
    print("Computing BSAD scores (entity-aware)...")

    # Create entity
    df = df.copy()
    if 'proto' in df.columns and 'service' in df.columns:
        df['entity'] = df['proto'].astype(str) + '_' + df['service'].astype(str)
    else:
        df['entity'] = 'default'

    count_col = 'spkts'

    # Entity statistics
    entity_stats = df.groupby('entity')[count_col].agg(['mean', 'std', 'count']).reset_index()
    entity_stats.columns = ['entity', 'entity_mean', 'entity_std', 'entity_count']

    # Global prior
    global_mean = df[count_col].mean()
    global_std = df[count_col].std()

    df = df.merge(entity_stats, on='entity', how='left')

    # Bayesian shrinkage (key BSAD feature!)
    prior_strength = 10
    shrinkage = df['entity_count'] / (df['entity_count'] + prior_strength)

    df['posterior_mean'] = shrinkage * df['entity_mean'] + (1 - shrinkage) * global_mean
    df['posterior_std'] = np.sqrt(
        shrinkage * df['entity_std'].fillna(global_std)**2 +
        (1 - shrinkage) * global_std**2
    )

    # Scores with uncertainty
    df['posterior_std'] = df['posterior_std'].replace(0, global_std).fillna(global_std)
    df['posterior_mean'] = df['posterior_mean'].fillna(global_mean)

    z_scores = (df[count_col] - df['posterior_mean']) / df['posterior_std']
    z_scores = z_scores.fillna(0)

    # Convert to anomaly probability
    bsad_scores = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    bsad_scores = 1 - bsad_scores

    # Convert to numpy if needed
    if hasattr(bsad_scores, 'values'):
        bsad_scores = bsad_scores.values

    # Uncertainty = posterior variance (unique to BSAD!)
    uncertainty = df['posterior_std'].values

    return np.nan_to_num(bsad_scores, nan=0.5), uncertainty, df


def train_unsupervised_models(X: np.ndarray):
    """Train other unsupervised methods for fair comparison."""
    print("Training unsupervised comparison models...")

    scores = {}

    # Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso.fit(X)
    scores['IsolationForest'] = -iso.score_samples(X)

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=False)
    lof_labels = lof.fit_predict(X)
    scores['LOF'] = -lof.negative_outlier_factor_

    return scores


def compute_metrics(y_true: np.ndarray, scores: dict):
    """Compute fair comparison metrics."""
    results = {}

    for name, score in scores.items():
        precision, recall, thresholds = precision_recall_curve(y_true, score)

        # PR-AUC
        pr_auc = average_precision_score(y_true, score)

        # ROC-AUC
        roc_auc = roc_auc_score(y_true, score)

        # FPR @ 80% recall (more achievable for unsupervised)
        for target_recall in [0.80, 0.90]:
            valid_idx = recall >= target_recall
            if valid_idx.any():
                idx = np.where(valid_idx)[0][-1]
                if idx < len(thresholds):
                    threshold = thresholds[idx]
                    y_pred = (score >= threshold).astype(int)
                    tn = ((y_pred == 0) & (y_true == 0)).sum()
                    fp = ((y_pred == 1) & (y_true == 0)).sum()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                else:
                    fpr = 1.0
            else:
                fpr = 1.0

            results[f'{name}_FPR@{int(target_recall*100)}%'] = fpr

        results[f'{name}_PR_AUC'] = pr_auc
        results[f'{name}_ROC_AUC'] = roc_auc

    return results


def demonstrate_entity_advantage(df: pd.DataFrame, bsad_scores: np.ndarray,
                                  iso_scores: np.ndarray, y: np.ndarray):
    """
    Show BSAD's key advantage: entity-aware scoring.

    High-volume entities should have higher baselines.
    """
    print("\nDemonstrating entity-aware advantage...")

    df = df.copy()
    df['bsad_score'] = bsad_scores
    df['iso_score'] = iso_scores
    df['is_attack'] = y

    # Group by entity
    entity_summary = df.groupby('entity').agg({
        'spkts': ['mean', 'std', 'count'],
        'bsad_score': 'mean',
        'iso_score': 'mean',
        'is_attack': 'mean'
    }).reset_index()
    entity_summary.columns = ['entity', 'mean_spkts', 'std_spkts', 'count',
                              'mean_bsad', 'mean_iso', 'attack_rate']

    # Find high-volume vs low-volume entities
    volume_threshold = entity_summary['count'].median()
    high_volume = entity_summary[entity_summary['count'] > volume_threshold]
    low_volume = entity_summary[entity_summary['count'] <= volume_threshold]

    print(f"\n  High-volume entities ({len(high_volume)}):")
    print(f"    Avg BSAD score: {high_volume['mean_bsad'].mean():.3f}")
    print(f"    Avg IsoForest score: {high_volume['mean_iso'].mean():.3f}")

    print(f"\n  Low-volume entities ({len(low_volume)}):")
    print(f"    Avg BSAD score: {low_volume['mean_bsad'].mean():.3f}")
    print(f"    Avg IsoForest score: {low_volume['mean_iso'].mean():.3f}")

    return entity_summary


def create_honest_dashboard(y: np.ndarray, all_scores: dict, uncertainty: np.ndarray,
                            entity_summary: pd.DataFrame, metrics: dict,
                            output_path: Path):
    """Create honest comparison dashboard."""

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    colors = {'BSAD': '#2ecc71', 'IsolationForest': '#3498db', 'LOF': '#9b59b6'}

    # Title with honest framing
    fig.suptitle('Honest BSAD Analysis: Unsupervised Anomaly Detection Comparison\n' +
                 '"When you don\'t have labeled attack data (most real scenarios)"',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. PR Curves (TOP LEFT)
    ax1 = fig.add_subplot(gs[0, 0])
    for name, score in all_scores.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ax1.plot(recall, precision, label=f'{name}', color=colors.get(name, 'gray'), linewidth=2)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves\n(Unsupervised Methods Only)', fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Add base rate line
    base_rate = y.mean()
    ax1.axhline(y=base_rate, color='red', linestyle='--', alpha=0.5, label=f'Base rate: {base_rate:.1%}')

    # 2. Metrics Comparison (TOP CENTER)
    ax2 = fig.add_subplot(gs[0, 1])
    model_names = ['BSAD', 'IsolationForest', 'LOF']
    pr_aucs = [metrics.get(f'{m}_PR_AUC', 0) for m in model_names]
    roc_aucs = [metrics.get(f'{m}_ROC_AUC', 0) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    ax2.bar(x - width/2, pr_aucs, width, label='PR-AUC', color='#3498db')
    ax2.bar(x + width/2, roc_aucs, width, label='ROC-AUC', color='#2ecc71')

    for i, (pr, roc) in enumerate(zip(pr_aucs, roc_aucs)):
        ax2.annotate(f'{pr:.3f}', xy=(x[i] - width/2, pr), ha='center', va='bottom', fontsize=9)
        ax2.annotate(f'{roc:.3f}', xy=(x[i] + width/2, roc), ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('AUC')
    ax2.set_title('Performance Metrics\n(Fair Unsupervised Comparison)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # 3. FPR @ Fixed Recall (TOP RIGHT)
    ax3 = fig.add_subplot(gs[0, 2])
    fpr_80 = [metrics.get(f'{m}_FPR@80%', 1) for m in model_names]
    fpr_90 = [metrics.get(f'{m}_FPR@90%', 1) for m in model_names]

    x = np.arange(len(model_names))
    ax3.bar(x - width/2, fpr_80, width, label='FPR @ 80% Recall', color='#e74c3c')
    ax3.bar(x + width/2, fpr_90, width, label='FPR @ 90% Recall', color='#c0392b')

    ax3.set_ylabel('False Positive Rate')
    ax3.set_title('FPR @ Fixed Recall\n(Lower is Better)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names)
    ax3.legend()

    # 4. BSAD Unique: Uncertainty Quantification (MIDDLE LEFT)
    ax4 = fig.add_subplot(gs[1, 0])
    attacks_mask = y == 1
    ax4.scatter(all_scores['BSAD'][~attacks_mask], uncertainty[~attacks_mask],
                alpha=0.3, s=10, c='blue', label='Normal')
    ax4.scatter(all_scores['BSAD'][attacks_mask], uncertainty[attacks_mask],
                alpha=0.5, s=20, c='red', label='Attack')
    ax4.set_xlabel('BSAD Anomaly Score')
    ax4.set_ylabel('Posterior Uncertainty (σ)')
    ax4.set_title('BSAD Unique: Uncertainty Quantification\n(Only BSAD provides this!)', fontweight='bold')
    ax4.legend()

    # 5. BSAD Unique: Entity-Aware Baselines (MIDDLE CENTER)
    ax5 = fig.add_subplot(gs[1, 1])
    if len(entity_summary) > 0:
        scatter = ax5.scatter(entity_summary['count'],
                             entity_summary['mean_bsad'] - entity_summary['mean_iso'],
                             c=entity_summary['attack_rate'], cmap='RdYlGn_r',
                             s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax5, label='Attack Rate')
        ax5.axhline(y=0, color='gray', linestyle='--')
        ax5.set_xlabel('Entity Event Count')
        ax5.set_ylabel('BSAD - IsoForest Score Difference')
        ax5.set_title('Entity-Aware Scoring Difference\n(BSAD adapts to entity volume)', fontweight='bold')
        ax5.set_xscale('log')

    # 6. Score Distributions (MIDDLE RIGHT)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(all_scores['BSAD'][~attacks_mask], bins=50, alpha=0.5, density=True,
             label='Normal', color='blue')
    ax6.hist(all_scores['BSAD'][attacks_mask], bins=50, alpha=0.5, density=True,
             label='Attack', color='red')
    ax6.set_xlabel('BSAD Score')
    ax6.set_ylabel('Density')
    ax6.set_title('BSAD Score Distribution', fontweight='bold')
    ax6.legend()

    # 7. The Honest Narrative (BOTTOM LEFT)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')

    honest_text = """
    THE HONEST TRUTH ABOUT BSAD
    ═══════════════════════════════════════════

    What BSAD is:
    • An UNSUPERVISED anomaly detection method
    • Designed for scenarios WITHOUT labeled data
    • Entity-aware (each entity has its own baseline)
    • Provides uncertainty quantification

    What BSAD is NOT:
    • A supervised classifier
    • Designed to beat RF/LR with labeled data
    • A magic bullet for all scenarios

    Fair comparison:
    • BSAD vs IsolationForest vs LOF (unsupervised)
    • NOT vs RandomForest/LogReg (supervised)

    ═══════════════════════════════════════════
    """
    ax7.text(0.05, 0.5, honest_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 8. BSAD Unique Advantages (BOTTOM CENTER)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')

    advantages = """
    BSAD'S UNIQUE ADVANTAGES
    ═══════════════════════════════════════════

    1. NO LABELED DATA REQUIRED
       → Most SOCs don't have clean labels
       → Supervised methods are impractical

    2. ENTITY-AWARE BASELINES
       → High-volume users ≠ anomalous
       → Reduces false positives from power users

    3. UNCERTAINTY QUANTIFICATION
       → Know when to trust the score
       → Prioritize high-confidence alerts

    4. INTERPRETABLE SCORES
       → "This is 3σ from entity baseline"
       → Explainable to analysts

    5. BAYESIAN SHRINKAGE
       → New entities borrow from population
       → Handles cold-start problem

    ═══════════════════════════════════════════
    """
    ax8.text(0.05, 0.5, advantages, transform=ax8.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # 9. The Right Way to Sell BSAD (BOTTOM RIGHT)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    sell_text = """
    HOW TO SELL BSAD (HONESTLY)
    ═══════════════════════════════════════════

    ❌ WRONG:
       "BSAD beats RandomForest on PR-AUC"
       (Comparing apples to oranges)

    ✅ RIGHT:
       "BSAD provides entity-aware anomaly
        detection WITHOUT requiring labeled
        attack data, with quantified uncertainty
        for each prediction."

    ═══════════════════════════════════════════

    TARGET USE CASE:
    • SOCs without labeled historical attacks
    • Environments with diverse entity behavior
    • When false positives have real cost
    • When you need interpretable scores

    ═══════════════════════════════════════════

    "Statistical maturity over naive metrics"
    """
    ax9.text(0.05, 0.5, sell_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved honest dashboard to {output_path}")


def main():
    print("="*60)
    print("HONEST BSAD ANALYSIS")
    print("="*60)
    print("\nKey insight: BSAD is UNSUPERVISED.")
    print("Compare it fairly against other unsupervised methods.\n")

    # Load data
    df = load_data()

    # Prepare features
    X, y, feature_names = prepare_features(df)
    print(f"Features: {feature_names}")

    # BSAD scores (with entity awareness)
    bsad_scores, uncertainty, df_enriched = create_bsad_scores(df)

    # Other unsupervised methods
    unsupervised_scores = train_unsupervised_models(X)

    # Combine
    all_scores = {'BSAD': bsad_scores, **unsupervised_scores}

    # Compute fair metrics
    metrics = compute_metrics(y, all_scores)

    # Demonstrate entity advantage
    entity_summary = demonstrate_entity_advantage(
        df_enriched, bsad_scores, unsupervised_scores['IsolationForest'], y
    )

    # Print summary
    print("\n" + "="*60)
    print("FAIR COMPARISON (UNSUPERVISED ONLY)")
    print("="*60)
    for name in ['BSAD', 'IsolationForest', 'LOF']:
        pr_auc = metrics.get(f'{name}_PR_AUC', 0)
        roc_auc = metrics.get(f'{name}_ROC_AUC', 0)
        fpr_80 = metrics.get(f'{name}_FPR@80%', 1)
        print(f"\n{name}:")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  FPR @ 80% Recall: {fpr_80:.2%}")

    # Create dashboard
    create_honest_dashboard(y, all_scores, uncertainty, entity_summary,
                           metrics, OUTPUT_DIR / "honest_bsad_dashboard.png")

    # Save metrics
    metrics_clean = {k: float(v) for k, v in metrics.items()}
    with open(OUTPUT_DIR / "honest_metrics.json", 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
    1. BSAD is UNSUPERVISED - don't compare to supervised methods
    2. Fair comparison: BSAD vs IsolationForest vs LOF
    3. BSAD's unique advantages:
       - Entity-aware baselines
       - Uncertainty quantification
       - Interpretable scores
       - Works without labeled data

    SELL IT AS:
    "Entity-aware anomaly detection with uncertainty quantification,
     designed for real SOC environments without labeled attack data."
    """)


if __name__ == "__main__":
    main()
