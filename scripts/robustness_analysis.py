#!/usr/bin/env python3
"""
Robustness Analysis for BSAD.

Tests model stability under various conditions:
1. Attack Rate Sensitivity: Performance across 1%, 2%, 3%, 5%, 10% attack rates
2. Temporal Drift: Train on period A, test on B, C (concept drift)
3. Cold Entities: Test on entities not seen during training
4. Ranking Stability: Spearman/Kendall correlation between temporal windows

Usage:
    python scripts/robustness_analysis.py --output outputs/robustness
    make robustness
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsad.config import Settings
from bsad import steps


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
N_ENTITIES = 100
N_DAYS = 60  # Longer period for drift analysis
BSAD_SAMPLES = 800
BSAD_TUNE = 400
BSAD_CHAINS = 2


# =============================================================================
# 1. Attack Rate Sensitivity
# =============================================================================

def attack_rate_sensitivity(
    output_dir: Path,
    rates: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10],
    seed: int = SEED
) -> Dict[str, Any]:
    """
    Test BSAD performance across different attack rates.

    Key question: Does BSAD maintain advantage in the rare-event regime?
    """
    print("\n" + "="*60)
    print("ATTACK RATE SENSITIVITY ANALYSIS")
    print("="*60)

    results = {'rates': [], 'pr_auc': [], 'roc_auc': [], 'recall_at_50': []}

    for rate in rates:
        print(f"\nTesting attack_rate = {rate:.1%}...")

        settings = Settings(
            n_entities=N_ENTITIES,
            n_days=30,
            attack_rate=rate,
            n_samples=BSAD_SAMPLES,
            n_tune=BSAD_TUNE,
            n_chains=BSAD_CHAINS,
            random_seed=seed,
        )

        # Generate and process
        events_df, attacks_df = steps.generate_data(settings)
        modeling_df, metadata = steps.build_features(events_df, settings)
        arrays = steps.get_model_arrays(modeling_df)

        # Train and score
        trace = steps.train_model(arrays, settings)
        scores = steps.compute_scores(arrays['y'], trace, arrays['entity_idx'])
        intervals = steps.compute_intervals(trace, arrays['entity_idx'])
        scored_df = steps.create_scored_df(modeling_df, scores, intervals)

        # Evaluate
        metrics = steps.evaluate(scored_df)

        results['rates'].append(rate)
        results['pr_auc'].append(metrics['pr_auc'])
        results['roc_auc'].append(metrics['roc_auc'])
        results['recall_at_50'].append(metrics.get('recall_at_50', 0))

        print(f"  PR-AUC: {metrics['pr_auc']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(results['rates'], results['pr_auc'], 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].set_xlabel('Attack Rate')
    axes[0].set_ylabel('PR-AUC')
    axes[0].set_title('PR-AUC vs Attack Rate')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)

    # Highlight rare-event regime
    axes[0].axvspan(0, 0.05, alpha=0.2, color='green', label='Rare-event regime (<5%)')
    axes[0].legend()

    axes[1].plot(results['rates'], results['recall_at_50'], 's-', linewidth=2, markersize=8, color='coral')
    axes[1].set_xlabel('Attack Rate')
    axes[1].set_ylabel('Recall@50')
    axes[1].set_title('Recall@50 vs Attack Rate')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'attack_rate_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


# =============================================================================
# 2. Temporal Drift Analysis
# =============================================================================

def temporal_drift_analysis(
    output_dir: Path,
    seed: int = SEED
) -> Dict[str, Any]:
    """
    Test performance degradation over time (concept drift).

    Train on first 20 days, test on days 21-40 and 41-60.
    """
    print("\n" + "="*60)
    print("TEMPORAL DRIFT ANALYSIS")
    print("="*60)

    settings = Settings(
        n_entities=N_ENTITIES,
        n_days=N_DAYS,  # 60 days
        attack_rate=0.02,
        n_samples=BSAD_SAMPLES,
        n_tune=BSAD_TUNE,
        n_chains=BSAD_CHAINS,
        random_seed=seed,
    )

    # Generate full dataset
    events_df, attacks_df = steps.generate_data(settings)
    modeling_df, metadata = steps.build_features(events_df, settings)
    modeling_df = modeling_df.sort_values('window')

    # Split into periods
    n = len(modeling_df)
    period_size = n // 3

    train_df = modeling_df.iloc[:period_size].copy()
    test_period_1 = modeling_df.iloc[period_size:2*period_size].copy()
    test_period_2 = modeling_df.iloc[2*period_size:].copy()

    print(f"Train period: {len(train_df)} observations")
    print(f"Test period 1: {len(test_period_1)} observations")
    print(f"Test period 2: {len(test_period_2)} observations")

    # Train on first period
    train_arrays = steps.get_model_arrays(train_df)
    trace = steps.train_model(train_arrays, settings)

    results = {'periods': ['Train', 'Test Period 1', 'Test Period 2'], 'pr_auc': [], 'roc_auc': []}

    # Evaluate on each period
    for name, df in [('Train', train_df), ('Test Period 1', test_period_1), ('Test Period 2', test_period_2)]:
        arrays = steps.get_model_arrays(df)
        scores = steps.compute_scores(arrays['y'], trace, arrays['entity_idx'])

        # Simple evaluation without create_scored_df
        y_true = df['has_attack'].values.astype(int)
        anomaly_scores = scores['anomaly_score']

        pr_auc = average_precision_score(y_true, anomaly_scores)
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, anomaly_scores) if y_true.sum() > 0 else 0.0

        results['pr_auc'].append(pr_auc)
        results['roc_auc'].append(roc_auc)

        print(f"  {name}: PR-AUC = {pr_auc:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(results['periods']))
    width = 0.35

    bars1 = ax.bar(x - width/2, results['pr_auc'], width, label='PR-AUC', color='steelblue')
    bars2 = ax.bar(x + width/2, results['roc_auc'], width, label='ROC-AUC', color='coral')

    ax.set_xlabel('Period')
    ax.set_ylabel('Score')
    ax.set_title('Performance Degradation Over Time (Drift Analysis)')
    ax.set_xticks(x)
    ax.set_xticklabels(results['periods'])
    ax.legend()
    ax.set_ylim([0, 1])

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / 'temporal_drift.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Compute drift magnitude
    train_pr = results['pr_auc'][0]
    drift_1 = (train_pr - results['pr_auc'][1]) / train_pr * 100
    drift_2 = (train_pr - results['pr_auc'][2]) / train_pr * 100

    results['drift_period_1_pct'] = drift_1
    results['drift_period_2_pct'] = drift_2

    print(f"\nDrift magnitude:")
    print(f"  Period 1: {drift_1:.1f}% degradation")
    print(f"  Period 2: {drift_2:.1f}% degradation")

    return results


# =============================================================================
# 3. Cold Entity Analysis
# =============================================================================

def cold_entity_analysis(
    output_dir: Path,
    seed: int = SEED
) -> Dict[str, Any]:
    """
    Test performance on entities not seen during training.

    Train on 80% of entities, test on the remaining 20% (cold start).
    """
    print("\n" + "="*60)
    print("COLD ENTITY ANALYSIS")
    print("="*60)

    settings = Settings(
        n_entities=N_ENTITIES,
        n_days=30,
        attack_rate=0.02,
        n_samples=BSAD_SAMPLES,
        n_tune=BSAD_TUNE,
        n_chains=BSAD_CHAINS,
        random_seed=seed,
    )

    # Generate data
    events_df, attacks_df = steps.generate_data(settings)
    modeling_df, metadata = steps.build_features(events_df, settings)

    # Split entities
    unique_entities = modeling_df['user_id'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_entities)

    n_train_entities = int(len(unique_entities) * 0.8)
    train_entities = set(unique_entities[:n_train_entities])
    cold_entities = set(unique_entities[n_train_entities:])

    train_df = modeling_df[modeling_df['user_id'].isin(train_entities)].copy()
    cold_df = modeling_df[modeling_df['user_id'].isin(cold_entities)].copy()

    # Re-index entities for train_df (must be contiguous 0..n-1 for PyMC)
    train_entity_mapping = {e: i for i, e in enumerate(train_df['user_id'].unique())}
    train_df['entity_idx'] = train_df['user_id'].map(train_entity_mapping)

    print(f"Train entities: {len(train_entities)}")
    print(f"Cold entities: {len(cold_entities)}")
    print(f"Train observations: {len(train_df)}")
    print(f"Cold observations: {len(cold_df)}")

    # Train on known entities
    train_arrays = steps.get_model_arrays(train_df)
    trace = steps.train_model(train_arrays, settings)

    results = {
        'train_entities': len(train_entities),
        'cold_entities': len(cold_entities),
        'metrics': {}
    }

    # Evaluate on train entities
    scores_train = steps.compute_scores(train_arrays['y'], trace, train_arrays['entity_idx'])
    intervals_train = steps.compute_intervals(trace, train_arrays['entity_idx'])
    scored_train = steps.create_scored_df(train_df, scores_train, intervals_train)
    metrics_train = steps.evaluate(scored_train)
    results['metrics']['known_entities'] = {
        'pr_auc': metrics_train['pr_auc'],
        'roc_auc': metrics_train['roc_auc']
    }
    print(f"\nKnown entities: PR-AUC = {metrics_train['pr_auc']:.3f}")

    # For cold entities, we need to handle unseen entity_idx
    # Option 1: Use population parameters (new entity → prior)
    # This requires modifying the scoring to handle new entities

    # Simplified approach: Score cold entities using population mean
    theta_samples = trace.posterior['theta'].values
    phi_samples = trace.posterior['phi'].values

    # Population mean theta
    theta_mean = theta_samples.mean()
    phi_mean = phi_samples.mean()

    # Score cold entities using population baseline
    from scipy import stats

    y_cold = cold_df['event_count'].values
    cold_scores = np.zeros(len(y_cold))

    for i, y in enumerate(y_cold):
        p = phi_mean / (phi_mean + theta_mean)
        cold_scores[i] = -stats.nbinom.logpmf(y, n=phi_mean, p=p)

    y_true_cold = cold_df['has_attack'].values.astype(int)

    if y_true_cold.sum() > 0:
        pr_auc_cold = average_precision_score(y_true_cold, cold_scores)
    else:
        pr_auc_cold = 0.0

    results['metrics']['cold_entities'] = {
        'pr_auc': pr_auc_cold,
        'n_attacks': int(y_true_cold.sum())
    }
    print(f"Cold entities: PR-AUC = {pr_auc_cold:.3f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Known Entities\n(in training)', 'Cold Entities\n(unseen)']
    pr_aucs = [metrics_train['pr_auc'], pr_auc_cold]
    colors = ['steelblue', 'coral']

    bars = ax.bar(categories, pr_aucs, color=colors, edgecolor='black')
    ax.set_ylabel('PR-AUC')
    ax.set_title('Cold Start Analysis: Known vs Unseen Entities')
    ax.set_ylim([0, 1])

    for bar, val in zip(bars, pr_aucs):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    fig.savefig(output_dir / 'cold_entity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


# =============================================================================
# 4. Ranking Stability
# =============================================================================

def ranking_stability_analysis(
    output_dir: Path,
    seed: int = SEED
) -> Dict[str, Any]:
    """
    Test ranking stability across temporal windows.

    Compute Spearman and Kendall correlations between entity rankings
    in different time windows.
    """
    print("\n" + "="*60)
    print("RANKING STABILITY ANALYSIS")
    print("="*60)

    settings = Settings(
        n_entities=N_ENTITIES,
        n_days=N_DAYS,
        attack_rate=0.02,
        n_samples=BSAD_SAMPLES,
        n_tune=BSAD_TUNE,
        n_chains=BSAD_CHAINS,
        random_seed=seed,
    )

    # Generate data
    events_df, attacks_df = steps.generate_data(settings)
    modeling_df, metadata = steps.build_features(events_df, settings)
    modeling_df = modeling_df.sort_values('window')

    # Train model on full data
    arrays = steps.get_model_arrays(modeling_df)
    trace = steps.train_model(arrays, settings)

    # Score full data
    scores = steps.compute_scores(arrays['y'], trace, arrays['entity_idx'])
    modeling_df['anomaly_score'] = scores['anomaly_score']

    # Split into weekly windows
    modeling_df['week'] = pd.to_datetime(modeling_df['window']).dt.isocalendar().week
    weeks = modeling_df['week'].unique()[:4]  # First 4 weeks

    # Compute entity rankings per week
    weekly_rankings = {}
    for week in weeks:
        week_df = modeling_df[modeling_df['week'] == week]
        entity_scores = week_df.groupby('user_id')['anomaly_score'].mean()
        entity_ranks = entity_scores.rank(ascending=False)
        weekly_rankings[week] = entity_ranks

    # Compute pairwise correlations
    results = {'spearman': [], 'kendall': [], 'pairs': []}

    weeks_list = list(weekly_rankings.keys())
    for i in range(len(weeks_list)):
        for j in range(i+1, len(weeks_list)):
            w1, w2 = weeks_list[i], weeks_list[j]

            # Get common entities
            common = set(weekly_rankings[w1].index) & set(weekly_rankings[w2].index)
            if len(common) < 10:
                continue

            ranks1 = weekly_rankings[w1].loc[list(common)].values
            ranks2 = weekly_rankings[w2].loc[list(common)].values

            spearman, _ = spearmanr(ranks1, ranks2)
            kendall, _ = kendalltau(ranks1, ranks2)

            results['pairs'].append(f'Week {w1} vs {w2}')
            results['spearman'].append(spearman)
            results['kendall'].append(kendall)

    print(f"\nRanking correlations across weeks:")
    for pair, sp, kt in zip(results['pairs'], results['spearman'], results['kendall']):
        print(f"  {pair}: Spearman={sp:.3f}, Kendall={kt:.3f}")

    results['mean_spearman'] = np.mean(results['spearman'])
    results['mean_kendall'] = np.mean(results['kendall'])

    print(f"\nMean Spearman: {results['mean_spearman']:.3f}")
    print(f"Mean Kendall: {results['mean_kendall']:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(results['pairs']))
    width = 0.35

    bars1 = ax.bar(x - width/2, results['spearman'], width, label='Spearman', color='steelblue')
    bars2 = ax.bar(x + width/2, results['kendall'], width, label='Kendall', color='coral')

    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good stability (0.7)')

    ax.set_xlabel('Week Pairs')
    ax.set_ylabel('Correlation')
    ax.set_title('Ranking Stability Across Temporal Windows')
    ax.set_xticks(x)
    ax.set_xticklabels(results['pairs'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(output_dir / 'ranking_stability.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BSAD Robustness Analysis')
    parser.add_argument('--output', '-o', type=str, default='outputs/robustness',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip slow analyses')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("BSAD ROBUSTNESS ANALYSIS")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
    }

    # 1. Attack rate sensitivity
    all_results['attack_rate_sensitivity'] = attack_rate_sensitivity(output_dir, seed=args.seed)

    if not args.skip_slow:
        # 2. Temporal drift
        all_results['temporal_drift'] = temporal_drift_analysis(output_dir, seed=args.seed)

        # 3. Cold entity
        all_results['cold_entity'] = cold_entity_analysis(output_dir, seed=args.seed)

        # 4. Ranking stability
        all_results['ranking_stability'] = ranking_stability_analysis(output_dir, seed=args.seed)

    # Save results
    results_path = output_dir / 'robustness_results.json'

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(clean_for_json(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"Plots saved to {output_dir}/")


if __name__ == '__main__':
    main()
