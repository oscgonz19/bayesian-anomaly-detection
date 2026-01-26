#!/usr/bin/env python3
"""
Reproducible Benchmark Suite for BSAD.

Compares BSAD against fair baselines:
- Count-specific: NB_MLE, NB_EmpiricalBayes, GLMM_NB, ZScore, GlobalNB
- Generic (reference): IsolationForest, LOF, OCSVM

Protocol:
- Fixed seeds for reproducibility
- Temporal splits (train early → test later)
- Multiple attack rates (1%, 2%, 5%)
- Structured output: metrics.json, curves.csv, plots

Usage:
    python scripts/benchmark.py --output outputs/benchmark
    python scripts/benchmark.py --attack-rates 0.01 0.02 0.05 --output outputs/benchmark
    make benchmark
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsad.config import Settings
from bsad import steps
from bsad.baselines import run_all_baselines
from triage.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    fpr_at_fixed_recall,
    alerts_per_k_windows,
)


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
ATTACK_RATES = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
K_VALUES = [10, 25, 50, 100, 200]  # For precision@k, recall@k
RECALL_TARGETS = [0.1, 0.2, 0.3, 0.5]  # For FPR analysis
N_ENTITIES = 100
N_DAYS = 30
BSAD_SAMPLES = 1000  # Reduced for benchmark speed
BSAD_TUNE = 500
BSAD_CHAINS = 2


# =============================================================================
# Temporal Split
# =============================================================================

def temporal_train_test_split(
    modeling_df: pd.DataFrame,
    train_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (train on early data, test on later).

    This is more realistic than random splits for time-series-like data.
    """
    modeling_df = modeling_df.sort_values('window')

    # Find split point
    n = len(modeling_df)
    split_idx = int(n * train_ratio)

    train_df = modeling_df.iloc[:split_idx].copy()
    test_df = modeling_df.iloc[split_idx:].copy()

    return train_df, test_df


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    k_values: List[int] = K_VALUES,
    recall_targets: List[float] = RECALL_TARGETS
) -> Dict[str, Any]:
    """Compute comprehensive metrics for a single model."""
    metrics = {
        'model': model_name,
        'pr_auc': float(average_precision_score(y_true, scores)),
        'roc_auc': float(roc_auc_score(y_true, scores)) if y_true.sum() > 0 else 0.0,
        'n_samples': len(y_true),
        'n_positives': int(y_true.sum()),
        'attack_rate': float(y_true.mean()),
    }

    # Precision@k and Recall@k
    for k in k_values:
        if k <= len(y_true):
            metrics[f'precision_at_{k}'] = float(precision_at_k(y_true, scores, k))
            metrics[f'recall_at_{k}'] = float(recall_at_k(y_true, scores, k))

    # FPR at fixed recall
    for r in recall_targets:
        metrics[f'fpr_at_recall_{int(r*100)}'] = float(fpr_at_fixed_recall(y_true, scores, r))
        metrics[f'alerts_per_1k_at_recall_{int(r*100)}'] = float(
            alerts_per_k_windows(y_true, scores, r, k=1000)
        )

    return metrics


def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Compute PR curve points."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return pd.DataFrame({
        'precision': precision[:-1],
        'recall': recall[:-1],
        'threshold': thresholds
    })


# =============================================================================
# BSAD Runner
# =============================================================================

def run_bsad(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    settings: Settings
) -> np.ndarray:
    """Run BSAD model and return test scores."""
    # Get arrays from training data
    train_arrays = steps.get_model_arrays(train_df)

    # Train model
    print("  Training BSAD (MCMC)...")
    trace = steps.train_model(train_arrays, settings)

    # Score test data
    test_arrays = steps.get_model_arrays(test_df)

    # Use trained posteriors to score test data
    # Need to map test entities to train entities
    scores = steps.compute_scores(
        test_arrays['y'],
        trace,
        test_arrays['entity_idx']
    )

    return scores['anomaly_score']


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark_single_rate(
    attack_rate: float,
    output_dir: Path,
    seed: int = SEED
) -> Dict[str, Any]:
    """Run benchmark for a single attack rate."""
    print(f"\n{'='*60}")
    print(f"Running benchmark with attack_rate={attack_rate:.1%}")
    print(f"{'='*60}")

    # Settings
    settings = Settings(
        n_entities=N_ENTITIES,
        n_days=N_DAYS,
        attack_rate=attack_rate,
        n_samples=BSAD_SAMPLES,
        n_tune=BSAD_TUNE,
        n_chains=BSAD_CHAINS,
        random_seed=seed,
    )

    # Generate data
    print("\n1. Generating synthetic data...")
    events_df, attacks_df = steps.generate_data(settings)
    modeling_df, metadata = steps.build_features(events_df, settings)

    print(f"   - Total observations: {len(modeling_df)}")
    print(f"   - Attack rate: {metadata['attack_rate']:.2%}")

    # Temporal split
    print("\n2. Temporal train/test split (70/30)...")
    train_df, test_df = temporal_train_test_split(modeling_df, train_ratio=0.7)
    print(f"   - Train: {len(train_df)} observations")
    print(f"   - Test: {len(test_df)} observations")
    print(f"   - Test attack rate: {test_df['has_attack'].mean():.2%}")

    # Get test arrays
    test_arrays = steps.get_model_arrays(test_df)
    y_test = test_arrays['y']
    entity_idx_test = test_arrays['entity_idx']
    y_true = test_df['has_attack'].values.astype(int)

    results = {'attack_rate': attack_rate, 'models': {}}

    # Run BSAD
    print("\n3. Running BSAD...")
    t0 = time.time()
    bsad_scores = run_bsad(train_df, test_df, settings)
    bsad_time = time.time() - t0
    results['models']['BSAD'] = {
        'metrics': compute_all_metrics(y_true, bsad_scores, 'BSAD'),
        'time': bsad_time
    }
    print(f"   - PR-AUC: {results['models']['BSAD']['metrics']['pr_auc']:.3f}")
    print(f"   - Time: {bsad_time:.1f}s")

    # Run baselines (fit on train, score on test)
    print("\n4. Running baselines...")
    train_arrays = steps.get_model_arrays(train_df)

    # Import baseline classes
    from bsad.baselines import (
        NB_MLE, NB_EmpiricalBayes, GLMM_NB, ZScoreBaseline, GlobalNB, GenericBaselines
    )

    baselines = [
        ('NB_MLE', NB_MLE()),
        ('NB_EmpBayes', NB_EmpiricalBayes()),
        ('GLMM_NB', GLMM_NB()),
        ('ZScore', ZScoreBaseline()),
        ('GlobalNB', GlobalNB()),
    ]

    for name, model in baselines:
        print(f"   Running {name}...")
        t0 = time.time()
        model.fit(train_arrays['y'], train_arrays['entity_idx'])
        scores = model.score(y_test, entity_idx_test)
        elapsed = time.time() - t0

        results['models'][name] = {
            'metrics': compute_all_metrics(y_true, scores, name),
            'time': elapsed
        }
        print(f"   - {name} PR-AUC: {results['models'][name]['metrics']['pr_auc']:.3f}")

    # Generic baselines
    print("   Running generic baselines (IF, LOF, OCSVM)...")
    X_train = GenericBaselines.get_features(train_df)
    X_test = GenericBaselines.get_features(test_df)

    # These need to be fit differently (on features, not counts)
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    generic_models = [
        ('IsolationForest', IsolationForest(contamination=attack_rate, random_state=seed)),
        ('OCSVM', OneClassSVM(nu=attack_rate, kernel='rbf', gamma='auto')),
    ]

    for name, model in generic_models:
        t0 = time.time()
        model.fit(X_train)
        if hasattr(model, 'decision_function'):
            scores = -model.decision_function(X_test)
        else:
            scores = -model.score_samples(X_test)
        elapsed = time.time() - t0

        results['models'][name] = {
            'metrics': compute_all_metrics(y_true, scores, name),
            'time': elapsed
        }
        print(f"   - {name} PR-AUC: {results['models'][name]['metrics']['pr_auc']:.3f}")

    # LOF (novelty=True for scoring new data)
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=attack_rate, novelty=True)
    lof.fit(X_train)
    lof_scores = -lof.decision_function(X_test)
    elapsed = time.time() - t0
    results['models']['LOF'] = {
        'metrics': compute_all_metrics(y_true, lof_scores, 'LOF'),
        'time': elapsed
    }
    print(f"   - LOF PR-AUC: {results['models']['LOF']['metrics']['pr_auc']:.3f}")

    # Store PR curves for plotting
    results['pr_curves'] = {}
    all_scores = {
        'BSAD': bsad_scores,
    }
    for name, model in baselines:
        model.fit(train_arrays['y'], train_arrays['entity_idx'])
        all_scores[name] = model.score(y_test, entity_idx_test)

    for name, scores in all_scores.items():
        curve = compute_pr_curve(y_true, scores)
        results['pr_curves'][name] = curve.to_dict('list')

    return results


def run_full_benchmark(
    attack_rates: List[float],
    output_dir: Path,
    seed: int = SEED
) -> Dict[str, Any]:
    """Run benchmark across multiple attack rates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'config': {
            'n_entities': N_ENTITIES,
            'n_days': N_DAYS,
            'bsad_samples': BSAD_SAMPLES,
            'k_values': K_VALUES,
            'recall_targets': RECALL_TARGETS,
        },
        'results_by_rate': {}
    }

    for rate in attack_rates:
        results = run_benchmark_single_rate(rate, output_dir, seed)
        all_results['results_by_rate'][f'{rate:.2f}'] = results

    return all_results


# =============================================================================
# Visualization
# =============================================================================

def create_benchmark_plots(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. PR-AUC comparison across attack rates
    fig, ax = plt.subplots(figsize=(12, 6))

    rates = []
    models = []
    pr_aucs = []

    for rate_str, rate_results in results['results_by_rate'].items():
        rate = float(rate_str)
        for model_name, model_results in rate_results['models'].items():
            rates.append(f"{rate:.0%}")
            models.append(model_name)
            pr_aucs.append(model_results['metrics']['pr_auc'])

    df_plot = pd.DataFrame({'Attack Rate': rates, 'Model': models, 'PR-AUC': pr_aucs})

    # Color palette: BSAD highlighted
    model_order = ['BSAD', 'NB_MLE', 'NB_EmpBayes', 'GLMM_NB', 'GlobalNB', 'ZScore',
                   'IsolationForest', 'LOF', 'OCSVM']
    colors = ['#e74c3c'] + ['#3498db'] * 5 + ['#95a5a6'] * 3  # Red for BSAD, blue for count-specific, gray for generic

    sns.barplot(data=df_plot, x='Attack Rate', y='PR-AUC', hue='Model',
                hue_order=[m for m in model_order if m in df_plot['Model'].unique()],
                palette=colors[:len(df_plot['Model'].unique())], ax=ax)

    ax.set_title('PR-AUC Comparison Across Attack Rates', fontsize=14)
    ax.set_ylabel('PR-AUC')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    fig.savefig(output_dir / '01_pr_auc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Recall@k curves
    fig, axes = plt.subplots(1, len(results['results_by_rate']), figsize=(5*len(results['results_by_rate']), 5))
    if len(results['results_by_rate']) == 1:
        axes = [axes]

    for ax, (rate_str, rate_results) in zip(axes, results['results_by_rate'].items()):
        rate = float(rate_str)

        for model_name in ['BSAD', 'NB_EmpBayes', 'GLMM_NB', 'IsolationForest']:
            if model_name not in rate_results['models']:
                continue
            metrics = rate_results['models'][model_name]['metrics']
            ks = [k for k in K_VALUES if f'recall_at_{k}' in metrics]
            recalls = [metrics[f'recall_at_{k}'] for k in ks]

            style = '-o' if model_name == 'BSAD' else '--s'
            ax.plot(ks, recalls, style, label=model_name, linewidth=2 if model_name == 'BSAD' else 1)

        ax.set_xlabel('k (top alerts)')
        ax.set_ylabel('Recall@k')
        ax.set_title(f'Attack Rate = {rate:.0%}')
        ax.legend()
        ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(output_dir / '02_recall_at_k.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Alerts per 1k at fixed recall
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use 30% recall as example
    data = []
    for rate_str, rate_results in results['results_by_rate'].items():
        rate = float(rate_str)
        for model_name, model_results in rate_results['models'].items():
            alerts = model_results['metrics'].get('alerts_per_1k_at_recall_30', None)
            if alerts is not None:
                data.append({
                    'Attack Rate': f"{rate:.0%}",
                    'Model': model_name,
                    'Alerts/1k': alerts
                })

    df_alerts = pd.DataFrame(data)
    if not df_alerts.empty:
        sns.barplot(data=df_alerts, x='Attack Rate', y='Alerts/1k', hue='Model', ax=ax)
        ax.set_title('Alerts per 1000 Windows at 30% Recall', fontsize=14)
        ax.set_ylabel('Alerts per 1000 windows')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    fig.savefig(output_dir / '03_alerts_at_recall.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Summary table
    summary_rows = []
    for rate_str, rate_results in results['results_by_rate'].items():
        for model_name, model_results in rate_results['models'].items():
            m = model_results['metrics']
            summary_rows.append({
                'Attack Rate': rate_str,
                'Model': model_name,
                'PR-AUC': f"{m['pr_auc']:.3f}",
                'ROC-AUC': f"{m['roc_auc']:.3f}",
                'Recall@50': f"{m.get('recall_at_50', 0):.3f}",
                'Precision@50': f"{m.get('precision_at_50', 0):.3f}",
                'Time (s)': f"{model_results['time']:.1f}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'summary_metrics.csv', index=False)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BSAD Benchmark Suite')
    parser.add_argument('--output', '-o', type=str, default='outputs/benchmark',
                        help='Output directory for results')
    parser.add_argument('--attack-rates', '-a', type=float, nargs='+', default=ATTACK_RATES,
                        help='Attack rates to test (e.g., 0.01 0.02 0.05)')
    parser.add_argument('--seed', '-s', type=int, default=SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer samples, single attack rate')

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.quick:
        global BSAD_SAMPLES, BSAD_TUNE, N_ENTITIES, N_DAYS
        BSAD_SAMPLES = 500
        BSAD_TUNE = 200
        N_ENTITIES = 50
        N_DAYS = 14
        attack_rates = [0.02]
    else:
        attack_rates = args.attack_rates

    print(f"BSAD Benchmark Suite")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Attack rates: {attack_rates}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")

    # Run benchmark
    results = run_full_benchmark(attack_rates, output_dir, args.seed)

    # Save results
    results_path = output_dir / 'benchmark_results.json'

    # Convert non-serializable items
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
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    create_benchmark_plots(results, output_dir)

    print(f"\nBenchmark complete! Outputs in {output_dir}/")


if __name__ == '__main__':
    main()
