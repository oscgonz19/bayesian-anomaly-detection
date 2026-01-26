"""
Calibration metrics and reliability analysis.

For anomaly detection, "calibration" means:
- Does P(anomaly | score > threshold) match the predicted probability?
- Are uncertainty bounds reliable?

Metrics:
- ECE (Expected Calibration Error)
- Reliability diagrams
- Coverage analysis for credible intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy.special import expit  # Sigmoid


def scores_to_probabilities(
    scores: np.ndarray,
    method: str = 'sigmoid',
    temperature: float = 1.0
) -> np.ndarray:
    """
    Convert anomaly scores to probabilities.

    Args:
        scores: Anomaly scores (higher = more anomalous)
        method: 'sigmoid', 'rank', or 'minmax'
        temperature: Scaling factor for sigmoid

    Returns:
        Probabilities in [0, 1]
    """
    if method == 'sigmoid':
        # Center scores and apply sigmoid
        centered = (scores - np.median(scores)) / (np.std(scores) + 1e-6)
        return expit(centered / temperature)

    elif method == 'rank':
        # Rank-based probability
        ranks = np.argsort(np.argsort(scores))
        return ranks / len(scores)

    elif method == 'minmax':
        # Min-max normalization
        min_s, max_s = scores.min(), scores.max()
        return (scores - min_s) / (max_s - min_s + 1e-6)

    else:
        raise ValueError(f"Unknown method: {method}")


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where:
    - B_b is the set of samples in bin b
    - acc(B_b) is the accuracy (fraction of positives)
    - conf(B_b) is the mean predicted probability

    Args:
        y_true: Binary ground truth (0/1)
        probs: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value and bin details
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_details = []

    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin > 0:
            accuracy = y_true[mask].mean()
            confidence = probs[mask].mean()
            bin_ece = (n_in_bin / len(y_true)) * abs(accuracy - confidence)
            ece += bin_ece

            bin_details.append({
                'bin': i,
                'lower': bin_edges[i],
                'upper': bin_edges[i + 1],
                'n_samples': int(n_in_bin),
                'accuracy': float(accuracy),
                'confidence': float(confidence),
                'gap': float(abs(accuracy - confidence))
            })
        else:
            bin_details.append({
                'bin': i,
                'lower': bin_edges[i],
                'upper': bin_edges[i + 1],
                'n_samples': 0,
                'accuracy': None,
                'confidence': None,
                'gap': None
            })

    return float(ece), {'bins': bin_details, 'n_bins': n_bins}


def maximum_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    _, details = expected_calibration_error(y_true, probs, n_bins)
    gaps = [b['gap'] for b in details['bins'] if b['gap'] is not None]
    return max(gaps) if gaps else 0.0


def reliability_diagram(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    title: str = 'Reliability Diagram'
) -> Tuple[plt.Figure, Dict]:
    """
    Create reliability diagram (calibration plot).

    Args:
        y_true: Binary ground truth
        probs: Predicted probabilities
        n_bins: Number of bins
        ax: Matplotlib axes (optional)
        title: Plot title

    Returns:
        Figure and calibration metrics
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ece, details = expected_calibration_error(y_true, probs, n_bins)

    # Extract bin data
    bins_with_data = [b for b in details['bins'] if b['n_samples'] > 0]
    confidences = [b['confidence'] for b in bins_with_data]
    accuracies = [b['accuracy'] for b in bins_with_data]
    n_samples = [b['n_samples'] for b in bins_with_data]

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)

    # Bar plot for gaps (background)
    bin_width = 1.0 / n_bins
    for b in bins_with_data:
        ax.bar(
            b['confidence'],
            b['accuracy'],
            width=bin_width * 0.8,
            alpha=0.5,
            color='steelblue',
            edgecolor='black'
        )

    # Scatter plot
    ax.scatter(confidences, accuracies, s=100, c='steelblue', edgecolors='black', zorder=5)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (Accuracy)')
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    metrics = {
        'ece': ece,
        'mce': maximum_calibration_error(y_true, probs, n_bins),
        'bin_details': details
    }

    return fig, metrics


def interval_coverage(
    y_observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.9
) -> Dict[str, float]:
    """
    Check if credible intervals have correct coverage.

    Args:
        y_observed: Observed values
        lower: Lower bounds of intervals
        upper: Upper bounds of intervals
        nominal_coverage: Expected coverage (e.g., 0.9 for 90%)

    Returns:
        Coverage statistics
    """
    in_interval = (y_observed >= lower) & (y_observed <= upper)
    actual_coverage = in_interval.mean()

    # Miscoverage: how far from nominal?
    miscoverage = abs(actual_coverage - nominal_coverage)

    return {
        'nominal_coverage': nominal_coverage,
        'actual_coverage': float(actual_coverage),
        'miscoverage': float(miscoverage),
        'n_observations': len(y_observed),
        'n_covered': int(in_interval.sum()),
        'coverage_is_conservative': actual_coverage > nominal_coverage
    }


def calibration_report(
    scored_df: pd.DataFrame,
    score_column: str = 'anomaly_score',
    label_column: str = 'has_attack',
    n_bins: int = 10
) -> Dict:
    """
    Generate full calibration report.

    Args:
        scored_df: DataFrame with scores and labels
        score_column: Column name for anomaly scores
        label_column: Column name for ground truth
        n_bins: Number of bins for calibration

    Returns:
        Comprehensive calibration report
    """
    scores = scored_df[score_column].values
    y_true = scored_df[label_column].values.astype(int)

    report = {}

    # Try different probability conversion methods
    for method in ['sigmoid', 'rank', 'minmax']:
        probs = scores_to_probabilities(scores, method=method)
        ece, details = expected_calibration_error(y_true, probs, n_bins)
        mce = maximum_calibration_error(y_true, probs, n_bins)

        report[f'{method}_conversion'] = {
            'ece': ece,
            'mce': mce,
            'bins': details['bins']
        }

    # Best method
    best_method = min(report.keys(), key=lambda k: report[k]['ece'])
    report['best_method'] = best_method.replace('_conversion', '')
    report['best_ece'] = report[best_method]['ece']

    # Interval coverage (if available)
    if 'predicted_lower' in scored_df.columns and 'predicted_upper' in scored_df.columns:
        coverage = interval_coverage(
            scored_df['event_count'].values,
            scored_df['predicted_lower'].values,
            scored_df['predicted_upper'].values,
            nominal_coverage=0.9
        )
        report['interval_coverage'] = coverage

    return report


def plot_calibration_comparison(
    scored_df: pd.DataFrame,
    output_path: Optional[str] = None,
    score_column: str = 'anomaly_score',
    label_column: str = 'has_attack'
) -> plt.Figure:
    """
    Plot calibration comparison across methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scores = scored_df[score_column].values
    y_true = scored_df[label_column].values.astype(int)

    methods = ['sigmoid', 'rank', 'minmax']
    titles = ['Sigmoid Scaling', 'Rank-Based', 'Min-Max Scaling']

    for ax, method, title in zip(axes, methods, titles):
        probs = scores_to_probabilities(scores, method=method)
        reliability_diagram(y_true, probs, n_bins=10, ax=ax, title=title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
