"""
Ranking metrics for operational evaluation.

Key insight: ROC-AUC is not actionable for SOC.
What matters is:
- How many real attacks in my top-k alerts?
- How much workload reduction vs baseline?
- How many false positives per true positive?
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import roc_curve


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Precision in top-k ranked items.

    "Of my top k alerts, how many are real attacks?"

    Args:
        y_true: Binary ground truth labels
        scores: Anomaly scores (higher = more anomalous)
        k: Number of top items to consider

    Returns:
        Precision@k value
    """
    if k > len(scores):
        k = len(scores)

    top_k_idx = np.argsort(scores)[-k:]
    return y_true[top_k_idx].mean()


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Recall in top-k ranked items.

    "What fraction of all attacks are in my top k alerts?"

    Args:
        y_true: Binary ground truth labels
        scores: Anomaly scores (higher = more anomalous)
        k: Number of top items to consider

    Returns:
        Recall@k value
    """
    if k > len(scores):
        k = len(scores)

    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0

    top_k_idx = np.argsort(scores)[-k:]
    tp = y_true[top_k_idx].sum()
    return tp / n_positives


def fpr_at_fixed_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.3
) -> float:
    """
    False positive rate at a fixed recall level.

    "To catch 30% of attacks, what fraction of normals do I falsely flag?"

    Args:
        y_true: Binary ground truth labels
        scores: Anomaly scores (higher = more anomalous)
        target_recall: Target recall level (default 0.3)

    Returns:
        FPR at the given recall
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Find index where TPR >= target_recall
    idx = np.searchsorted(tpr, target_recall)
    if idx >= len(fpr):
        idx = len(fpr) - 1

    return fpr[idx]


def alerts_per_k_windows(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.3,
    k: int = 1000
) -> float:
    """
    Number of alerts generated per k windows at fixed recall.

    "If I process 1000 windows and want 30% recall, how many alerts?"

    Args:
        y_true: Binary ground truth labels
        scores: Anomaly scores
        target_recall: Target recall level
        k: Number of windows to normalize to

    Returns:
        Alerts per k windows
    """
    fpr = fpr_at_fixed_recall(y_true, scores, target_recall)

    n_positives = y_true.sum()
    n_negatives = len(y_true) - n_positives
    n_total = len(y_true)

    # At target recall: TP = recall * n_positives, FP = fpr * n_negatives
    tp = target_recall * n_positives
    fp = fpr * n_negatives
    total_alerts = tp + fp

    # Normalize to k windows
    return (total_alerts / n_total) * k


def workload_reduction(
    y_true: np.ndarray,
    scores_baseline: np.ndarray,
    scores_model: np.ndarray,
    target_recall: float = 0.3,
) -> Dict[str, float]:
    """
    Workload reduction compared to baseline.

    "How many fewer alerts does my model generate vs baseline?"

    Args:
        y_true: Binary ground truth labels
        scores_baseline: Baseline model scores
        scores_model: New model scores
        target_recall: Fixed recall for comparison

    Returns:
        Dict with baseline alerts, model alerts, reduction factor
    """
    alerts_baseline = alerts_per_k_windows(y_true, scores_baseline, target_recall)
    alerts_model = alerts_per_k_windows(y_true, scores_model, target_recall)

    reduction = alerts_baseline / alerts_model if alerts_model > 0 else float('inf')

    return {
        "baseline_alerts_per_1k": alerts_baseline,
        "model_alerts_per_1k": alerts_model,
        "reduction_factor": reduction,
        "percent_reduction": (1 - alerts_model / alerts_baseline) * 100 if alerts_baseline > 0 else 0,
    }


def ranking_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    ks: List[int] = [10, 25, 50, 100],
    recalls: List[float] = [0.1, 0.2, 0.3, 0.5],
) -> pd.DataFrame:
    """
    Generate comprehensive ranking metrics report.

    Args:
        y_true: Binary ground truth labels
        scores: Anomaly scores
        ks: List of k values for precision/recall@k
        recalls: List of recall targets for FPR analysis

    Returns:
        DataFrame with all metrics
    """
    metrics = []

    # Precision@k and Recall@k
    for k in ks:
        metrics.append({
            "metric": f"Precision@{k}",
            "value": precision_at_k(y_true, scores, k),
        })
        metrics.append({
            "metric": f"Recall@{k}",
            "value": recall_at_k(y_true, scores, k),
        })

    # FPR at fixed recall
    for r in recalls:
        metrics.append({
            "metric": f"FPR@Recall={r}",
            "value": fpr_at_fixed_recall(y_true, scores, r),
        })
        metrics.append({
            "metric": f"Alerts/1k@Recall={r}",
            "value": alerts_per_k_windows(y_true, scores, r),
        })

    return pd.DataFrame(metrics)
