"""
Detection and operational evaluation metrics.

METRIC PHILOSOPHY
-----------------
Standard ML metrics (PR-AUC, ROC-AUC) measure ranking quality globally.
SOC operations care about a different question: given an alert budget, how
much of the attack surface is covered?

This module provides both:
  - Standard ranking metrics: compute_pr_auc, compute_roc_auc
  - Operational metrics: compute_recall_at_k, compute_precision_at_k
  - Combined report: compute_all_metrics, format_metrics_report

LABEL SEMANTICS
---------------
y_true: binary array (1 = attack window, 0 = benign window).
scores:  anomaly scores (higher = more anomalous).

Labels are not used during model training. They are used here for
evaluation of detection quality on labelled hold-out data.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def compute_recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Fraction of all true attacks captured in the top-k scored observations.

    "Of all attacks in this dataset, what fraction land in my top-k alerts?"

    Args:
        y_true: Binary ground-truth labels.
        scores: Anomaly scores (higher = more anomalous).
        k:      Number of top alerts to consider.

    Returns:
        Recall@k in [0, 1]. Returns 0.0 if there are no positives.
    """
    n_positives = int(y_true.sum())
    if n_positives == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[-k:]
    return float(y_true[top_k_idx].sum() / n_positives)


def compute_precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Fraction of top-k alerts that are true attacks.

    "Of my top-k alerts, how many are genuine?"

    Args:
        y_true: Binary ground-truth labels.
        scores: Anomaly scores (higher = more anomalous).
        k:      Number of top alerts to consider.

    Returns:
        Precision@k in [0, 1].
    """
    top_k_idx = np.argsort(scores)[-k:]
    return float(y_true[top_k_idx].mean())


def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Area under the Precision-Recall curve (average precision).

    Preferred over ROC-AUC for class-imbalanced problems (rare attacks).
    A random scorer achieves PR-AUC ≈ attack_rate; a perfect scorer
    achieves PR-AUC = 1.0.

    Args:
        y_true: Binary ground-truth labels.
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        PR-AUC (average precision) in [0, 1].
    """
    return float(average_precision_score(y_true, scores))


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Area under the Receiver Operating Characteristic curve.

    Args:
        y_true: Binary ground-truth labels.
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        ROC-AUC in [0, 1].
    """
    return float(roc_auc_score(y_true, scores))


def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    """
    Compute the full set of detection and operational metrics.

    Args:
        y_true:   Binary ground-truth labels.
        scores:   Anomaly scores.
        k_values: Alert-cutoffs for precision@k / recall@k.
                  Defaults to [10, 25, 50, 100].

    Returns:
        dict with keys:
          pr_auc, roc_auc, n_observations, n_positives, attack_rate,
          recall_at_<k>, precision_at_<k> for each k,
          pr_curve (dict with precision/recall lists).
    """
    if k_values is None:
        k_values = [10, 25, 50, 100]

    metrics: dict = {
        "pr_auc": compute_pr_auc(y_true, scores),
        "roc_auc": compute_roc_auc(y_true, scores),
        "n_observations": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "attack_rate": float(y_true.mean()),
    }

    for k in k_values:
        if k <= len(y_true):
            metrics[f"recall_at_{k}"] = compute_recall_at_k(y_true, scores, k)
            metrics[f"precision_at_{k}"] = compute_precision_at_k(y_true, scores, k)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

    return metrics


def format_metrics_report(metrics: dict) -> str:
    """
    Format a metrics dict as a human-readable string for CLI output.

    Args:
        metrics: Dict from compute_all_metrics.

    Returns:
        Multi-line formatted string.
    """
    lines = [
        f"PR-AUC:         {metrics.get('pr_auc', 0):.3f}",
        f"ROC-AUC:        {metrics.get('roc_auc', 0):.3f}",
        f"Attack rate:    {metrics.get('attack_rate', 0):.2%}",
        f"Observations:   {metrics.get('n_observations', 0):,}",
        f"Positives:      {metrics.get('n_positives', 0):,}",
    ]

    # Add all Recall@k and Precision@k lines found in the dict
    recall_keys = sorted(
        [k for k in metrics if k.startswith("recall_at_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    for key in recall_keys:
        k = key.split("_")[-1]
        lines.append(f"Recall@{k:<6}  {metrics[key]:.3f}")

    precision_keys = sorted(
        [k for k in metrics if k.startswith("precision_at_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    for key in precision_keys:
        k = key.split("_")[-1]
        lines.append(f"Precision@{k:<4}  {metrics[key]:.3f}")

    return "\n".join(lines)
