"""
Threshold calibration for alert budgets.

Key insight: SOCs have limited analyst capacity.
Instead of maximizing detection, we optimize for:
- Fixed alert budget (e.g., 50 alerts/day)
- Fixed recall target (e.g., catch 30% of attacks)
- Maximum workload reduction
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from sklearn.metrics import roc_curve, precision_recall_curve


@dataclass
class AlertBudget:
    """
    Calibrate detection thresholds based on operational constraints.

    Supports three modes:
    1. Fixed alert count (e.g., max 50 alerts/day)
    2. Fixed recall (e.g., detect 30% of attacks)
    3. Fixed FPR (e.g., max 5% false positive rate)
    """
    mode: str = "fixed_recall"  # "fixed_alerts", "fixed_recall", "fixed_fpr"
    target: float = 0.3  # target value for the constraint

    def calibrate(
        self,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        n_windows_per_day: int = 1000,
    ) -> Dict:
        """
        Find threshold that satisfies the budget constraint.

        Args:
            scores: Anomaly/risk scores (higher = more anomalous)
            y_true: Ground truth labels (required for recall/fpr modes)
            n_windows_per_day: Number of time windows per day (for alert count)

        Returns:
            Dict with threshold and operational metrics
        """
        if self.mode == "fixed_alerts":
            return self._calibrate_fixed_alerts(scores, y_true, n_windows_per_day)
        elif self.mode == "fixed_recall":
            return self._calibrate_fixed_recall(scores, y_true)
        elif self.mode == "fixed_fpr":
            return self._calibrate_fixed_fpr(scores, y_true)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _calibrate_fixed_alerts(
        self, scores: np.ndarray, y_true: Optional[np.ndarray], n_windows_per_day: int
    ) -> Dict:
        """Set threshold to generate exactly target alerts per day."""
        # Target alerts as fraction of windows
        target_fraction = self.target / n_windows_per_day
        target_percentile = 100 * (1 - target_fraction)

        threshold = np.percentile(scores, target_percentile)

        alerts = (scores >= threshold).sum()
        alerts_per_day = alerts * (n_windows_per_day / len(scores))

        result = {
            "threshold": threshold,
            "alerts_per_day": alerts_per_day,
            "target_alerts": self.target,
        }

        if y_true is not None:
            y_pred = (scores >= threshold).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            result["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            result["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
            result["fpr"] = fp / (y_true == 0).sum() if (y_true == 0).sum() > 0 else 0

        return result

    def _calibrate_fixed_recall(
        self, scores: np.ndarray, y_true: np.ndarray
    ) -> Dict:
        """Set threshold to achieve target recall."""
        if y_true is None:
            raise ValueError("y_true required for fixed_recall mode")

        fpr, tpr, thresholds = roc_curve(y_true, scores)

        # Find threshold where TPR >= target
        idx = np.searchsorted(tpr, self.target)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1

        threshold = thresholds[idx]
        actual_recall = tpr[idx]
        actual_fpr = fpr[idx]

        y_pred = (scores >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        return {
            "threshold": threshold,
            "target_recall": self.target,
            "actual_recall": actual_recall,
            "fpr": actual_fpr,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "alerts": (scores >= threshold).sum(),
        }

    def _calibrate_fixed_fpr(
        self, scores: np.ndarray, y_true: np.ndarray
    ) -> Dict:
        """Set threshold to achieve target FPR."""
        if y_true is None:
            raise ValueError("y_true required for fixed_fpr mode")

        fpr, tpr, thresholds = roc_curve(y_true, scores)

        # Find threshold where FPR <= target
        idx = np.searchsorted(fpr, self.target)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1

        threshold = thresholds[idx]
        actual_fpr = fpr[idx]
        actual_recall = tpr[idx]

        y_pred = (scores >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        return {
            "threshold": threshold,
            "target_fpr": self.target,
            "actual_fpr": actual_fpr,
            "recall": actual_recall,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "alerts": (scores >= threshold).sum(),
        }


def calibrate_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    mode: str = "fixed_recall",
    target: float = 0.3,
    n_windows_per_day: int = 1000,
) -> Dict:
    """
    Convenience function for threshold calibration.

    Args:
        scores: Anomaly scores
        y_true: Ground truth labels
        mode: "fixed_alerts", "fixed_recall", or "fixed_fpr"
        target: Target value for the constraint
        n_windows_per_day: Windows per day (for alert mode)

    Returns:
        Dict with threshold and metrics
    """
    budget = AlertBudget(mode=mode, target=target)
    return budget.calibrate(scores, y_true, n_windows_per_day)


def build_alert_budget_curve(
    scores: np.ndarray,
    y_true: np.ndarray,
    recall_targets: np.ndarray = None,
) -> pd.DataFrame:
    """
    Build a curve showing alerts required at different recall levels.

    Args:
        scores: Anomaly scores
        y_true: Ground truth labels
        recall_targets: Recall levels to evaluate (default: 0.1 to 0.9)

    Returns:
        DataFrame with recall, fpr, alerts columns
    """
    if recall_targets is None:
        recall_targets = np.arange(0.1, 1.0, 0.1)

    results = []
    for target in recall_targets:
        budget = AlertBudget(mode="fixed_recall", target=target)
        result = budget.calibrate(scores, y_true)
        result["recall_target"] = target
        results.append(result)

    return pd.DataFrame(results)
