"""
Risk scoring: Transform anomaly scores into actionable risk metrics.

Key insight: Raw anomaly scores are not enough for SOC triage.
We need to incorporate:
- Score magnitude (how anomalous?)
- Score uncertainty (how confident?)
- Entity context (is this entity usually noisy?)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskScorer:
    """
    Compute risk scores from anomaly detection outputs.

    Risk = f(anomaly_score, confidence, entity_baseline)

    Attributes:
        score_weight: Weight for raw anomaly score (default 0.5)
        confidence_weight: Weight for confidence (1/uncertainty) (default 0.3)
        novelty_weight: Weight for entity novelty (default 0.2)
    """
    score_weight: float = 0.5
    confidence_weight: float = 0.3
    novelty_weight: float = 0.2

    def __post_init__(self):
        total = self.score_weight + self.confidence_weight + self.novelty_weight
        if not np.isclose(total, 1.0):
            # Normalize weights
            self.score_weight /= total
            self.confidence_weight /= total
            self.novelty_weight /= total

    def compute(
        self,
        anomaly_scores: np.ndarray,
        score_std: Optional[np.ndarray] = None,
        entity_history_counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute composite risk scores.

        Args:
            anomaly_scores: Raw anomaly scores (higher = more anomalous)
            score_std: Standard deviation of scores (uncertainty)
            entity_history_counts: Number of historical observations per entity

        Returns:
            risk_scores: Composite risk scores in [0, 1]
        """
        n = len(anomaly_scores)

        # Normalize anomaly scores to [0, 1]
        score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
        if score_max > score_min:
            norm_scores = (anomaly_scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.zeros(n)

        # Confidence component (inverse of uncertainty)
        if score_std is not None and score_std.max() > 0:
            # Higher uncertainty -> lower confidence -> lower risk boost
            confidence = 1.0 / (1.0 + score_std / score_std.max())
        else:
            confidence = np.ones(n) * 0.5

        # Novelty component (fewer observations -> higher novelty)
        if entity_history_counts is not None and entity_history_counts.max() > 0:
            # Fewer observations = more novel = higher risk
            novelty = 1.0 - (entity_history_counts / entity_history_counts.max())
        else:
            novelty = np.ones(n) * 0.5

        # Composite risk score
        risk = (
            self.score_weight * norm_scores +
            self.confidence_weight * confidence +
            self.novelty_weight * novelty
        )

        return risk


def compute_risk_score(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    std_col: Optional[str] = "score_std",
    entity_col: Optional[str] = "entity",
    weights: tuple = (0.5, 0.3, 0.2),
) -> pd.Series:
    """
    Convenience function to compute risk scores from a DataFrame.

    Args:
        df: DataFrame with anomaly scores
        score_col: Column name for anomaly scores
        std_col: Column name for score std (optional)
        entity_col: Column name for entity (optional)
        weights: (score_weight, confidence_weight, novelty_weight)

    Returns:
        Series with risk scores
    """
    scorer = RiskScorer(*weights)

    scores = df[score_col].values

    std = df[std_col].values if std_col and std_col in df.columns else None

    # Compute entity history counts
    history = None
    if entity_col and entity_col in df.columns:
        history = df.groupby(entity_col)[score_col].transform("count").values

    return pd.Series(scorer.compute(scores, std, history), index=df.index, name="risk_score")
