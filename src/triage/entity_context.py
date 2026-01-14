"""
Entity context for alert enrichment.

Key insight: Analysts need context, not just scores.
For each alert, provide:
- Entity baseline behavior
- Historical alert density
- Comparison to similar entities
- Confidence level interpretation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EntityContext:
    """
    Context information for a single entity.

    Provides the "story" behind an anomaly score that
    helps analysts decide whether to investigate.
    """
    entity_id: str
    baseline_mean: float
    baseline_std: float
    current_value: float
    anomaly_score: float
    percentile_rank: float  # How unusual vs all observations
    entity_percentile: float  # How unusual vs this entity's history
    historical_alerts: int  # Past alerts for this entity
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> Dict:
        return {
            "entity_id": self.entity_id,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "current_value": self.current_value,
            "anomaly_score": self.anomaly_score,
            "sigma_deviation": self._sigma_deviation(),
            "percentile_rank": self.percentile_rank,
            "entity_percentile": self.entity_percentile,
            "historical_alerts": self.historical_alerts,
            "confidence": self.confidence,
            "narrative": self._generate_narrative(),
        }

    def _sigma_deviation(self) -> float:
        """How many standard deviations from entity baseline."""
        if self.baseline_std > 0:
            return (self.current_value - self.baseline_mean) / self.baseline_std
        return 0.0

    def _generate_narrative(self) -> str:
        """Generate human-readable explanation."""
        sigma = self._sigma_deviation()

        if sigma > 3:
            deviation_text = f"extremely high ({sigma:.1f}σ above baseline)"
        elif sigma > 2:
            deviation_text = f"significantly high ({sigma:.1f}σ above baseline)"
        elif sigma > 1:
            deviation_text = f"moderately high ({sigma:.1f}σ above baseline)"
        elif sigma < -2:
            deviation_text = f"unusually low ({abs(sigma):.1f}σ below baseline)"
        else:
            deviation_text = "within normal range"

        history_text = (
            f"This entity has {self.historical_alerts} previous alerts."
            if self.historical_alerts > 0
            else "This entity has no previous alerts (first-time anomaly)."
        )

        confidence_text = {
            "high": "High confidence: narrow uncertainty bounds.",
            "medium": "Medium confidence: some uncertainty in baseline.",
            "low": "Low confidence: limited historical data for this entity.",
        }.get(self.confidence, "")

        return f"""
Entity: {self.entity_id}
Current value: {self.current_value:.1f} (baseline: {self.baseline_mean:.1f} ± {self.baseline_std:.1f})
Deviation: {deviation_text}
{history_text}
{confidence_text}
        """.strip()


@dataclass
class EntityHistory:
    """
    Track historical behavior for all entities.
    """
    entity_stats: Dict[str, Dict] = field(default_factory=dict)
    alert_history: Dict[str, List] = field(default_factory=dict)

    def update(
        self,
        entity_id: str,
        value: float,
        is_alert: bool = False,
        timestamp: Optional[str] = None,
    ):
        """Update entity statistics with new observation."""
        if entity_id not in self.entity_stats:
            self.entity_stats[entity_id] = {
                "values": [],
                "count": 0,
            }

        self.entity_stats[entity_id]["values"].append(value)
        self.entity_stats[entity_id]["count"] += 1

        if is_alert:
            if entity_id not in self.alert_history:
                self.alert_history[entity_id] = []
            self.alert_history[entity_id].append({
                "value": value,
                "timestamp": timestamp,
            })

    def get_stats(self, entity_id: str) -> Dict:
        """Get statistics for an entity."""
        if entity_id not in self.entity_stats:
            return {"mean": 0, "std": 0, "count": 0}

        values = np.array(self.entity_stats[entity_id]["values"])
        return {
            "mean": values.mean(),
            "std": values.std() if len(values) > 1 else 0,
            "count": len(values),
            "min": values.min(),
            "max": values.max(),
        }

    def get_alert_count(self, entity_id: str) -> int:
        """Get number of historical alerts for entity."""
        return len(self.alert_history.get(entity_id, []))


def build_entity_history(
    df: pd.DataFrame,
    entity_col: str = "entity",
    value_col: str = "event_count",
    score_col: str = "anomaly_score",
    alert_threshold: Optional[float] = None,
) -> EntityHistory:
    """
    Build entity history from a scored DataFrame.

    Args:
        df: DataFrame with entities and scores
        entity_col: Column name for entity identifier
        value_col: Column name for the count/value
        score_col: Column name for anomaly scores
        alert_threshold: Score threshold to count as alert (optional)

    Returns:
        EntityHistory object
    """
    history = EntityHistory()

    # Determine alert threshold if not provided
    if alert_threshold is None:
        alert_threshold = df[score_col].quantile(0.95)

    for _, row in df.iterrows():
        is_alert = row[score_col] >= alert_threshold
        history.update(
            entity_id=str(row[entity_col]),
            value=row[value_col],
            is_alert=is_alert,
        )

    return history


def enrich_alerts(
    df: pd.DataFrame,
    history: EntityHistory,
    entity_col: str = "entity",
    value_col: str = "event_count",
    score_col: str = "anomaly_score",
    score_std_col: Optional[str] = "score_std",
    top_k: int = 100,
) -> List[Dict]:
    """
    Enrich top alerts with entity context.

    Args:
        df: DataFrame with scored observations
        history: EntityHistory object
        entity_col: Entity column name
        value_col: Value column name
        score_col: Score column name
        score_std_col: Score std column name (optional)
        top_k: Number of top alerts to enrich

    Returns:
        List of enriched alert dictionaries
    """
    # Get top-k by score
    top_df = df.nlargest(top_k, score_col)

    # Compute global percentiles
    all_scores = df[score_col].values
    percentile_map = {
        score: (all_scores < score).mean() * 100
        for score in top_df[score_col].unique()
    }

    enriched = []
    for _, row in top_df.iterrows():
        entity_id = str(row[entity_col])
        stats = history.get_stats(entity_id)

        # Determine confidence
        if stats["count"] >= 50:
            confidence = "high"
        elif stats["count"] >= 10:
            confidence = "medium"
        else:
            confidence = "low"

        # Entity-specific percentile
        entity_values = history.entity_stats.get(entity_id, {}).get("values", [])
        if entity_values:
            entity_pct = (np.array(entity_values) < row[value_col]).mean() * 100
        else:
            entity_pct = 50.0

        context = EntityContext(
            entity_id=entity_id,
            baseline_mean=stats["mean"],
            baseline_std=stats["std"],
            current_value=row[value_col],
            anomaly_score=row[score_col],
            percentile_rank=percentile_map.get(row[score_col], 50),
            entity_percentile=entity_pct,
            historical_alerts=history.get_alert_count(entity_id),
            confidence=confidence,
        )

        enriched.append(context.to_dict())

    return enriched
