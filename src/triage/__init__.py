"""
Triage module: From detection to decision.

This module transforms anomaly scores into actionable SOC workflows:
- Risk scoring with uncertainty
- Alert budget calibration
- Ranking metrics for operational evaluation
- Entity context for analyst prioritization
"""

from .risk_score import compute_risk_score, RiskScorer
from .calibrate_thresholds import calibrate_threshold, AlertBudget, build_alert_budget_curve
from .ranking_metrics import (
    precision_at_k,
    recall_at_k,
    fpr_at_fixed_recall,
    alerts_per_k_windows,
    workload_reduction,
    ranking_report
)
from .entity_context import EntityContext, build_entity_history, enrich_alerts

__all__ = [
    "compute_risk_score",
    "RiskScorer",
    "calibrate_threshold",
    "AlertBudget",
    "build_alert_budget_curve",
    "precision_at_k",
    "recall_at_k",
    "fpr_at_fixed_recall",
    "alerts_per_k_windows",
    "workload_reduction",
    "ranking_report",
    "EntityContext",
    "build_entity_history",
    "enrich_alerts",
]
