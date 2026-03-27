"""
Triage module: From detection to decision.

This module transforms anomaly scores into actionable SOC workflows:
- Risk scoring with uncertainty
- Alert budget calibration
- Ranking metrics for operational evaluation
- Entity context for analyst prioritization
"""

from .calibrate_thresholds import AlertBudget, build_alert_budget_curve, calibrate_threshold
from .entity_context import EntityContext, build_entity_history, enrich_alerts
from .ranking_metrics import (
    alerts_per_k_windows,
    fpr_at_fixed_recall,
    precision_at_k,
    ranking_report,
    recall_at_k,
    workload_reduction,
)
from .risk_score import RiskScorer, compute_risk_score

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
