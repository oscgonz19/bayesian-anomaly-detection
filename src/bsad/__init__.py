"""
Bayesian Security Anomaly Detection (BSAD)

A reproducible Bayesian/MCMC anomaly detection pipeline for security event logs.

Simple usage:
    from bsad.config import Settings
    from bsad.pipeline import Pipeline

    settings = Settings(n_entities=100, n_days=14)
    pipeline = Pipeline(settings)
    state = pipeline.run_demo()

Benchmarking:
    from bsad.baselines import run_all_baselines, NB_MLE, NB_EmpiricalBayes

Calibration:
    from bsad.calibration import calibration_report, reliability_diagram
"""

__version__ = "0.2.0"

# Main exports for simple usage
from bsad.config import Settings
from bsad.pipeline import Pipeline, PipelineState

# Baselines for comparison
from bsad.baselines import (
    NB_MLE,
    NB_EmpiricalBayes,
    GLMM_NB,
    ZScoreBaseline,
    GlobalNB,
    run_all_baselines,
)

# Calibration
from bsad.calibration import (
    calibration_report,
    reliability_diagram,
    expected_calibration_error,
)

__all__ = [
    # Core
    "Settings",
    "Pipeline",
    "PipelineState",
    # Baselines
    "NB_MLE",
    "NB_EmpiricalBayes",
    "GLMM_NB",
    "ZScoreBaseline",
    "GlobalNB",
    "run_all_baselines",
    # Calibration
    "calibration_report",
    "reliability_diagram",
    "expected_calibration_error",
]
