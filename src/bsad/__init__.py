"""
Bayesian Security Anomaly Detection (BSAD)

A reproducible Bayesian/MCMC anomaly detection pipeline for security event logs.

Simple usage:
    from bsad.config import Settings
    from bsad.pipeline import Pipeline

    settings = Settings(n_entities=100, n_days=14)
    pipeline = Pipeline(settings)
    state = pipeline.run_demo()
"""

__version__ = "0.1.0"

# Main exports for simple usage
from bsad.config import Settings
from bsad.pipeline import Pipeline, PipelineState

__all__ = ["Settings", "Pipeline", "PipelineState"]
