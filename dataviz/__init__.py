"""
Data Visualization Scripts for BSAD

This package contains visualization scripts for all stages of the
Bayesian Security Anomaly Detection pipeline.

Scripts:
    - 01_data_exploration.py: Raw event data visualization
    - 02_feature_analysis.py: Feature engineering visualizations
    - 03_model_diagnostics.py: MCMC and model diagnostics
    - 04_anomaly_results.py: Anomaly detection results
    - 05_evaluation_plots.py: Performance metrics visualization
    - 06_full_report.py: Complete visualization report
"""

__all__ = [
    "plot_config",
    "save_figure",
]

import matplotlib.pyplot as plt
from pathlib import Path

# Default plot configuration
plot_config = {
    "figure.figsize": (12, 8),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def save_figure(fig: plt.Figure, name: str, output_dir: str = "outputs/figures") -> Path:
    """Save figure to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return filepath
