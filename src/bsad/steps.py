"""
Pipeline steps as thin orchestration functions.

Each function is a thin wrapper that delegates to the appropriate
focused module. This file exists for backward compatibility with
pipeline.py and the CLI—new code should import from the dedicated
modules directly:

  bsad.data_generator  – synthetic data generation
  bsad.features        – feature engineering
  bsad.model           – model building and fitting
  bsad.scoring         – posterior anomaly scoring
  bsad.evaluation      – detection and operational metrics

WHAT BELONGS HERE
-----------------
Nothing statistically meaningful lives here. This file only:
  1. Adapts Settings → module-specific configs
  2. Calls the module functions
  3. Returns the same types that pipeline.py expects

ATTACK-TYPE DETECTABILITY NOTE
-------------------------------
Four attack types are injected in synthetic data:
  - brute_force         Detectable: large count burst on one entity-day
  - credential_stuffing Partially detectable: count burst if events are high
  - geo_anomaly         NOT reliably detectable: count elevation only if
                        attacker generates many events; location not modeled
  - device_anomaly      NOT reliably detectable: device features not modeled;
                        new-device attacks are a realistic false-negative class
"""

from pathlib import Path
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bsad import evaluation as _eval
from bsad import scoring as _scoring
from bsad.config import Settings
from bsad.data_generator import GeneratorConfig, generate_synthetic_data
from bsad.features import FeatureConfig, build_modeling_table
from bsad.features import get_model_arrays as _get_model_arrays
from bsad.model import ModelConfig, build_hierarchical_negbinom_model, fit_model, get_model_diagnostics

AttackType = Literal["brute_force", "credential_stuffing", "geo_anomaly", "device_anomaly", "none"]


# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================


def generate_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic security event logs with attack patterns."""
    config = GeneratorConfig(
        n_users=settings.n_entities,
        n_days=settings.n_days,
        n_ips=settings.n_ips,
        n_endpoints=settings.n_endpoints,
        events_per_user_day_mean=settings.events_per_user_day_mean,
        attack_rate=settings.attack_rate,
        brute_force_multiplier=settings.brute_force_multiplier,
        credential_stuffing_users=settings.credential_stuffing_users,
        credential_stuffing_events_per_user=settings.credential_stuffing_events_per_user,
        device_anomaly_new_devices=settings.device_anomaly_new_devices,
        random_seed=settings.random_seed,
    )
    return generate_synthetic_data(config)


# =============================================================================
# Step 2: Build Features
# =============================================================================


def build_features(events_df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Transform raw events into a modeling table with windowed features."""
    config = FeatureConfig(
        entity_column=settings.entity_column,
        window_size=settings.window_size,
        include_temporal=settings.include_temporal,
    )
    return build_modeling_table(events_df, config)


# =============================================================================
# Step 3: Get Model Arrays
# =============================================================================


def get_model_arrays(modeling_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract numpy arrays for the PyMC model."""
    return _get_model_arrays(modeling_df)


# =============================================================================
# Step 4: Train Model
# =============================================================================


def train_model(arrays: dict, settings: Settings) -> az.InferenceData:
    """Build and fit the hierarchical Negative Binomial model."""
    config = ModelConfig(
        n_samples=settings.n_samples,
        n_tune=settings.n_tune,
        n_chains=settings.n_chains,
        target_accept=settings.target_accept,
        mu_prior_rate=settings.mu_prior_rate,
        alpha_prior_sd=settings.alpha_prior_sd,
        overdispersion_prior_sd=settings.overdispersion_prior_sd,
        random_seed=settings.random_seed,
    )
    model = build_hierarchical_negbinom_model(
        y=arrays["y"],
        entity_idx=arrays["entity_idx"],
        n_entities=arrays["n_entities"],
        config=config,
    )
    return fit_model(model, config)


# =============================================================================
# Step 5 & 6: Score and Build Scored DataFrame
# =============================================================================


def compute_scores(
    y: np.ndarray,
    trace: az.InferenceData,
    entity_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute anomaly scores from the MCMC posterior."""
    return _scoring.compute_scores(y, trace, entity_idx)


def compute_intervals(
    trace: az.InferenceData,
    entity_idx: np.ndarray,
    credible_mass: float = 0.9,
) -> dict[str, np.ndarray]:
    """Compute posterior predictive intervals for each observation."""
    return _scoring.compute_intervals(trace, entity_idx, credible_mass)


def create_scored_df(
    modeling_df: pd.DataFrame,
    scores: dict,
    intervals: dict,
) -> pd.DataFrame:
    """Join scores and intervals to modeling table, sorted by anomaly score."""
    return _scoring.create_scored_df(modeling_df, scores, intervals)


# =============================================================================
# Step 7: Evaluate
# =============================================================================


def evaluate(scored_df: pd.DataFrame, k_values: list[int] | None = None) -> dict:
    """Compute detection and operational evaluation metrics."""
    if k_values is None:
        k_values = [10, 25, 50, 100]
    y_true = scored_df["has_attack"].astype(int).values
    scores = scored_df["anomaly_score"].values
    return _eval.compute_all_metrics(y_true, scores, k_values=k_values)


# =============================================================================
# Step 8: Create Plots
# =============================================================================


def create_plots(
    scored_df: pd.DataFrame,
    metrics: dict,
    trace: az.InferenceData | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate diagnostic and result visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plots = {}

    path = output_dir / "score_distribution.png"
    _plot_score_distribution(scored_df, path)
    plots["score_distribution"] = path

    path = output_dir / "precision_recall_curve.png"
    _plot_pr_curve(metrics, path)
    plots["precision_recall_curve"] = path

    path = output_dir / "top_anomalies.png"
    _plot_top_anomalies(scored_df, path)
    plots["top_anomalies"] = path

    if trace is not None:
        import arviz as az
        path = output_dir / "model_diagnostics.png"
        az.plot_trace(trace, var_names=["mu", "alpha", "phi"], figsize=(14, 8), compact=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plots["model_diagnostics"] = path

    return plots


def _plot_score_distribution(scored_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    benign = scored_df[~scored_df["has_attack"]]["anomaly_score"]
    attack = scored_df[scored_df["has_attack"]]["anomaly_score"]
    axes[0].hist(benign, bins=50, alpha=0.7, label=f"Benign (n={len(benign):,})", color="steelblue", density=True)
    axes[0].hist(attack, bins=50, alpha=0.7, label=f"Attack (n={len(attack):,})", color="crimson", density=True)
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Anomaly Score Distribution by Class")
    axes[0].legend()
    plot_data = scored_df[["anomaly_score", "has_attack"]].copy()
    plot_data["Class"] = plot_data["has_attack"].map({True: "Attack", False: "Benign"})
    sns.boxplot(data=plot_data, x="Class", y="anomaly_score", hue="Class", ax=axes[1],
                palette={"Benign": "steelblue", "Attack": "crimson"}, legend=False)
    axes[1].set_ylabel("Anomaly Score")
    axes[1].set_title("Score Distribution by Class")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pr_curve(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if "pr_curve" in metrics:
        ax.plot(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], linewidth=2, color="steelblue")
        ax.fill_between(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], alpha=0.2, color="steelblue")
    ax.axhline(y=metrics.get("attack_rate", 0.02), color="gray", linestyle="--",
               label=f"Baseline (attack rate = {metrics.get('attack_rate', 0.02):.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (PR-AUC = {metrics.get('pr_auc', 0):.3f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_top_anomalies(scored_df: pd.DataFrame, path: Path, n: int = 20) -> None:
    from matplotlib.patches import Patch
    top_df = scored_df.head(n).copy().sort_values("anomaly_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))
    colors = ["crimson" if attack else "steelblue" for attack in top_df["has_attack"]]
    y_pos = np.arange(len(top_df))
    lower_err = np.maximum(0, top_df["anomaly_score"] - top_df["score_lower"])
    upper_err = np.maximum(0, top_df["score_upper"] - top_df["anomaly_score"])
    ax.barh(y_pos, top_df["anomaly_score"], xerr=[lower_err, upper_err], color=colors, alpha=0.8, capsize=3)
    labels = [
        f"{row['user_id']} ({row['window'].strftime('%m-%d') if hasattr(row['window'], 'strftime') else row['window']})"
        for _, row in top_df.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"Top {n} Anomalies (Red = Attack, Blue = Benign)")
    legend_elements = [Patch(facecolor="crimson", alpha=0.8, label="Attack"), Patch(facecolor="steelblue", alpha=0.8, label="Benign")]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Step 9: Model Diagnostics
# =============================================================================


def get_diagnostics(trace: az.InferenceData) -> dict:
    """Get MCMC convergence diagnostics."""
    return get_model_diagnostics(trace)
