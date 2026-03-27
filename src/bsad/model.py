"""
Hierarchical Negative Binomial anomaly detection model.

MODEL SCOPE
-----------
This module implements a single model: a hierarchical Negative Binomial
on aggregated event counts per entity per time window.

What it models:
  - Overdispersed count data (Variance >> Mean)
  - Entity-specific baseline rates via partial pooling (shrinkage)
  - Global population distribution of rates

What it does NOT model:
  - Location, device, IP, or endpoint features (contextual only)
  - Temporal autocorrelation or trend
  - Multi-entity interactions (e.g. coordinated attacks)
  - Continuous covariates

MODEL STRUCTURE
---------------
Population:
  mu    ~ Exponential(rate=mu_prior_rate)      # global mean event rate
  alpha ~ HalfNormal(sigma=alpha_prior_sd)     # concentration / pooling strength

Entity-level:
  theta[e] ~ Gamma(alpha=mu*alpha, beta=alpha) # entity-specific rate

Observation:
  phi   ~ HalfNormal(sigma=overdispersion_prior_sd)  # overdispersion
  y     ~ NegativeBinomial(mu=theta[entity_idx], alpha=phi)

Anomaly score: -log p(y | posterior), computed in bsad.scoring.
"""

import os
from dataclasses import dataclass

import arviz as az
import numpy as np
import pymc as pm


@dataclass
class ModelConfig:
    """Configuration for model building and MCMC sampling.

    Decoupled from GeneratorConfig and FeatureConfig so that
    model hyperparameters can be varied independently.
    """

    # MCMC settings
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 4
    target_accept: float = 0.9

    # Prior hyperparameters
    mu_prior_rate: float = 0.1
    alpha_prior_sd: float = 2.0
    overdispersion_prior_sd: float = 2.0

    # Reproducibility
    random_seed: int = 42


def build_hierarchical_negbinom_model(
    y: np.ndarray,
    entity_idx: np.ndarray,
    n_entities: int,
    config: ModelConfig | None = None,
) -> pm.Model:
    """
    Construct the hierarchical Negative Binomial PyMC model.

    Does NOT fit the model—call fit_model() to run MCMC.

    Args:
        y:           Observed event counts, shape (n_obs,), int.
        entity_idx:  Entity index for each observation, shape (n_obs,),
                     values in [0, n_entities).
        n_entities:  Number of unique entities.
        config:      Model hyperparameters; defaults to ModelConfig().

    Returns:
        Unsampled PyMC model object.
    """
    if config is None:
        config = ModelConfig()

    coords = {
        "entity": np.arange(n_entities),
        "obs": np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:
        entity_idx_data = pm.Data("entity_idx", entity_idx, dims="obs")
        y_data = pm.Data("y_obs", y, dims="obs")

        # Hyperpriors (population level)
        mu = pm.Exponential("mu", lam=config.mu_prior_rate)
        alpha = pm.HalfNormal("alpha", sigma=config.alpha_prior_sd)

        # Entity-level rates (partial pooling)
        theta = pm.Gamma("theta", alpha=mu * alpha, beta=alpha, dims="entity")

        # Overdispersion
        phi = pm.HalfNormal("phi", sigma=config.overdispersion_prior_sd)

        # Likelihood
        pm.NegativeBinomial(
            "y",
            mu=theta[entity_idx_data],
            alpha=phi,
            observed=y_data,
            dims="obs",
        )

    return model


def fit_model(model: pm.Model, config: ModelConfig | None = None) -> az.InferenceData:
    """
    Run MCMC sampling on a pre-built PyMC model.

    Args:
        model:  PyMC model (from build_hierarchical_negbinom_model).
        config: Sampling configuration; defaults to ModelConfig().

    Returns:
        ArviZ InferenceData with posterior and posterior_predictive groups.
    """
    if config is None:
        config = ModelConfig()

    rng = np.random.default_rng(config.random_seed)
    seed = int(rng.integers(0, 2**31))

    with model:
        trace = pm.sample(
            draws=config.n_samples,
            tune=config.n_tune,
            chains=config.n_chains,
            target_accept=config.target_accept,
            random_seed=seed,
            cores=min(config.n_chains, os.cpu_count() or 1),
            return_inferencedata=True,
            progressbar=True,
        )
        trace.extend(pm.sample_posterior_predictive(trace, random_seed=seed + 1))

    return trace


def get_model_diagnostics(trace: az.InferenceData) -> dict:
    """
    Summarise MCMC convergence diagnostics.

    Returns:
        dict with keys:
          r_hat_max    – maximum R-hat across mu, alpha, phi (should be < 1.05)
          ess_bulk_min – minimum bulk ESS (should be > 400)
          divergences  – count of divergent transitions (should be 0)
          converged    – True if r_hat_max < 1.05
    """
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])
    return {
        "r_hat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "divergences": int(trace.sample_stats["diverging"].sum().values),
        "converged": bool(summary["r_hat"].max() < 1.05),
    }
