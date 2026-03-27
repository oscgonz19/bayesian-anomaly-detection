"""
Posterior-based anomaly scoring for the hierarchical NB model.

SCORING APPROACH
----------------
Score = -log p(y | posterior)

This is the negative marginal log-likelihood averaged over posterior samples.
Higher scores mean the observed count is less likely under the model's
learned baseline for that entity.

COMPUTATION
-----------
We loop over all posterior samples and compute the NB log-PMF, then
use log-sum-exp for numerical stability:

  score_i = -[logsumexp_s log p(y_i | theta[entity_i]^(s), phi^(s)) - log S]

where S is the number of posterior samples (chains × draws).

UNCERTAINTY
-----------
score_std measures how much the score varies across posterior samples.
High score_std means the model is uncertain about whether this observation
is anomalous—important context for analyst triage.

SCALABILITY NOTE
----------------
The current implementation iterates over samples in a Python loop with
scipy.stats.nbinom. This is correct but slow for large posteriors.
A vectorized or approximate scorer would be a natural next step.
"""

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp


def compute_scores(
    y: np.ndarray,
    trace: az.InferenceData,
    entity_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute per-observation anomaly scores from MCMC posterior.

    Args:
        y:          Observed event counts, shape (n_obs,).
        trace:      Fitted ArviZ InferenceData (must have posterior group).
        entity_idx: Integer entity index for each observation, shape (n_obs,).

    Returns:
        dict with keys:
          anomaly_score – mean negative log-likelihood, shape (n_obs,)
          score_std     – std across posterior samples, shape (n_obs,)
          score_lower   – 5th percentile across samples, shape (n_obs,)
          score_upper   – 95th percentile across samples, shape (n_obs,)
    """
    theta = trace.posterior["theta"].values  # (chains, draws, entities)
    phi = trace.posterior["phi"].values      # (chains, draws)

    n_chains, n_draws, n_entities = theta.shape
    theta_flat = theta.reshape(-1, n_entities)  # (S, entities)
    phi_flat = phi.reshape(-1)                  # (S,)
    n_samples = theta_flat.shape[0]
    n_obs = len(y)

    log_likelihoods = np.zeros((n_samples, n_obs))

    for s in range(n_samples):
        mu_s = theta_flat[s, entity_idx]
        phi_s = phi_flat[s]
        # NB parameterisation: n=phi, p=phi/(phi+mu)
        p_param = phi_s / (phi_s + mu_s)
        log_likelihoods[s, :] = stats.nbinom.logpmf(y, n=phi_s, p=p_param)

    # Log-marginal likelihood averaged over posterior samples
    avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples)
    anomaly_scores = -avg_log_lik

    # Per-sample score for uncertainty quantification
    individual_scores = -log_likelihoods
    return {
        "anomaly_score": anomaly_scores,
        "score_std": np.std(individual_scores, axis=0),
        "score_lower": np.percentile(individual_scores, 5, axis=0),
        "score_upper": np.percentile(individual_scores, 95, axis=0),
    }


def compute_intervals(
    trace: az.InferenceData,
    entity_idx: np.ndarray,
    credible_mass: float = 0.9,
) -> dict[str, np.ndarray]:
    """
    Compute posterior predictive intervals for each observation.

    Uses posterior_predictive group if available (exact); otherwise
    approximates via mean and NB variance formula.

    Args:
        trace:          Fitted ArviZ InferenceData.
        entity_idx:     Integer entity index per observation.
        credible_mass:  Width of the credible interval (default 0.9 = 90%).

    Returns:
        dict with keys: predicted_mean, predicted_lower, predicted_upper.
    """
    if hasattr(trace, "posterior_predictive") and "y" in trace.posterior_predictive:
        ppc = trace.posterior_predictive["y"].values  # (chains, draws, obs)
        ppc_flat = ppc.reshape(-1, ppc.shape[-1])
        alpha = (1 - credible_mass) / 2
        return {
            "predicted_mean": np.mean(ppc_flat, axis=0),
            "predicted_lower": np.percentile(ppc_flat, alpha * 100, axis=0),
            "predicted_upper": np.percentile(ppc_flat, (1 - alpha) * 100, axis=0),
        }

    # Approximate via NB variance: Var = mu + mu^2 / phi
    theta = trace.posterior["theta"].values
    phi = trace.posterior["phi"].values
    theta_flat = theta.reshape(-1, theta.shape[-1])
    phi_flat = phi.reshape(-1)
    means = theta_flat[:, entity_idx]
    mean = np.mean(means, axis=0)
    avg_phi = np.mean(phi_flat)
    variance = mean + mean**2 / avg_phi
    std = np.sqrt(variance)
    from scipy import stats as _stats
    z = _stats.norm.ppf((1 + credible_mass) / 2)
    return {
        "predicted_mean": mean,
        "predicted_lower": np.maximum(0, mean - z * std),
        "predicted_upper": mean + z * std,
    }


def create_scored_df(
    modeling_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    intervals: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Join anomaly scores and predictive intervals to the modeling table.

    Args:
        modeling_df: Feature table (output of build_modeling_table).
        scores:      Output of compute_scores.
        intervals:   Output of compute_intervals.

    Returns:
        Scored DataFrame sorted by anomaly_score descending, with
        added columns: anomaly_score, score_std, score_lower,
        score_upper, predicted_mean, predicted_lower, predicted_upper,
        anomaly_rank, exceeds_interval.
    """
    result = modeling_df.copy()

    result["anomaly_score"] = scores["anomaly_score"]
    result["score_std"] = scores["score_std"]
    result["score_lower"] = scores["score_lower"]
    result["score_upper"] = scores["score_upper"]

    result["predicted_mean"] = intervals["predicted_mean"]
    result["predicted_lower"] = intervals["predicted_lower"]
    result["predicted_upper"] = intervals["predicted_upper"]

    result["anomaly_rank"] = (
        result["anomaly_score"].rank(ascending=False, method="first").astype(int)
    )
    result["exceeds_interval"] = result["event_count"] > result["predicted_upper"]

    return result.sort_values("anomaly_score", ascending=False)
