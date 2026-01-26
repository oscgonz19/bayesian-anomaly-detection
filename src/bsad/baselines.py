"""
Baseline models for fair comparison with BSAD.

These baselines are designed for COUNT DATA with ENTITY STRUCTURE,
making them "fair" competitors unlike generic anomaly detectors (IF, LOF).

Baselines:
1. NB_MLE: Negative Binomial with MLE per entity (no pooling)
2. NB_EmpiricalBayes: NB with shrinkage toward global mean (simple partial pooling)
3. GLMM_NB: Generalized Linear Mixed Model with NB (frequentist hierarchical)
4. ZScore: Simple z-score per entity (non-probabilistic baseline)
5. GlobalNB: Single NB for all entities (complete pooling)

Generic baselines (for reference, not count-specific):
6. IsolationForest
7. LocalOutlierFactor
8. OneClassSVM
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Protocol, Dict, Any, Optional
from dataclasses import dataclass
import warnings


class AnomalyScorer(Protocol):
    """Protocol for anomaly scoring models."""

    def fit(self, y: np.ndarray, entity_idx: np.ndarray) -> "AnomalyScorer":
        """Fit the model."""
        ...

    def score(self, y: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        ...


@dataclass
class BaselineResult:
    """Container for baseline results."""
    name: str
    scores: np.ndarray
    params: Dict[str, Any]
    fit_time: float = 0.0


# =============================================================================
# 1. Negative Binomial MLE (No Pooling)
# =============================================================================

class NB_MLE:
    """
    Negative Binomial with MLE per entity.

    Each entity gets its own (mu, alpha) estimated via MLE.
    No information sharing between entities.

    Anomaly score: -log P(y | MLE params)
    """

    def __init__(self, min_obs_per_entity: int = 3):
        self.min_obs = min_obs_per_entity
        self.entity_params: Dict[int, tuple] = {}
        self.global_params: tuple = (1.0, 1.0)  # Fallback

    def fit(self, y: np.ndarray, entity_idx: np.ndarray) -> "NB_MLE":
        """Fit NB parameters per entity using MLE."""
        unique_entities = np.unique(entity_idx)

        # Global fallback
        self.global_params = self._fit_nb_mle(y)

        for entity in unique_entities:
            mask = entity_idx == entity
            y_entity = y[mask]

            if len(y_entity) >= self.min_obs:
                self.entity_params[entity] = self._fit_nb_mle(y_entity)
            else:
                # Use global for sparse entities
                self.entity_params[entity] = self.global_params

        return self

    def _fit_nb_mle(self, y: np.ndarray) -> tuple:
        """Fit NB via method of moments (fast approximation to MLE)."""
        mean = np.mean(y) + 1e-6
        var = np.var(y) + 1e-6

        # Method of moments for NB
        # E[Y] = mu, Var[Y] = mu + mu^2/alpha
        # => alpha = mu^2 / (var - mu)
        if var > mean:
            alpha = mean**2 / (var - mean)
        else:
            # Underdispersion: use Poisson-like (high alpha)
            alpha = 100.0

        alpha = np.clip(alpha, 0.1, 1000.0)
        return (mean, alpha)

    def score(self, y: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
        """Compute anomaly scores as negative log-likelihood."""
        scores = np.zeros(len(y))

        for i, (yi, ei) in enumerate(zip(y, entity_idx)):
            mu, alpha = self.entity_params.get(ei, self.global_params)
            # NB parameterization: n=alpha, p=alpha/(alpha+mu)
            p = alpha / (alpha + mu)
            scores[i] = -stats.nbinom.logpmf(yi, n=alpha, p=p)

        return scores


# =============================================================================
# 2. Empirical Bayes NB (Simple Shrinkage)
# =============================================================================

class NB_EmpiricalBayes:
    """
    Negative Binomial with Empirical Bayes shrinkage.

    Each entity's rate is shrunk toward the global mean.
    Shrinkage strength depends on sample size.

    Formula: theta_eb[e] = w[e] * theta_mle[e] + (1-w[e]) * theta_global
    where w[e] = n[e] / (n[e] + k) for some k (shrinkage strength)
    """

    def __init__(self, shrinkage_strength: float = 5.0):
        self.k = shrinkage_strength
        self.entity_params: Dict[int, tuple] = {}
        self.global_mu: float = 1.0
        self.global_alpha: float = 1.0

    def fit(self, y: np.ndarray, entity_idx: np.ndarray) -> "NB_EmpiricalBayes":
        """Fit with empirical Bayes shrinkage."""
        # Global estimates
        self.global_mu = np.mean(y) + 1e-6
        var = np.var(y) + 1e-6
        if var > self.global_mu:
            self.global_alpha = self.global_mu**2 / (var - self.global_mu)
        else:
            self.global_alpha = 100.0
        self.global_alpha = np.clip(self.global_alpha, 0.1, 1000.0)

        unique_entities = np.unique(entity_idx)

        for entity in unique_entities:
            mask = entity_idx == entity
            y_entity = y[mask]
            n_entity = len(y_entity)

            # Entity MLE
            mu_mle = np.mean(y_entity) + 1e-6

            # Shrinkage weight
            w = n_entity / (n_entity + self.k)

            # Shrunk estimate
            mu_eb = w * mu_mle + (1 - w) * self.global_mu

            # Use global alpha (could also shrink this)
            self.entity_params[entity] = (mu_eb, self.global_alpha)

        return self

    def score(self, y: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        scores = np.zeros(len(y))

        for i, (yi, ei) in enumerate(zip(y, entity_idx)):
            mu, alpha = self.entity_params.get(ei, (self.global_mu, self.global_alpha))
            p = alpha / (alpha + mu)
            scores[i] = -stats.nbinom.logpmf(yi, n=alpha, p=p)

        return scores


# =============================================================================
# 3. GLMM Negative Binomial (Frequentist Hierarchical)
# =============================================================================

class GLMM_NB:
    """
    Generalized Linear Mixed Model with Negative Binomial.

    Uses statsmodels for frequentist mixed effects.
    Falls back to simpler model if statsmodels unavailable.

    Model: log(mu_i) = beta_0 + u[entity_i]
    where u[entity] ~ N(0, sigma^2)
    """

    def __init__(self):
        self.intercept: float = 0.0
        self.random_effects: Dict[int, float] = {}
        self.alpha: float = 1.0
        self._use_statsmodels: bool = False

    def fit(self, y: np.ndarray, entity_idx: np.ndarray) -> "GLMM_NB":
        """Fit GLMM-NB model."""
        try:
            return self._fit_statsmodels(y, entity_idx)
        except Exception:
            return self._fit_simple(y, entity_idx)

    def _fit_statsmodels(self, y: np.ndarray, entity_idx: np.ndarray) -> "GLMM_NB":
        """Fit using statsmodels MixedLM with NB."""
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.families import NegativeBinomial
        except ImportError:
            return self._fit_simple(y, entity_idx)

        # Create DataFrame
        df = pd.DataFrame({
            'y': y,
            'entity': entity_idx,
            'intercept': 1
        })

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Fit NB GLM with entity as categorical (approximation to GLMM)
                # True GLMM would use mixed effects, but this is a reasonable proxy
                model = sm.GLM(
                    df['y'],
                    sm.add_constant(pd.get_dummies(df['entity'], drop_first=True)),
                    family=NegativeBinomial()
                )
                result = model.fit(disp=0)

                self.intercept = result.params[0]
                self.alpha = 1.0 / result.scale if hasattr(result, 'scale') else 1.0

                # Extract entity effects
                unique_entities = np.unique(entity_idx)
                for i, entity in enumerate(unique_entities):
                    if i == 0:
                        self.random_effects[entity] = 0.0
                    else:
                        param_name = f'entity_{entity}'
                        if param_name in result.params.index:
                            self.random_effects[entity] = result.params[param_name]
                        else:
                            self.random_effects[entity] = 0.0

                self._use_statsmodels = True

        except Exception:
            return self._fit_simple(y, entity_idx)

        return self

    def _fit_simple(self, y: np.ndarray, entity_idx: np.ndarray) -> "GLMM_NB":
        """Simple fallback: log-normal random effects."""
        global_mean = np.mean(y) + 1e-6
        self.intercept = np.log(global_mean)

        # Estimate entity effects
        unique_entities = np.unique(entity_idx)
        for entity in unique_entities:
            mask = entity_idx == entity
            entity_mean = np.mean(y[mask]) + 1e-6
            # Random effect on log scale
            self.random_effects[entity] = np.log(entity_mean) - self.intercept

        # Variance-based alpha
        var = np.var(y) + 1e-6
        if var > global_mean:
            self.alpha = global_mean**2 / (var - global_mean)
        else:
            self.alpha = 100.0
        self.alpha = np.clip(self.alpha, 0.1, 1000.0)

        return self

    def score(self, y: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        scores = np.zeros(len(y))

        for i, (yi, ei) in enumerate(zip(y, entity_idx)):
            re = self.random_effects.get(ei, 0.0)
            mu = np.exp(self.intercept + re)
            p = self.alpha / (self.alpha + mu)
            scores[i] = -stats.nbinom.logpmf(yi, n=self.alpha, p=p)

        return scores


# =============================================================================
# 4. Z-Score Baseline (Per Entity)
# =============================================================================

class ZScoreBaseline:
    """
    Simple z-score per entity.

    Not probabilistic, but a common baseline.
    Score = |y - mean[entity]| / std[entity]
    """

    def __init__(self, min_obs: int = 3):
        self.min_obs = min_obs
        self.entity_stats: Dict[int, tuple] = {}
        self.global_mean: float = 0.0
        self.global_std: float = 1.0

    def fit(self, y: np.ndarray, entity_idx: np.ndarray) -> "ZScoreBaseline":
        """Compute per-entity statistics."""
        self.global_mean = np.mean(y)
        self.global_std = np.std(y) + 1e-6

        unique_entities = np.unique(entity_idx)
        for entity in unique_entities:
            mask = entity_idx == entity
            y_entity = y[mask]

            if len(y_entity) >= self.min_obs:
                self.entity_stats[entity] = (np.mean(y_entity), np.std(y_entity) + 1e-6)
            else:
                self.entity_stats[entity] = (self.global_mean, self.global_std)

        return self

    def score(self, y: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
        """Compute z-scores (absolute value)."""
        scores = np.zeros(len(y))

        for i, (yi, ei) in enumerate(zip(y, entity_idx)):
            mean, std = self.entity_stats.get(ei, (self.global_mean, self.global_std))
            scores[i] = np.abs(yi - mean) / std

        return scores


# =============================================================================
# 5. Global NB (Complete Pooling)
# =============================================================================

class GlobalNB:
    """
    Single Negative Binomial for all entities (complete pooling).

    Ignores entity structure entirely.
    Useful as a lower bound for hierarchical methods.
    """

    def __init__(self):
        self.mu: float = 1.0
        self.alpha: float = 1.0

    def fit(self, y: np.ndarray, entity_idx: np.ndarray) -> "GlobalNB":
        """Fit single NB to all data."""
        self.mu = np.mean(y) + 1e-6
        var = np.var(y) + 1e-6

        if var > self.mu:
            self.alpha = self.mu**2 / (var - self.mu)
        else:
            self.alpha = 100.0
        self.alpha = np.clip(self.alpha, 0.1, 1000.0)

        return self

    def score(self, y: np.ndarray, entity_idx: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (same params for all)."""
        p = self.alpha / (self.alpha + self.mu)
        return -stats.nbinom.logpmf(y, n=self.alpha, p=p)


# =============================================================================
# 6-8. Generic Baselines (Not Count-Specific)
# =============================================================================

class GenericBaselines:
    """
    Wrapper for generic anomaly detection baselines.

    These don't model counts directly but are common comparisons.
    Uses aggregated features, not raw counts.
    """

    @staticmethod
    def get_features(modeling_df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix for generic detectors."""
        feature_cols = [
            'event_count', 'unique_ips', 'unique_endpoints',
            'unique_devices', 'unique_locations', 'failed_count'
        ]
        # Use available columns
        available = [c for c in feature_cols if c in modeling_df.columns]
        if not available:
            available = ['event_count']

        X = modeling_df[available].values
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    @staticmethod
    def isolation_forest(X: np.ndarray, contamination: float = 0.05, seed: int = 42) -> np.ndarray:
        """Isolation Forest scores."""
        clf = IsolationForest(
            contamination=contamination,
            random_state=seed,
            n_estimators=100
        )
        # IF returns negative scores (lower = more anomalous)
        # Negate to match convention (higher = more anomalous)
        return -clf.fit_predict(X).astype(float) - clf.score_samples(X)

    @staticmethod
    def lof(X: np.ndarray, n_neighbors: int = 20, contamination: float = 0.05) -> np.ndarray:
        """Local Outlier Factor scores."""
        clf = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, len(X) - 1),
            contamination=contamination,
            novelty=False
        )
        clf.fit_predict(X)
        # LOF returns negative scores, negate
        return -clf.negative_outlier_factor_

    @staticmethod
    def ocsvm(X: np.ndarray, nu: float = 0.05, seed: int = 42) -> np.ndarray:
        """One-Class SVM scores."""
        clf = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
        clf.fit(X)
        # Decision function: negative = anomalous
        return -clf.decision_function(X)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_all_baselines(
    y: np.ndarray,
    entity_idx: np.ndarray,
    modeling_df: Optional[pd.DataFrame] = None,
    include_generic: bool = True
) -> Dict[str, np.ndarray]:
    """
    Run all baseline models and return scores.

    Args:
        y: Count data (target variable)
        entity_idx: Entity indices
        modeling_df: Optional DataFrame for generic baselines
        include_generic: Whether to include IF/LOF/OCSVM

    Returns:
        Dict mapping model name to anomaly scores
    """
    results = {}

    # Count-specific baselines
    print("  Running NB_MLE (no pooling)...")
    nb_mle = NB_MLE()
    nb_mle.fit(y, entity_idx)
    results['NB_MLE'] = nb_mle.score(y, entity_idx)

    print("  Running NB_EmpiricalBayes (shrinkage)...")
    nb_eb = NB_EmpiricalBayes()
    nb_eb.fit(y, entity_idx)
    results['NB_EmpBayes'] = nb_eb.score(y, entity_idx)

    print("  Running GLMM_NB (frequentist hierarchical)...")
    glmm = GLMM_NB()
    glmm.fit(y, entity_idx)
    results['GLMM_NB'] = glmm.score(y, entity_idx)

    print("  Running ZScore (per entity)...")
    zscore = ZScoreBaseline()
    zscore.fit(y, entity_idx)
    results['ZScore'] = zscore.score(y, entity_idx)

    print("  Running GlobalNB (complete pooling)...")
    global_nb = GlobalNB()
    global_nb.fit(y, entity_idx)
    results['GlobalNB'] = global_nb.score(y, entity_idx)

    # Generic baselines
    if include_generic and modeling_df is not None:
        print("  Running generic baselines (IF, LOF, OCSVM)...")
        X = GenericBaselines.get_features(modeling_df)
        results['IsolationForest'] = GenericBaselines.isolation_forest(X)
        results['LOF'] = GenericBaselines.lof(X)
        results['OCSVM'] = GenericBaselines.ocsvm(X)

    return results
