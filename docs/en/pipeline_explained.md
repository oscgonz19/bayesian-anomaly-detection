# Pipeline Explained: Step-by-Step Implementation Guide

## Overview

This document walks through the BSAD pipeline from a data scientist/ML engineer perspective. We'll cover each step in detail, including the code, the reasoning, and practical tips.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Generate   │───▶│   Build     │───▶│   Train     │───▶│   Score     │───▶│  Evaluate   │
│    Data     │    │  Features   │    │   Model     │    │  Anomalies  │    │  Results    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Step 1: Data Generation

### Purpose
Create synthetic security event logs with realistic attack patterns for model development and testing.

### Code Location
`src/bsad/steps.py` → `generate_data()`

### Input/Output
```python
Input:  Settings(n_entities=200, n_days=30, attack_rate=0.02)
Output: (events_df, attacks_df)  # pd.DataFrames
```

### How It Works

```python
def generate_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic security events with injected attacks.

    The generator creates:
    1. Entity base rates from Gamma distribution
    2. Time windows with day/night patterns
    3. Event counts per entity-window
    4. Injected attacks at specified rate
    """
    rng = np.random.default_rng(settings.random_seed)

    # 1. Create entities with varying base rates
    entity_rates = rng.gamma(shape=2, scale=5, size=settings.n_entities)

    # 2. Generate time windows
    start_date = pd.Timestamp('2024-01-01')
    windows = pd.date_range(start_date, periods=settings.n_days * 24, freq='H')

    # 3. Generate events per entity-window
    events = []
    for entity_id in range(settings.n_entities):
        for window in windows:
            # Apply day/night pattern
            hour = window.hour
            time_factor = 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12)

            # Generate count
            rate = entity_rates[entity_id] * time_factor
            count = rng.poisson(rate)

            events.append({
                'entity_id': f'user_{entity_id:04d}',
                'time_window': window,
                'event_count': count,
            })

    events_df = pd.DataFrame(events)

    # 4. Inject attacks
    attacks_df = _inject_attacks(events_df, settings, rng)

    return events_df, attacks_df
```

### Attack Injection Logic

```python
def _inject_attacks(df: pd.DataFrame, settings: Settings, rng) -> pd.DataFrame:
    """Inject attack patterns into subset of entity-windows."""

    n_attack_windows = int(len(df) * settings.attack_rate)
    attack_indices = rng.choice(len(df), size=n_attack_windows, replace=False)

    attack_types = ['brute_force', 'credential_stuffing', 'geo_anomaly', 'device_anomaly']
    attack_multipliers = {
        'brute_force': (10, 50),       # High intensity
        'credential_stuffing': (3, 8), # Moderate intensity
        'geo_anomaly': (1, 3),         # Low intensity (flagged by location)
        'device_anomaly': (1, 2),      # Low intensity (flagged by device)
    }

    attacks = []
    for idx in attack_indices:
        attack_type = rng.choice(attack_types)
        mult_range = attack_multipliers[attack_type]
        multiplier = rng.uniform(*mult_range)

        # Amplify event count
        df.loc[idx, 'event_count'] = int(df.loc[idx, 'event_count'] * multiplier)
        df.loc[idx, 'is_attack'] = True
        df.loc[idx, 'attack_type'] = attack_type

        attacks.append({
            'index': idx,
            'attack_type': attack_type,
            'multiplier': multiplier,
        })

    return pd.DataFrame(attacks)
```

### Practical Tips

1. **Seed Everything**: Always pass `random_seed` for reproducibility
2. **Realistic Patterns**: The day/night sinusoid mimics real user behavior
3. **Varied Intensities**: Different attack types have different detectability
4. **Ground Truth**: Keep `attacks_df` for evaluation later

---

## Step 2: Feature Engineering

### Purpose
Transform raw events into a modeling-ready table with one row per entity-window.

### Code Location
`src/bsad/steps.py` → `build_features()`

### Input/Output
```python
Input:  events_df (raw events), Settings
Output: (modeling_df, metadata)  # pd.DataFrame, dict
```

### How It Works

```python
def build_features(events_df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """
    Build feature table for modeling.

    Key transformations:
    1. Aggregate events by (entity, time_window)
    2. Create numeric entity indices for PyMC
    3. Compute summary statistics
    """
    # 1. Aggregate (if not already aggregated)
    if 'event_count' not in events_df.columns:
        modeling_df = events_df.groupby(['entity_id', 'time_window']).agg(
            event_count=('event_id', 'count'),
            has_attack=('is_attack', 'any'),
        ).reset_index()
    else:
        modeling_df = events_df.copy()

    # 2. Create entity index mapping
    entity_map = {eid: idx for idx, eid in enumerate(modeling_df['entity_id'].unique())}
    modeling_df['entity_idx'] = modeling_df['entity_id'].map(entity_map)

    # 3. Compute metadata
    metadata = {
        'n_entities': len(entity_map),
        'n_windows': modeling_df['time_window'].nunique(),
        'n_observations': len(modeling_df),
        'entity_map': entity_map,
    }

    return modeling_df, metadata
```

### Model Arrays Extraction

```python
def get_model_arrays(modeling_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Extract numpy arrays for PyMC model.

    Returns dict with:
    - 'y': observed event counts
    - 'entity_idx': entity index per observation
    - 'n_entities': total unique entities
    """
    return {
        'y': modeling_df['event_count'].values.astype(np.int64),
        'entity_idx': modeling_df['entity_idx'].values.astype(np.int64),
        'n_entities': modeling_df['entity_idx'].nunique(),
    }
```

### Feature Table Schema

| Column | Type | Purpose |
|--------|------|---------|
| `entity_id` | str | Human-readable identifier |
| `entity_idx` | int | Zero-indexed for PyMC |
| `time_window` | datetime | Window start time |
| `event_count` | int | **Primary feature** |
| `has_attack` | bool | Ground truth label |
| `attack_type` | str | Attack category (optional) |

### Practical Tips

1. **Integer Counts**: PyMC's NegativeBinomial expects integer counts
2. **Contiguous Indices**: `entity_idx` must be 0 to n-1 with no gaps
3. **Memory Efficiency**: Convert to numpy arrays before training

---

## Step 3: Model Training

### Purpose
Fit a hierarchical Bayesian model to learn entity-specific rates with partial pooling.

### Code Location
`src/bsad/steps.py` → `train_model()`

### Input/Output
```python
Input:  arrays (dict from get_model_arrays), Settings
Output: az.InferenceData  # ArviZ trace object
```

### How It Works

```python
def train_model(arrays: dict, settings: Settings) -> az.InferenceData:
    """
    Train hierarchical Negative Binomial model.

    Model structure:
    - Population: mu (mean rate), alpha (concentration)
    - Entity: theta[i] (entity-specific rate)
    - Observation: y ~ NegBinomial(theta[entity], phi)
    """
    with pm.Model() as model:
        # ===== Population-Level Priors =====
        # mu: expected event rate across all entities
        mu = pm.Exponential('mu', lam=0.1)

        # alpha: controls pooling strength
        # High alpha → entity rates cluster around mu
        # Low alpha → entity rates vary widely
        alpha = pm.HalfNormal('alpha', sigma=2)

        # ===== Entity-Level Parameters =====
        # theta: entity-specific rate (partial pooling from Gamma)
        # E[theta] = mu, Var[theta] = mu/alpha
        theta = pm.Gamma(
            'theta',
            alpha=mu * alpha,  # shape
            beta=alpha,         # rate
            shape=arrays['n_entities']
        )

        # ===== Overdispersion =====
        # phi: controls variance beyond Poisson
        # NegBinom variance = mu + mu^2/phi
        phi = pm.HalfNormal('phi', sigma=1)

        # ===== Likelihood =====
        y_obs = pm.NegativeBinomial(
            'y_obs',
            mu=theta[arrays['entity_idx']],
            alpha=phi,
            observed=arrays['y']
        )

        # ===== Sampling =====
        trace = pm.sample(
            draws=settings.n_samples,
            tune=settings.n_tune,
            chains=settings.n_chains,
            random_seed=settings.random_seed,
            cores=settings.n_cores,
            target_accept=settings.target_accept,
            return_inferencedata=True,
        )

    return trace
```

### Understanding the Priors

```
Population Level
================
mu ~ Exp(0.1)
  → Mean = 10
  → Covers typical event rates (1-100+)

alpha ~ HalfNormal(2)
  → Moderate concentration
  → Allows model to learn pooling strength

Entity Level
============
theta[i] ~ Gamma(mu*alpha, alpha)
  → E[theta] = mu (all entities share population mean)
  → Var[theta] = mu/alpha (variance controlled by alpha)

Observation Level
=================
y ~ NegBinomial(theta[entity], phi)
  → E[y] = theta
  → Var[y] = theta + theta²/phi (overdispersion)
```

### Convergence Diagnostics

```python
def get_diagnostics(trace: az.InferenceData) -> dict:
    """Check MCMC convergence."""
    summary = az.summary(trace, var_names=['mu', 'alpha', 'phi'])

    return {
        'r_hat_max': float(summary['r_hat'].max()),
        'ess_bulk_min': int(summary['ess_bulk'].min()),
        'ess_tail_min': int(summary['ess_tail'].min()),
        'divergences': int(trace.sample_stats.diverging.sum()),
    }
```

**Quality Thresholds:**
| Metric | Good | Bad |
|--------|------|-----|
| R-hat | < 1.01 | > 1.1 |
| ESS (bulk) | > 400 | < 100 |
| ESS (tail) | > 400 | < 100 |
| Divergences | 0 | > 0 |

### Practical Tips

1. **Start Small**: Test with 100 samples first to catch errors
2. **Watch Divergences**: Non-zero divergences → reparameterize
3. **Target Accept**: Increase to 0.95 if divergences occur
4. **JAX Backend**: Use `pm.sample(nuts_sampler='numpyro')` for 10x speedup

---

## Step 4: Anomaly Scoring

### Purpose
Compute anomaly scores and uncertainty intervals using the posterior predictive distribution.

### Code Location
`src/bsad/steps.py` → `compute_scores()`, `compute_intervals()`

### Input/Output
```python
Input:  y (counts), trace, entity_idx
Output: dict with 'anomaly_score', 'mean', 'std'
```

### How It Works

```python
def compute_scores(y: np.ndarray, trace: az.InferenceData,
                   entity_idx: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute anomaly scores from posterior predictive.

    Score = -log P(y | posterior)

    Interpretation:
    - Low score (0-3): Normal behavior
    - Medium score (3-6): Unusual
    - High score (6+): Highly anomalous
    """
    from scipy.stats import nbinom
    from scipy.special import logsumexp

    # Extract posterior samples
    theta = trace.posterior['theta'].values  # (chains, draws, entities)
    phi = trace.posterior['phi'].values      # (chains, draws)

    # Flatten across chains
    theta_flat = theta.reshape(-1, theta.shape[-1])  # (samples, entities)
    phi_flat = phi.flatten()                          # (samples,)

    n_samples = len(phi_flat)
    scores = np.zeros(len(y))

    for i, (y_i, idx_i) in enumerate(zip(y, entity_idx)):
        # Get entity's rate samples
        theta_i = theta_flat[:, idx_i]

        # NegBinom parameterization: n=phi, p=phi/(phi+mu)
        n_param = phi_flat
        p_param = phi_flat / (phi_flat + theta_i)

        # Log probability under each posterior sample
        log_probs = nbinom.logpmf(y_i, n=n_param, p=p_param)

        # Average over posterior (log-sum-exp trick)
        avg_log_prob = logsumexp(log_probs) - np.log(n_samples)

        # Anomaly score = negative log probability
        scores[i] = -avg_log_prob

    return {
        'anomaly_score': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
    }
```

### Credible Intervals

```python
def compute_intervals(trace: az.InferenceData, entity_idx: np.ndarray,
                      credible_mass: float = 0.9) -> dict[str, np.ndarray]:
    """
    Compute posterior predictive intervals for each observation.

    Returns lower/upper bounds of 90% credible interval.
    """
    from scipy.stats import nbinom

    theta = trace.posterior['theta'].values.reshape(-1, -1)
    phi = trace.posterior['phi'].values.flatten()

    n_obs = len(entity_idx)
    lower = np.zeros(n_obs)
    upper = np.zeros(n_obs)
    median = np.zeros(n_obs)

    alpha = (1 - credible_mass) / 2  # e.g., 0.05 for 90% CI

    for i, idx in enumerate(entity_idx):
        theta_i = theta[:, idx]

        # Generate predictive samples
        n_param = phi
        p_param = phi / (phi + theta_i)
        y_pred = nbinom.rvs(n=n_param, p=p_param)

        lower[i] = np.quantile(y_pred, alpha)
        upper[i] = np.quantile(y_pred, 1 - alpha)
        median[i] = np.median(y_pred)

    return {
        'lower': lower,
        'upper': upper,
        'median': median,
    }
```

### Creating the Scored DataFrame

```python
def create_scored_df(modeling_df: pd.DataFrame, scores: dict,
                     intervals: dict) -> pd.DataFrame:
    """Combine modeling data with scores and intervals."""
    scored_df = modeling_df.copy()

    scored_df['anomaly_score'] = scores['anomaly_score']
    scored_df['pred_lower'] = intervals['lower']
    scored_df['pred_upper'] = intervals['upper']
    scored_df['pred_median'] = intervals['median']

    # Sort by score (highest first)
    scored_df = scored_df.sort_values('anomaly_score', ascending=False)

    return scored_df
```

### Practical Tips

1. **Vectorize When Possible**: The loop is slow; batch operations help
2. **Use Log-Sum-Exp**: Avoid numerical underflow with small probabilities
3. **CI Width**: Wide intervals → uncertain entity, narrow → confident
4. **Score Calibration**: Scores aren't probabilities; calibrate if needed

---

## Step 5: Evaluation

### Purpose
Measure detection performance against ground truth labels.

### Code Location
`src/bsad/steps.py` → `evaluate()`

### Input/Output
```python
Input:  scored_df (with 'anomaly_score' and 'has_attack')
Output: dict with metrics
```

### How It Works

```python
def evaluate(scored_df: pd.DataFrame, k_values: list[int] | None = None) -> dict:
    """
    Evaluate anomaly detection performance.

    Metrics:
    - PR-AUC: Precision-Recall Area Under Curve
    - ROC-AUC: Receiver Operating Characteristic AUC
    - Recall@K: Fraction of attacks in top K scores
    """
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, auc
    )

    y_true = scored_df['has_attack'].astype(int).values
    scores = scored_df['anomaly_score'].values

    # PR-AUC (primary metric for rare events)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    # ROC-AUC (for reference)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Recall@K
    k_values = k_values or [50, 100, 200]
    recall_at_k = {}
    total_attacks = y_true.sum()

    for k in k_values:
        if k <= len(scored_df):
            top_k = scored_df.head(k)['has_attack'].sum()
            recall_at_k[f'recall_at_{k}'] = top_k / total_attacks if total_attacks > 0 else 0

    return {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'attack_rate': float(y_true.mean()),
        'n_attacks': int(total_attacks),
        'n_total': len(scored_df),
        **recall_at_k,
    }
```

### Why PR-AUC Over ROC-AUC?

```
Example with 2% attack rate:

Model that always predicts "no attack":
- Accuracy: 98%
- ROC-AUC: ~0.50 (random)
- PR-AUC: ~0.02 (matches attack rate)

Our model:
- ROC-AUC: 0.93 (looks great!)
- PR-AUC: 0.85 (actually great!)

PR-AUC is harder to game with class imbalance.
```

### Practical Tips

1. **Sort First**: `scored_df` must be sorted by score (descending)
2. **Choose K Wisely**: K should match analyst capacity (50-100 typical)
3. **Baseline Comparison**: Random PR-AUC = attack_rate
4. **Per-Attack-Type**: Break down metrics by attack type for insights

---

## Complete Pipeline Run

### Using the Pipeline Class

```python
from bsad import Settings, Pipeline

# Configure
settings = Settings(
    n_entities=200,
    n_days=30,
    n_samples=2000,
    random_seed=42,
)

# Run
pipeline = Pipeline(settings)
state = pipeline.run_demo()

# Access results
print(f"PR-AUC: {state.metrics['pr_auc']:.3f}")
print(f"Top anomaly: {state.scored_df.iloc[0]['entity_id']}")
```

### Using the CLI

```bash
# Full demo
bsad demo --n-entities 200 --n-days 30 --samples 2000

# Step by step
bsad generate-data --output data/events.parquet
bsad train --input data/events.parquet --output outputs/model.nc
bsad score --model outputs/model.nc --output outputs/scores.parquet
bsad evaluate --scores outputs/scores.parquet
```

### Using Step Functions Directly

```python
from bsad import steps, io
from bsad.config import Settings

settings = Settings(n_entities=100, n_days=14)

# Step 1: Generate
events_df, attacks_df = steps.generate_data(settings)

# Step 2: Features
modeling_df, metadata = steps.build_features(events_df, settings)
arrays = steps.get_model_arrays(modeling_df)

# Step 3: Train
trace = steps.train_model(arrays, settings)

# Step 4: Score
scores = steps.compute_scores(arrays['y'], trace, arrays['entity_idx'])
intervals = steps.compute_intervals(trace, arrays['entity_idx'])
scored_df = steps.create_scored_df(modeling_df, scores, intervals)

# Step 5: Evaluate
metrics = steps.evaluate(scored_df)
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Divergences | Posterior geometry issues | Increase `target_accept` to 0.95 |
| Low ESS | Correlated samples | Increase `n_samples` |
| Slow sampling | Large dataset | Use JAX backend |
| Memory error | Too many entities | Reduce batch size or use VI |
| Score NaN | Zero probability observation | Add small epsilon to log |

### Debug Mode

```python
# Minimal run for debugging
settings = Settings(
    n_entities=20,
    n_days=7,
    n_samples=100,
    n_tune=100,
    n_chains=1,
)
```

---

## Summary

| Step | Input | Output | Key Function |
|------|-------|--------|--------------|
| 1. Generate | Settings | events_df, attacks_df | `generate_data()` |
| 2. Features | events_df | modeling_df, arrays | `build_features()` |
| 3. Train | arrays | trace (InferenceData) | `train_model()` |
| 4. Score | y, trace | scored_df | `compute_scores()` |
| 5. Evaluate | scored_df | metrics dict | `evaluate()` |

The pipeline is designed to be:
- **Modular**: Each step is a pure function
- **Transparent**: Inspect any intermediate state
- **Reproducible**: Seed all random operations
- **Extensible**: Add new steps without modifying existing ones
