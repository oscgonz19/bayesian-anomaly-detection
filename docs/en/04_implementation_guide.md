# Implementation Guide

## Table of Contents
1. [Synthetic Data Generation](#synthetic-data-generation)
2. [Feature Engineering Pipeline](#feature-engineering-pipeline)
3. [Model Training](#model-training)
4. [Anomaly Scoring](#anomaly-scoring)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Visualization](#visualization)

---

## Synthetic Data Generation

### Overview

The data generator creates realistic security event logs with known attack patterns. This allows:
- Model development without sensitive real data
- Controlled evaluation with ground truth labels
- Reproducible experiments

### Generator Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GeneratorConfig                          │
│  n_users, n_ips, n_days, attack_rate, random_seed          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               generate_baseline_events()                    │
│  - User activity patterns (log-normal rates)               │
│  - Diurnal patterns (business hours bias)                  │
│  - Day-of-week effects (weekday bias)                      │
│  - Primary IP/location/device per user                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Attack Injection                          │
│  - inject_brute_force_attack()                             │
│  - inject_credential_stuffing_attack()                     │
│  - inject_geo_anomaly_attack()                             │
│  - inject_device_anomaly_attack()                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output DataFrames                        │
│  events_df: All events with is_attack, attack_type labels  │
│  attacks_df: Attack metadata for analysis                  │
└─────────────────────────────────────────────────────────────┘
```

### Baseline Event Generation

```python
def generate_baseline_events(config: GeneratorConfig, rng: np.random.Generator):
    # 1. User-specific activity rates (heterogeneous baseline)
    user_rates = rng.lognormal(
        mean=np.log(config.events_per_user_day_mean),
        sigma=0.5,
        size=config.n_users,
    )
    # Result: Some users very active (rate=20), others sparse (rate=2)

    # 2. Temporal patterns
    for day in range(n_days):
        dow_multiplier = 1.0 if weekday else 0.3  # Weekend reduction

        for user in users:
            n_events = rng.poisson(user_rates[user] * dow_multiplier)

            for event in range(n_events):
                # Diurnal pattern: Beta(2,2) mapped to business hours
                hour = int(rng.beta(2, 2) * 14 + 7) % 24

                # Primary IP/location with small variation
                ip = primary_ip if rng.random() < 0.9 else random_ip
                location = primary_loc if rng.random() < 0.95 else random_loc
```

### Attack Patterns

#### 1. Brute Force Attack

```python
def inject_brute_force_attack(df, config, rng):
    """
    Characteristics:
    - Single IP → Single user
    - 50-200 events in 1-hour window
    - 90% failed attempts (401), 10% success at end
    - Unusual source location (TOR, VPN, Unknown)
    """
    n_events = rng.integers(50, 200)
    for i in range(n_events):
        status = 401 if i < n_events - 1 else 200  # Final success
        # All events concentrated in single hour
```

#### 2. Credential Stuffing

```python
def inject_credential_stuffing_attack(df, config, rng):
    """
    Characteristics:
    - Single IP → Multiple users (10-30)
    - 3-10 attempts per user
    - Lower success rate (15%)
    - Spread across day
    """
    n_target_users = rng.integers(10, 30)
    for user in target_users:
        n_attempts = rng.integers(3, 10)
        # Distributed attempts throughout day
```

#### 3. Geographic Anomaly

```python
def inject_geo_anomaly_attack(df, config, rng):
    """
    Characteristics:
    - Legitimate credentials from suspicious location
    - Successful access (compromised account)
    - Higher data transfer (exfiltration)
    - Unusual locations: North-Korea, Iran, TOR, VPN
    """
    locations = ["North-Korea", "Iran", "TOR-Exit", "Suspicious-Proxy"]
    bytes_transferred = rng.lognormal(8, 1)  # Higher than normal
```

#### 4. Device Anomaly

```python
def inject_device_anomaly_attack(df, config, rng):
    """
    Characteristics:
    - Single user with many new device fingerprints
    - Indicates account sharing or compromise
    - Multiple new devices in short period (3-8)
    """
    n_new_devices = rng.integers(3, 8)
    for device in new_devices:
        # Generate unique fingerprint
        # Multiple events per device
```

### Event Schema

| Field | Type | Description |
|-------|------|-------------|
| timestamp | datetime | Event timestamp |
| user_id | string | User identifier (user_0001) |
| ip_address | string | Source IP (ip_0042 or attack_ip_XXXX) |
| endpoint | string | API endpoint (/api/v1/login) |
| status_code | int | HTTP status (200, 401, 403, 500) |
| location | string | Geographic region |
| device_fingerprint | string | Device identifier (MD5 hash) |
| bytes_transferred | int | Request/response size |
| is_attack | bool | Ground truth label |
| attack_type | string | Attack category or "none" |

---

## Feature Engineering Pipeline

### Time Window Aggregation

```python
def create_time_windows(df: pd.DataFrame, config: FeatureConfig):
    """
    Aggregate raw events into entity-window features.

    Input: Raw events (one row per event)
    Output: Aggregated features (one row per entity-window)
    """
    # Create window identifier
    df["window"] = df["timestamp"].dt.floor(config.window_size)  # "1D", "1H"

    # Aggregate
    grouped = df.groupby([entity_column, "window"]).agg({
        "timestamp": "count",           # event_count
        "ip_address": "nunique",        # unique_ips
        "endpoint": "nunique",          # unique_endpoints
        "device_fingerprint": "nunique",# unique_devices
        "location": "nunique",          # unique_locations
        "bytes_transferred": "sum",     # bytes_total
        "is_attack": "any",             # has_attack (ground truth)
    })
```

### Feature Definitions

| Feature | Computation | Anomaly Signal |
|---------|-------------|----------------|
| event_count | COUNT(*) | High count → brute force |
| unique_ips | COUNT(DISTINCT ip) | Many IPs → distributed attack |
| unique_endpoints | COUNT(DISTINCT endpoint) | Many endpoints → recon |
| unique_devices | COUNT(DISTINCT device) | Many devices → account sharing |
| unique_locations | COUNT(DISTINCT location) | Many locations → geo anomaly |
| failed_count | COUNT(status IN (4xx, 5xx)) | Many failures → brute force |
| bytes_total | SUM(bytes) | High bytes → exfiltration |

### Temporal Features

```python
def add_temporal_features(df: pd.DataFrame):
    """Add time-based features for pattern detection."""
    df["hour"] = df["window"].dt.hour           # 0-23
    df["day_of_week"] = df["window"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_business_hours"] = (
        (df["hour"] >= 9) &
        (df["hour"] <= 17) &
        (~df["is_weekend"])
    )
```

### Entity-Level Statistics

```python
def add_entity_features(df: pd.DataFrame, entity_column: str):
    """
    Add entity historical statistics for context.

    These help identify deviations from entity-specific baseline.
    """
    entity_stats = df.groupby(entity_column)["event_count"].agg(["mean", "std"])

    df = df.merge(entity_stats, on=entity_column)
    df["count_zscore"] = (df["event_count"] - df["entity_mean"]) / df["entity_std"]
```

### Entity Encoding

```python
def encode_entity_ids(df: pd.DataFrame, entity_column: str):
    """
    Create integer encoding for PyMC indexing.

    entity_idx: 0, 1, 2, ..., n_entities-1
    """
    unique_entities = df[entity_column].unique()
    mapping = {entity: idx for idx, entity in enumerate(unique_entities)}
    df["entity_idx"] = df[entity_column].map(mapping)
    return df, mapping
```

### Output Schema

| Column | Type | Usage |
|--------|------|-------|
| entity_idx | int | Model indexing |
| window | datetime | Time reference |
| event_count | int | **Target variable (y)** |
| unique_* | int | Additional features |
| has_attack | bool | Ground truth |
| attack_type | str | Attack category |

---

## Model Training

### Training Pipeline

```python
def train_pipeline(events_df: pd.DataFrame, config: ModelConfig):
    # 1. Feature engineering
    modeling_df, metadata = build_modeling_table(events_df)
    arrays = get_model_arrays(modeling_df)

    # 2. Build model
    model = build_hierarchical_negbinom_model(
        y=arrays["y"],
        entity_idx=arrays["entity_idx"],
        n_entities=metadata["n_entities"],
        config=config,
    )

    # 3. MCMC sampling
    trace = fit_model(model, config)

    # 4. Diagnostics
    diagnostics = get_model_diagnostics(trace)

    return trace, modeling_df, diagnostics
```

### Fitting Process

```python
def fit_model(model: pm.Model, config: ModelConfig) -> az.InferenceData:
    with model:
        # NUTS sampling with automatic tuning
        trace = pm.sample(
            draws=config.n_samples,      # 2000 posterior samples
            tune=config.n_tune,          # 1000 warmup samples
            chains=config.n_chains,      # 4 independent chains
            target_accept=config.target_accept,  # 0.9 acceptance
            cores=min(config.n_chains, os.cpu_count()),
            return_inferencedata=True,
        )

        # Add posterior predictive for scoring
        trace.extend(pm.sample_posterior_predictive(trace))

    return trace
```

### Trace Structure

The returned `InferenceData` contains:

```python
trace.posterior          # Posterior samples
  - mu:    (chain, draw)           # Shape: (4, 2000)
  - alpha: (chain, draw)           # Shape: (4, 2000)
  - phi:   (chain, draw)           # Shape: (4, 2000)
  - theta: (chain, draw, entity)   # Shape: (4, 2000, n_entities)

trace.posterior_predictive
  - y:     (chain, draw, obs)      # Shape: (4, 2000, n_obs)

trace.sample_stats
  - diverging: (chain, draw)       # Divergence flags
  - energy:    (chain, draw)       # Hamiltonian energy
```

---

## Anomaly Scoring

### Scoring Algorithm

```python
def compute_anomaly_scores(y_observed, trace, entity_idx):
    """
    Compute anomaly scores from posterior predictive.

    Score = -log p(y_observed | posterior)

    Higher score = more anomalous (less probable under model)
    """
    # Extract posterior samples
    theta = trace.posterior["theta"].values  # (chains, draws, entities)
    phi = trace.posterior["phi"].values      # (chains, draws)

    # Flatten chains: (chains * draws, ...)
    theta_flat = theta.reshape(-1, n_entities)
    phi_flat = phi.reshape(-1)

    n_samples = theta_flat.shape[0]  # 8000 samples (4 chains × 2000 draws)

    # Compute log-likelihood for each posterior sample
    log_likelihoods = np.zeros((n_samples, n_obs))

    for s in range(n_samples):
        mu_s = theta_flat[s, entity_idx]  # Entity-specific rates
        phi_s = phi_flat[s]

        # Negative Binomial log PMF
        log_likelihoods[s, :] = stats.nbinom.logpmf(
            y_observed,
            n=phi_s,                    # scipy's n parameter
            p=phi_s / (phi_s + mu_s)    # scipy's p parameter
        )

    # Average log-likelihood (log-sum-exp for numerical stability)
    avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples)

    # Anomaly score = negative log likelihood
    anomaly_scores = -avg_log_lik

    return anomaly_scores
```

### Log-Sum-Exp Trick

For numerical stability when averaging probabilities:

```python
# Naive (unstable):
avg_prob = np.mean(np.exp(log_likelihoods), axis=0)
avg_log_lik = np.log(avg_prob)

# Stable (log-sum-exp):
from scipy.special import logsumexp
avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples)
```

### Score Interpretation

```
score = -log p(y | model)

score = 0:   p = 1.0    (perfectly expected)
score = 2:   p = 0.14   (somewhat unlikely)
score = 5:   p = 0.007  (very unlikely)
score = 10:  p = 4.5e-5 (extremely unlikely)
```

### Uncertainty Quantification

```python
# Scores from each posterior sample
individual_scores = -log_likelihoods  # Shape: (n_samples, n_obs)

# Summary statistics
score_mean = np.mean(individual_scores, axis=0)
score_std = np.std(individual_scores, axis=0)
score_lower = np.percentile(individual_scores, 5, axis=0)
score_upper = np.percentile(individual_scores, 95, axis=0)
```

---

## Evaluation Metrics

### Why These Metrics?

| Metric | Purpose | Why Important |
|--------|---------|---------------|
| PR-AUC | Overall ranking quality | Handles class imbalance better than ROC-AUC |
| Recall@K | Operational effectiveness | "How many attacks in top K alerts?" |
| Precision@K | Alert quality | "What fraction of top K are real attacks?" |

### PR-AUC Implementation

```python
def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Precision-Recall Area Under Curve.

    Preferred over ROC-AUC for imbalanced data because:
    1. Focuses on minority class (attacks)
    2. Not inflated by true negatives
    3. Directly measures precision/recall tradeoff
    """
    return average_precision_score(y_true, scores)
```

### Recall@K Implementation

```python
def compute_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Fraction of all attacks captured in top K scores.

    Recall@K = (attacks in top K) / (total attacks)

    Operationally: "If we investigate top K alerts daily,
                   what fraction of attacks do we catch?"
    """
    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0

    top_k_idx = np.argsort(scores)[-k:]  # Indices of top K scores
    tp_at_k = y_true[top_k_idx].sum()    # True positives in top K

    return tp_at_k / n_positives
```

### Precision@K Implementation

```python
def compute_precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Fraction of top K predictions that are true attacks.

    Precision@K = (attacks in top K) / K

    Operationally: "Of the alerts we send to analysts,
                   what fraction are real?"
    """
    top_k_idx = np.argsort(scores)[-k:]
    tp_at_k = y_true[top_k_idx].sum()

    return tp_at_k / k
```

---

## Visualization

### Score Distribution Plot

```python
def plot_score_distribution(scored_df, output_path):
    """
    Compare score distributions: Attack vs Benign

    Shows separation quality between classes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    benign_scores = scored_df[~scored_df["has_attack"]]["anomaly_score"]
    attack_scores = scored_df[scored_df["has_attack"]]["anomaly_score"]

    axes[0].hist(benign_scores, bins=50, alpha=0.7, label="Benign", density=True)
    axes[0].hist(attack_scores, bins=50, alpha=0.7, label="Attack", density=True)
    axes[0].legend()

    # Box plot
    sns.boxplot(data=scored_df, x="has_attack", y="anomaly_score", ax=axes[1])
```

### Top Anomalies Plot

```python
def plot_top_anomalies(scored_df, n=20, output_path):
    """
    Horizontal bar chart of top anomalies with error bars.

    Red bars = ground truth attacks
    Blue bars = ground truth benign
    Error bars = score uncertainty (90% CI)
    """
    top_df = scored_df.head(n)

    colors = ["crimson" if attack else "steelblue" for attack in top_df["has_attack"]]

    plt.barh(
        range(n),
        top_df["anomaly_score"],
        xerr=[
            top_df["anomaly_score"] - top_df["score_lower"],
            top_df["score_upper"] - top_df["anomaly_score"],
        ],
        color=colors,
    )
```

### Posterior Uncertainty Examples

```python
def plot_posterior_uncertainty(scored_df, n_examples=6):
    """
    Show predicted intervals vs actual observations.

    For each example entity-window:
    - Blue band: 90% posterior predictive interval
    - Blue mark: Predicted mean
    - Dot: Actual observed count
    """
    for example in examples:
        # Plot prediction interval
        ax.barh(0, predicted_upper - predicted_lower, left=predicted_lower)
        ax.plot(predicted_mean, 0, "b|")

        # Plot actual observation
        ax.plot(actual_count, 0, "o", color="red" if attack else "green")
```

---

## Next: [API Reference](05_api_reference.md)
