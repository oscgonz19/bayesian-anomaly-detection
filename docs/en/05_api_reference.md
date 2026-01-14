# API Reference

## Table of Contents
1. [Data Generation Module](#data-generation-module)
2. [Features Module](#features-module)
3. [Model Module](#model-module)
4. [Scoring Module](#scoring-module)
5. [Evaluation Module](#evaluation-module)
6. [Visualization Module](#visualization-module)
7. [CLI Commands](#cli-commands)

---

## Data Generation Module

`bsad.data_generator`

### Classes

#### `GeneratorConfig`

Configuration dataclass for synthetic data generation.

```python
@dataclass
class GeneratorConfig:
    n_users: int = 200                    # Number of user entities
    n_ips: int = 100                      # Number of IP addresses
    n_endpoints: int = 50                 # Number of API endpoints
    n_days: int = 30                      # Days to simulate
    events_per_user_day_mean: float = 5.0 # Mean events per user per day
    events_per_user_day_std: float = 3.0  # Std deviation
    attack_rate: float = 0.02             # Fraction of entity-windows with attacks
    random_seed: int = 42                 # Reproducibility seed

    # Attack parameters
    brute_force_multiplier: tuple[int, int] = (50, 200)
    credential_stuffing_users: tuple[int, int] = (10, 30)
    credential_stuffing_events_per_user: tuple[int, int] = (3, 10)
    geo_anomaly_locations: int = 5
    device_anomaly_new_devices: tuple[int, int] = (3, 8)
```

### Functions

#### `generate_synthetic_data`

```python
def generate_synthetic_data(
    config: GeneratorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate complete synthetic security event dataset.

    Parameters
    ----------
    config : GeneratorConfig, optional
        Generation configuration. Uses defaults if None.

    Returns
    -------
    events_df : pd.DataFrame
        All events with columns:
        - timestamp: datetime
        - user_id: str
        - ip_address: str
        - endpoint: str
        - status_code: int
        - location: str
        - device_fingerprint: str
        - bytes_transferred: int
        - is_attack: bool
        - attack_type: str

    attacks_df : pd.DataFrame
        Attack metadata with columns:
        - attack_type: str
        - target_entity: str or list
        - source_ip: str
        - window_start: datetime
        - n_events: int

    Example
    -------
    >>> from bsad.data_generator import GeneratorConfig, generate_synthetic_data
    >>> config = GeneratorConfig(n_users=100, n_days=14, attack_rate=0.05)
    >>> events_df, attacks_df = generate_synthetic_data(config)
    >>> print(f"Generated {len(events_df)} events with {attacks_df['n_events'].sum()} attack events")
    """
```

#### `generate_baseline_events`

```python
def generate_baseline_events(
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate baseline (benign) security events.

    Creates event logs with realistic patterns including
    user-specific activity rates, diurnal patterns, and
    day-of-week effects.

    Parameters
    ----------
    config : GeneratorConfig
        Generation configuration
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    pd.DataFrame
        Baseline events with all columns, is_attack=False
    """
```

#### `inject_brute_force_attack`

```python
def inject_brute_force_attack(
    df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject brute force attack pattern.

    Characteristics:
    - Single IP targeting single user
    - 50-200 events in short window
    - 90% failed attempts (401)
    - Final success (200)

    Returns
    -------
    df : pd.DataFrame
        Events with attack injected
    records : list[dict]
        Attack metadata records
    """
```

#### `save_synthetic_data`

```python
def save_synthetic_data(
    events_df: pd.DataFrame,
    attacks_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save generated data to parquet files.

    Saves events to output_path and attacks to
    output_path.replace('.parquet', '_attacks.parquet')
    """
```

---

## Features Module

`bsad.features`

### Classes

#### `FeatureConfig`

```python
@dataclass
class FeatureConfig:
    entity_column: str = "user_id"        # Column to group by
    window_size: Literal["1H", "6H", "1D"] = "1D"  # Aggregation window
    include_temporal: bool = True          # Add hour, day_of_week, etc.
    include_categorical: bool = True       # Add categorical encodings
```

### Functions

#### `build_modeling_table`

```python
def build_modeling_table(
    events_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build complete modeling table from raw events.

    Performs:
    1. Time window aggregation
    2. Temporal feature extraction
    3. Entity-level statistics
    4. Entity ID encoding

    Parameters
    ----------
    events_df : pd.DataFrame
        Raw event logs
    config : FeatureConfig, optional
        Feature engineering configuration

    Returns
    -------
    modeling_df : pd.DataFrame
        Feature table with columns:
        - entity_idx: int (encoded entity ID)
        - window: datetime
        - event_count: int (target variable)
        - unique_ips, unique_endpoints, etc.
        - hour, day_of_week, is_weekend, is_business_hours
        - entity_mean_count, entity_std_count, count_zscore
        - has_attack: bool
        - attack_type: str

    metadata : dict
        - entity_column: str
        - entity_mapping: dict[str, int]
        - n_entities: int
        - n_windows: int
        - attack_rate: float
        - feature_columns: list[str]

    Example
    -------
    >>> from bsad.features import FeatureConfig, build_modeling_table
    >>> config = FeatureConfig(window_size="1D")
    >>> modeling_df, metadata = build_modeling_table(events_df, config)
    >>> print(f"Created {metadata['n_windows']} windows for {metadata['n_entities']} entities")
    """
```

#### `get_model_arrays`

```python
def get_model_arrays(
    modeling_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Extract numpy arrays for PyMC model.

    Returns
    -------
    dict with keys:
        - y: Event counts (int64), shape (n_obs,)
        - entity_idx: Entity indices (int64), shape (n_obs,)
        - is_attack: Ground truth labels (bool), shape (n_obs,)
        - window_idx: Window indices (int64), shape (n_obs,)
    """
```

#### `create_time_windows`

```python
def create_time_windows(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Aggregate events into time windows per entity.

    Computes per-window features:
    - event_count: Number of events
    - unique_ips: Unique IP addresses
    - unique_endpoints: Unique endpoints accessed
    - unique_devices: Unique device fingerprints
    - unique_locations: Unique locations
    - failed_count: Failed requests (4xx/5xx)
    - bytes_total: Total bytes transferred
    - has_attack: Contains attack events
    """
```

---

## Model Module

`bsad.model`

### Classes

#### `ModelConfig`

```python
@dataclass
class ModelConfig:
    # Sampling parameters
    n_samples: int = 2000         # Posterior samples per chain
    n_tune: int = 1000            # Warmup/tuning samples
    n_chains: int = 4             # Independent MCMC chains
    target_accept: float = 0.9    # NUTS acceptance rate target
    random_seed: int = 42         # Reproducibility seed

    # Prior parameters
    mu_prior_rate: float = 0.1           # Exponential rate for μ
    alpha_prior_sd: float = 2.0          # HalfNormal σ for α
    overdispersion_prior_sd: float = 2.0 # HalfNormal σ for φ
```

### Functions

#### `build_hierarchical_negbinom_model`

```python
def build_hierarchical_negbinom_model(
    y: np.ndarray,
    entity_idx: np.ndarray,
    n_entities: int,
    config: ModelConfig | None = None,
) -> pm.Model:
    """
    Build hierarchical Negative Binomial model.

    Model structure:
        μ ~ Exponential(0.1)           # Population mean
        α ~ HalfNormal(2)              # Concentration
        θ_i ~ Gamma(μα, α)             # Entity rates
        φ ~ HalfNormal(2)              # Overdispersion
        y ~ NegativeBinomial(θ, φ)     # Observations

    Parameters
    ----------
    y : np.ndarray, shape (n_obs,)
        Event counts per observation
    entity_idx : np.ndarray, shape (n_obs,)
        Entity index for each observation (0 to n_entities-1)
    n_entities : int
        Total number of unique entities
    config : ModelConfig, optional
        Model configuration

    Returns
    -------
    pm.Model
        PyMC model object (not yet fitted)

    Example
    -------
    >>> model = build_hierarchical_negbinom_model(
    ...     y=arrays["y"],
    ...     entity_idx=arrays["entity_idx"],
    ...     n_entities=metadata["n_entities"],
    ... )
    """
```

#### `fit_model`

```python
def fit_model(
    model: pm.Model,
    config: ModelConfig | None = None,
) -> az.InferenceData:
    """
    Fit model using NUTS sampler.

    Performs:
    1. MCMC sampling with automatic tuning
    2. Posterior predictive sampling
    3. Convergence diagnostics

    Parameters
    ----------
    model : pm.Model
        PyMC model from build_hierarchical_negbinom_model
    config : ModelConfig, optional
        Sampling configuration

    Returns
    -------
    az.InferenceData
        ArviZ InferenceData with groups:
        - posterior: Parameter samples
        - posterior_predictive: Simulated observations
        - sample_stats: MCMC diagnostics
        - observed_data: Input data

    Example
    -------
    >>> trace = fit_model(model, config)
    >>> print(f"Sampled {trace.posterior.dims['draw']} draws")
    """
```

#### `get_model_diagnostics`

```python
def get_model_diagnostics(trace: az.InferenceData) -> dict:
    """
    Compute model diagnostics.

    Returns
    -------
    dict with keys:
        - r_hat_max: Maximum R-hat across parameters (target: < 1.05)
        - ess_bulk_min: Minimum bulk ESS (target: > 400)
        - ess_tail_min: Minimum tail ESS (target: > 400)
        - divergences: Number of divergent transitions (target: 0)
        - converged: bool, True if r_hat_max < 1.05
    """
```

#### `save_trace` / `load_trace`

```python
def save_trace(trace: az.InferenceData, path: str | Path) -> None:
    """Save inference data to NetCDF file."""

def load_trace(path: str | Path) -> az.InferenceData:
    """Load inference data from NetCDF file."""
```

---

## Scoring Module

`bsad.scoring`

### Functions

#### `compute_anomaly_scores`

```python
def compute_anomaly_scores(
    y_observed: np.ndarray,
    trace: az.InferenceData,
    entity_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute anomaly scores from posterior predictive.

    score = -log p(y_observed | posterior predictive)

    Higher scores indicate more anomalous observations.

    Parameters
    ----------
    y_observed : np.ndarray, shape (n_obs,)
        Observed event counts
    trace : az.InferenceData
        Fitted model trace with posterior samples
    entity_idx : np.ndarray, shape (n_obs,)
        Entity index for each observation

    Returns
    -------
    dict with keys:
        - anomaly_score: np.ndarray, shape (n_obs,)
            Point estimate scores (posterior mean)
        - score_std: np.ndarray, shape (n_obs,)
            Score standard deviation across posterior
        - score_lower: np.ndarray, shape (n_obs,)
            5th percentile of scores
        - score_upper: np.ndarray, shape (n_obs,)
            95th percentile of scores

    Example
    -------
    >>> scores = compute_anomaly_scores(y, trace, entity_idx)
    >>> print(f"Max anomaly score: {scores['anomaly_score'].max():.2f}")
    """
```

#### `compute_predictive_intervals`

```python
def compute_predictive_intervals(
    trace: az.InferenceData,
    entity_idx: np.ndarray,
    credible_mass: float = 0.9,
) -> dict[str, np.ndarray]:
    """
    Compute predictive intervals for each observation.

    Returns
    -------
    dict with keys:
        - predicted_mean: Expected value
        - predicted_lower: Lower bound of credible interval
        - predicted_upper: Upper bound of credible interval
    """
```

#### `create_scored_dataframe`

```python
def create_scored_dataframe(
    modeling_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    intervals: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Create scored DataFrame with anomaly scores and metadata.

    Adds columns:
    - anomaly_score, score_std, score_lower, score_upper
    - predicted_mean, predicted_lower, predicted_upper
    - anomaly_rank: Rank by score (1 = most anomalous)
    - exceeds_interval: bool, count > predicted_upper

    Returns DataFrame sorted by anomaly_score descending.
    """
```

#### `get_top_anomalies`

```python
def get_top_anomalies(
    scored_df: pd.DataFrame,
    n: int = 100,
) -> pd.DataFrame:
    """
    Get top N anomalies with relevant columns.

    Returns subset of scored_df with display-friendly columns.
    """
```

---

## Evaluation Module

`bsad.evaluation`

### Functions

#### `compute_all_metrics`

```python
def compute_all_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0/1)
    scores : np.ndarray
        Anomaly scores (higher = more anomalous)
    k_values : list[int], optional
        K values for Recall@K (default: [10, 25, 50, 100])

    Returns
    -------
    dict with keys:
        - pr_auc: Precision-Recall AUC
        - roc_auc: ROC AUC
        - n_observations: Total observations
        - n_positives: Number of attacks
        - attack_rate: Fraction of attacks
        - recall_at_{k}: Recall at top K
        - precision_at_{k}: Precision at top K
        - pr_curve: dict with precision, recall, thresholds arrays
    """
```

#### `compute_pr_auc`

```python
def compute_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Precision-Recall Area Under Curve.

    Preferred over ROC-AUC for imbalanced datasets.
    """
```

#### `compute_recall_at_k`

```python
def compute_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Compute recall at top K predictions.

    Recall@K = (true positives in top K) / (total positives)
    """
```

#### `format_metrics_report`

```python
def format_metrics_report(metrics: dict) -> str:
    """
    Format metrics as human-readable report string.

    Suitable for console output.
    """
```

---

## Visualization Module

`bsad.visualization`

### Functions

#### `create_all_plots`

```python
def create_all_plots(
    scored_df: pd.DataFrame,
    metrics: dict,
    trace: az.InferenceData | None = None,
    output_dir: str | Path = "outputs/plots",
) -> dict[str, Path]:
    """
    Generate all standard plots and save to directory.

    Creates:
    - score_distribution.png: Histogram of scores by class
    - precision_recall_curve.png: PR curve
    - top_anomalies.png: Bar chart of top anomalies
    - posterior_uncertainty.png: Predictive interval examples
    - model_diagnostics.png: MCMC trace plots (if trace provided)

    Returns dict mapping plot names to file paths.
    """
```

---

## CLI Commands

### `bsad generate-data`

```bash
bsad generate-data [OPTIONS]

Options:
  -n, --n-entities INTEGER   Number of user entities [default: 200]
  -d, --n-days INTEGER       Number of days to simulate [default: 30]
  -a, --attack-rate FLOAT    Fraction of entity-windows with attacks [default: 0.02]
  -s, --seed INTEGER         Random seed [default: 42]
  -o, --output PATH          Output path for events [default: data/events.parquet]
```

### `bsad train`

```bash
bsad train [OPTIONS]

Options:
  -i, --input PATH           Input events file [default: data/events.parquet]
  -o, --output PATH          Output model path [default: outputs/model.nc]
  -s, --samples INTEGER      Posterior samples [default: 2000]
  -t, --tune INTEGER         Tuning samples [default: 1000]
  -c, --chains INTEGER       MCMC chains [default: 4]
  --seed INTEGER             Random seed [default: 42]
```

### `bsad score`

```bash
bsad score [OPTIONS]

Options:
  -m, --model PATH           Trained model path [default: outputs/model.nc]
  -i, --input PATH           Modeling table path [default: outputs/modeling_table.parquet]
  -o, --output PATH          Scores output path [default: outputs/scores.parquet]
```

### `bsad evaluate`

```bash
bsad evaluate [OPTIONS]

Options:
  -s, --scores PATH          Scored data file [default: outputs/scores.parquet]
  -o, --output PATH          Metrics output path [default: outputs/metrics.json]
  -p, --plots PATH           Plot output directory [default: outputs/plots]
```

### `bsad demo`

```bash
bsad demo [OPTIONS]

Run complete pipeline: generate → train → score → evaluate

Options:
  -o, --output-dir PATH      Output directory [default: outputs]
  -n, --n-entities INTEGER   Number of entities [default: 200]
  -d, --n-days INTEGER       Number of days [default: 30]
  -s, --samples INTEGER      Posterior samples [default: 1000]
  --seed INTEGER             Random seed [default: 42]
```

---

## Next: [Tutorial](06_tutorial.md)
