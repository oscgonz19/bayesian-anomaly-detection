# Tutorial: Step-by-Step Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Step-by-Step Pipeline](#step-by-step-pipeline)
4. [Interpreting Results](#interpreting-results)
5. [Customization](#customization)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Using Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/bayesian-security-anomaly-detection.git
cd bayesian-security-anomaly-detection

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate bsad

# Verify installation
bsad --help
```

### Using pip

```bash
# Clone repository
git clone https://github.com/yourusername/bayesian-security-anomaly-detection.git
cd bayesian-security-anomaly-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install package
pip install -e ".[dev]"

# Verify installation
bsad --help
```

### Verify Installation

```bash
# Check CLI is available
bsad --help

# Check Python imports
python -c "from bsad import __version__; print(f'BSAD version: {__version__}')"

# Check PyMC installation
python -c "import pymc as pm; print(f'PyMC version: {pm.__version__}')"
```

---

## Quick Start

### One-Command Demo

```bash
# Run complete pipeline with defaults
make demo

# Or with custom parameters
bsad demo --n-entities 100 --n-days 14 --samples 500
```

This will:
1. Generate synthetic security events
2. Train the Bayesian model
3. Score all observations
4. Evaluate performance
5. Generate plots

Output in `outputs/` directory.

### Expected Output

```
================================================================
BAYESIAN SECURITY ANOMALY DETECTION - DEMO
================================================================

Step 1/4: Generating synthetic security events
  Generated 45,231 events
  Attack events: 1,847

Step 2/4: Training hierarchical Bayesian model
  Entities: 200
  Windows: 5,892
  Sampling 1000 posterior draws (this may take a few minutes)...
  R-hat: 1.002, Divergences: 0

Step 3/4: Scoring observations
  Scored 5,892 entity-windows

Step 4/4: Evaluating performance
  PR-AUC: 0.847
  Recall@50: 0.412
  Recall@100: 0.623

================================================================
DEMO COMPLETE
================================================================

Generated Artifacts:
  Events:   outputs/data/events.parquet
  Model:    outputs/model.nc
  Scores:   outputs/scores.parquet
  Metrics:  outputs/metrics.json
  Plots:    outputs/plots/
```

---

## Step-by-Step Pipeline

### Step 1: Generate Synthetic Data

```bash
bsad generate-data \
    --n-entities 200 \
    --n-days 30 \
    --attack-rate 0.02 \
    --seed 42 \
    --output data/events.parquet
```

**What this does:**
- Creates 200 synthetic users with heterogeneous activity patterns
- Simulates 30 days of security events
- Injects ~2% attack windows (brute force, credential stuffing, etc.)
- Saves events to Parquet file

**Verify:**
```python
import pandas as pd

events = pd.read_parquet("data/events.parquet")
print(f"Total events: {len(events):,}")
print(f"Attack events: {events['is_attack'].sum():,}")
print(f"Attack types: {events[events['is_attack']]['attack_type'].value_counts().to_dict()}")
```

### Step 2: Train Model

```bash
bsad train \
    --input data/events.parquet \
    --output outputs/model.nc \
    --samples 2000 \
    --tune 1000 \
    --chains 4
```

**What this does:**
- Loads events and builds modeling table (feature engineering)
- Constructs hierarchical Negative Binomial model
- Runs MCMC sampling (NUTS) with 4 chains × 2000 samples
- Saves posterior samples to NetCDF file

**Expected duration:** 5-15 minutes depending on data size and hardware.

**Verify:**
```python
import arviz as az

trace = az.from_netcdf("outputs/model.nc")
print(az.summary(trace, var_names=["mu", "alpha", "phi"]))
```

### Step 3: Score Observations

```bash
bsad score \
    --model outputs/model.nc \
    --input outputs/modeling_table.parquet \
    --output outputs/scores.parquet
```

**What this does:**
- Loads trained model (posterior samples)
- Computes anomaly scores for each entity-window
- Calculates score uncertainty (std, 90% CI)
- Ranks observations by anomaly score

**Verify:**
```python
import pandas as pd

scores = pd.read_parquet("outputs/scores.parquet")
print(f"Top 5 anomalies:")
print(scores[["user_id", "window", "event_count", "anomaly_score", "has_attack"]].head())
```

### Step 4: Evaluate Performance

```bash
bsad evaluate \
    --scores outputs/scores.parquet \
    --output outputs/metrics.json \
    --plots outputs/plots
```

**What this does:**
- Computes PR-AUC, ROC-AUC, Recall@K metrics
- Generates diagnostic plots
- Saves metrics to JSON file

**Verify:**
```python
import json

with open("outputs/metrics.json") as f:
    metrics = json.load(f)

print(f"PR-AUC: {metrics['pr_auc']:.3f}")
print(f"Recall@100: {metrics['recall_at_100']:.3f}")
```

---

## Interpreting Results

### Understanding Anomaly Scores

```python
scores_df = pd.read_parquet("outputs/scores.parquet")

# Score interpretation
# Higher score = more anomalous = less probable under model

# Typical benign scores: 2-5
# Typical attack scores: 6-15+

print(scores_df.groupby("has_attack")["anomaly_score"].describe())
```

### Score Components

Each observation has:

| Field | Meaning |
|-------|---------|
| `anomaly_score` | Point estimate (-log probability) |
| `score_std` | Uncertainty in score |
| `score_lower` | 5th percentile |
| `score_upper` | 95th percentile |
| `predicted_mean` | Expected event count |
| `predicted_lower` | Lower bound (90% CI) |
| `predicted_upper` | Upper bound (90% CI) |
| `exceeds_interval` | Observed > predicted_upper |

### Evaluating Detection Quality

```python
# Good detection separates attack/benign distributions
import matplotlib.pyplot as plt

benign = scores_df[~scores_df["has_attack"]]["anomaly_score"]
attack = scores_df[scores_df["has_attack"]]["anomaly_score"]

plt.hist(benign, bins=50, alpha=0.7, label="Benign", density=True)
plt.hist(attack, bins=50, alpha=0.7, label="Attack", density=True)
plt.legend()
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.show()
```

### Operational Thresholds

```python
# Find threshold for desired precision
from sklearn.metrics import precision_recall_curve

y_true = scores_df["has_attack"].astype(int)
y_scores = scores_df["anomaly_score"]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Example: Find threshold for 50% precision
target_precision = 0.5
idx = np.argmin(np.abs(precision[:-1] - target_precision))
threshold = thresholds[idx]
print(f"Threshold for {target_precision:.0%} precision: {threshold:.2f}")
print(f"Corresponding recall: {recall[idx]:.2%}")
```

---

## Customization

### Custom Attack Rate

```python
from bsad.data_generator import GeneratorConfig, generate_synthetic_data

# Higher attack rate for testing
config = GeneratorConfig(
    n_users=100,
    n_days=14,
    attack_rate=0.10,  # 10% attack rate
)
events_df, attacks_df = generate_synthetic_data(config)
```

### Different Window Sizes

```python
from bsad.features import FeatureConfig, build_modeling_table

# Hourly windows (more granular)
config = FeatureConfig(window_size="1H")
modeling_df, metadata = build_modeling_table(events_df, config)

# 6-hour windows
config = FeatureConfig(window_size="6H")
modeling_df, metadata = build_modeling_table(events_df, config)
```

### Custom Priors

```python
from bsad.model import ModelConfig, build_hierarchical_negbinom_model

# Stronger pooling prior (more entity similarity)
config = ModelConfig(
    alpha_prior_sd=1.0,  # Tighter concentration prior
)

# Faster sampling (fewer samples)
config = ModelConfig(
    n_samples=1000,
    n_tune=500,
    n_chains=2,
)
```

### IP-Level Analysis

```python
# Group by IP instead of user
config = FeatureConfig(entity_column="ip_address")
modeling_df, metadata = build_modeling_table(events_df, config)
```

---

## Troubleshooting

### Slow Sampling

**Symptom:** MCMC takes >30 minutes

**Solutions:**
1. Reduce sample size:
   ```bash
   bsad train --samples 500 --tune 250 --chains 2
   ```

2. Reduce data size:
   ```bash
   bsad generate-data --n-entities 50 --n-days 7
   ```

3. Use faster hardware (GPU not supported for NUTS)

### Convergence Warnings

**Symptom:** R-hat > 1.05 or many divergences

**Solutions:**
1. Increase target_accept:
   ```python
   config = ModelConfig(target_accept=0.95)
   ```

2. Increase tuning:
   ```python
   config = ModelConfig(n_tune=2000)
   ```

3. Check for extreme data values (outliers)

### Memory Errors

**Symptom:** Out of memory during sampling

**Solutions:**
1. Reduce chains:
   ```bash
   bsad train --chains 2
   ```

2. Reduce data size
3. Use machine with more RAM (16GB+ recommended)

### Poor Detection Performance

**Symptom:** Low PR-AUC (<0.5)

**Possible causes:**
1. Insufficient data: Increase n_days or n_entities
2. Low attack rate: Hard to distinguish with few positives
3. Attack patterns similar to baseline: Check attack injection

**Diagnostic:**
```python
# Check if attacks have distinct signatures
print(scores_df.groupby("attack_type")["event_count"].describe())
print(scores_df.groupby("attack_type")["anomaly_score"].describe())
```

---

## Next Steps

- Read [Theoretical Foundations](02_theoretical_foundations.md) for deeper understanding
- Explore [Model Architecture](03_model_architecture.md) for customization options
- Check [API Reference](05_api_reference.md) for programmatic usage

---

## Example Notebook

For interactive exploration, see `notebooks/exploration.ipynb`:

```python
# Load and explore data
import pandas as pd
from bsad.data_generator import GeneratorConfig, generate_synthetic_data
from bsad.features import build_modeling_table
from bsad.visualization import plot_score_distribution

# Generate data
config = GeneratorConfig(n_users=50, n_days=7, random_seed=123)
events_df, attacks_df = generate_synthetic_data(config)

# Build features
modeling_df, metadata = build_modeling_table(events_df)

# Explore
print(f"Events: {len(events_df):,}")
print(f"Modeling rows: {len(modeling_df):,}")
print(f"Attack rate: {metadata['attack_rate']:.2%}")
```
