# Bayesian Security Anomaly Detection (BSAD)

A reproducible Bayesian/MCMC anomaly detection pipeline for security event logs. Uses hierarchical Negative Binomial models with partial pooling to detect rare attack patterns while quantifying uncertainty.

## Problem Statement

Security teams face the challenge of identifying malicious activity in high-volume event logs where attacks are rare (typically <1% of events). Traditional threshold-based approaches suffer from:

1. **High false positive rates** - fixed thresholds don't adapt to entity-specific baselines
2. **No uncertainty quantification** - point estimates don't indicate confidence
3. **Poor generalization** - models overfit to entities with limited history

This project demonstrates a Bayesian approach that addresses these issues through:

- **Hierarchical modeling**: Partial pooling shares information across entities (users/IPs) while respecting individual variation
- **Posterior predictive scoring**: Anomaly scores derived from `-log p(y | posterior)` naturally incorporate uncertainty
- **Interpretable outputs**: Credible intervals provide actionable confidence bounds

## Method

### Model Architecture

We use a Hierarchical Negative Binomial model for event counts:

```
# Global priors
mu ~ Exponential(0.1)
alpha ~ HalfNormal(2)

# Entity-level parameters (partial pooling)
theta_entity ~ Gamma(mu, alpha)

# Observations
y_counts ~ NegativeBinomial(theta_entity, overdispersion)
```

The Negative Binomial handles overdispersion common in security logs (variance > mean), and hierarchical structure allows entities with sparse data to borrow strength from the population.

### Anomaly Scoring

For each observation, we compute:

```
anomaly_score = -log p(y_observed | posterior predictive)
```

High scores indicate observations unlikely under the learned model. Unlike point-estimate methods, this score incorporates full posterior uncertainty—an observation might be flagged even if close to the mean if the model is confident, or not flagged despite being far from the mean if uncertainty is high.

### Attack Patterns Detected

The synthetic data generator creates realistic attack patterns:

| Pattern | Description | Signal |
|---------|-------------|--------|
| Brute Force | High-frequency login attempts from single IP | Event burst in short window |
| Credential Stuffing | Moderate attempts across many users | Elevated count + multi-user targeting |
| Geo Anomaly | Logins from unusual locations | Location entropy spike |
| Device Anomaly | New device fingerprints | Device diversity increase |

## Installation

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
```

### Requirements

- Python 3.10+
- PyMC 5.10+ (JAX backend recommended for faster sampling)
- See `pyproject.toml` for full dependencies

## Quickstart

### Run Full Demo

```bash
# Single command to generate data, train, score, and evaluate
make demo

# Or using CLI directly
bsad demo --output-dir outputs/
```

### Step-by-Step Usage

```bash
# 1. Generate synthetic security logs with attack patterns
bsad generate-data --n-entities 500 --n-days 30 --attack-rate 0.02 --output data/events.parquet

# 2. Train hierarchical Bayesian model
bsad train --input data/events.parquet --output outputs/model.nc --samples 2000

# 3. Score observations for anomalies
bsad score --model outputs/model.nc --input data/events.parquet --output outputs/scores.parquet

# 4. Evaluate against ground truth
bsad evaluate --scores outputs/scores.parquet --output outputs/metrics.json
```

## Outputs

### Generated Artifacts

| File | Description |
|------|-------------|
| `data/events.parquet` | Synthetic event logs with ground truth labels |
| `outputs/model.nc` | ArviZ InferenceData with posterior samples |
| `outputs/scores.parquet` | Anomaly scores with uncertainty intervals |
| `outputs/metrics.json` | PR-AUC, Recall@K evaluation metrics |
| `outputs/plots/` | Visualization artifacts |

### Plots

1. **Anomaly Score Distribution**: Histogram comparing attack vs. benign score distributions
2. **Top Anomalies Table**: Ranked list with scores, credible intervals, and ground truth
3. **Posterior Uncertainty**: Example entities showing posterior predictive intervals

### Example Output

```
Evaluation Results
==================
PR-AUC:     0.847
Recall@100: 0.623
Recall@50:  0.412

Top 10 Anomalies:
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Entity   ┃ Window    ┃ Score      ┃ Count   ┃ Ground Truth ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ user_042 │ 2024-01-15│ 8.92       │ 847     │ ATTACK       │
│ ip_198   │ 2024-01-08│ 7.45       │ 523     │ ATTACK       │
│ user_117 │ 2024-01-22│ 6.81       │ 412     │ ATTACK       │
│ ...      │           │            │         │              │
└──────────┴───────────┴────────────┴─────────┴──────────────┘
```

## Evaluation Metrics

### Why PR-AUC?

Precision-Recall AUC is preferred over ROC-AUC for rare events because:

- **Class imbalance**: With ~2% attack rate, ROC-AUC can be misleadingly high (0.95+) even with poor precision
- **Actionable threshold selection**: PR curves directly show the precision/recall tradeoff at different score thresholds
- **Focus on positives**: Security teams care about "of the alerts we generate, how many are real?" (precision) and "of real attacks, how many do we catch?" (recall)

### Why Recall@K?

In operational settings, analysts can only investigate a fixed number of alerts per day. Recall@K answers: "If we investigate the top K anomalies, what fraction of true attacks do we find?"

- `Recall@50`: Fraction of attacks in top 50 scores (high-priority queue)
- `Recall@100`: Fraction of attacks in top 100 scores (daily review capacity)

## Visualization

The `dataviz/` folder contains comprehensive visualization scripts for every stage of the pipeline:

```bash
# Run individual visualization scripts
make viz-explore   # Data exploration: event timelines, distributions, attack patterns
make viz-features  # Feature engineering: aggregated features, overdispersion, correlations
make viz-model     # Model diagnostics: trace plots, posteriors, convergence (R-hat, ESS)
make viz-results   # Anomaly results: score distributions, top anomalies, attack breakdowns
make viz-eval      # Evaluation metrics: PR curve, ROC curve, Recall@K, lift curves

# Run all visualizations at once
make viz-all

# Generate a comprehensive PDF report
make viz-report
```

### Visualization Scripts

| Script | Description |
|--------|-------------|
| `01_data_exploration.py` | Event timeline, hourly heatmap, user activity, attack pattern analysis |
| `02_feature_analysis.py` | Event count distributions, overdispersion analysis, temporal features |
| `03_model_diagnostics.py` | MCMC trace plots, posterior distributions, convergence diagnostics, PPC |
| `04_anomaly_results.py` | Score distributions, top anomalies, attack type breakdown, prediction intervals |
| `05_evaluation_plots.py` | PR/ROC curves, Recall@K, lift curves, baseline comparisons |
| `06_full_report.py` | Orchestrates all scripts, generates complete PDF report |

## Project Structure

```
bayesian-security-anomaly-detection/
├── src/bsad/                           # Core package (simplified!)
│   ├── __init__.py                     # Exports: Settings, Pipeline
│   ├── config.py                       # Settings dataclass (all config in one place)
│   ├── io.py                           # File I/O helpers (parquet, NetCDF, JSON)
│   ├── steps.py                        # Pipeline steps as pure functions
│   ├── pipeline.py                     # Pipeline class (orchestration)
│   └── cli.py                          # Typer CLI (bsad demo, train, score)
├── notebooks/
│   └── 01_end_to_end_walkthrough.ipynb # Complete tutorial (80+ cells)
├── dataviz/
│   ├── 01_data_exploration.py          # Raw event data visualizations
│   ├── 02_feature_analysis.py          # Feature engineering visualizations
│   ├── 03_model_diagnostics.py         # MCMC diagnostics and convergence
│   ├── 04_anomaly_results.py           # Anomaly detection results
│   ├── 05_evaluation_plots.py          # PR-AUC, ROC, Recall@K plots
│   └── 06_full_report.py               # Complete report generator
├── tests/                              # Unit tests
├── docs/
│   ├── en/                             # English documentation
│   └── es/                             # Spanish documentation
├── app/
│   └── streamlit_app.py                # Optional dashboard
├── data/                               # Generated datasets
├── outputs/                            # Model artifacts and results
├── Makefile
├── pyproject.toml
└── README.md
```

### Code Architecture

The codebase follows a **simple, script-like pipeline** design:

| File | Purpose |
|------|---------|
| `config.py` | All settings in one `Settings` dataclass |
| `io.py` | File I/O helpers (no business logic) |
| `steps.py` | Pure functions for each pipeline step |
| `pipeline.py` | `Pipeline` class that orchestrates steps |
| `cli.py` | Thin CLI layer that calls Pipeline |

**Key principle**: Steps are pure functions that don't call each other. The `Pipeline` class controls the flow.

## Configuration

Environment variables for model tuning:

```bash
export BSAD_RANDOM_SEED=42
export BSAD_SAMPLER_CORES=4
export BSAD_TARGET_ACCEPT=0.9
```

## Limitations and Next Steps

### Current Limitations

1. **Synthetic data only**: Real security logs have different distributional properties, missing data patterns, and label noise. The synthetic generator captures key attack signatures but not full operational complexity.

2. **Single feature type**: The model uses event counts per entity/window. Production systems would benefit from multi-modal features (bytes transferred, endpoint diversity, session duration).

3. **Static windows**: Fixed time windows (e.g., hourly) may miss attacks spanning window boundaries or sub-window bursts. Adaptive windowing or continuous-time models could improve detection.

4. **No temporal dynamics**: The current model treats windows independently. Sequential models (HMMs, state-space models) could capture entity behavioral drift and attack progression.

5. **Scalability**: MCMC sampling scales poorly beyond ~10K entities. Variational inference (ADVI) or amortized inference could enable larger deployments.

6. **Label quality**: Ground truth in real settings is noisy and incomplete. Semi-supervised or positive-unlabeled learning approaches may be more appropriate.

### Potential Extensions

- **Multi-feature model**: Extend to multivariate observations (count + bytes + unique endpoints)
- **Temporal modeling**: Add autoregressive components or hidden Markov structure
- **Online learning**: Incremental posterior updates as new data arrives
- **Calibration**: Post-hoc calibration to convert scores to attack probabilities
- **Explainability**: Feature attribution for flagged anomalies

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis, 3rd Edition*
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3
- Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*

## License

MIT License - see LICENSE file for details.
