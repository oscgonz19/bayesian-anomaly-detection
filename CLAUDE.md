# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BSAD (Bayesian Security Anomaly Detection) is a hierarchical Negative Binomial anomaly detector for security event count data. It uses PyMC for MCMC inference and is designed for SOC workflows where attacks are rare (<5%) and entity-specific baselines matter.

**What the model actually does:** scores aggregated event counts per entity per time window using `-log p(y | posterior)`. It does NOT model IP, location, device, or endpoint features—those are contextual metadata only.

## Common Commands

```bash
# Environment setup
make env                    # Create conda environment
conda activate bsad
make install-dev            # Install with dev dependencies

# Run demo pipeline (generate data -> train -> score -> evaluate)
make demo                   # Full demo (~2000 samples)
make demo-fast              # Quick demo (~500 samples)

# Individual pipeline steps via CLI
bsad generate-data --n-entities 200 --n-days 30 --output data/events.parquet
bsad train --input data/events.parquet --output outputs/model.nc --samples 2000
bsad score --model outputs/model.nc --input outputs/modeling_table.parquet --output outputs/scores.parquet
bsad evaluate --scores outputs/scores.parquet --output outputs/metrics.json

# Testing
pytest tests/ -v                               # Run all tests
pytest tests/test_model.py -v                  # Run specific test file
pytest tests/test_triage.py -v -k "test_name" # Run single test
pytest tests/ -v -m "not slow"                 # Skip slow MCMC tests
PYTHONPATH=src pytest tests/ -v                # If import issues occur

# Linting and formatting
make lint                   # ruff + mypy
make format                 # black + ruff --fix

# Streamlit dashboard
make streamlit              # Runs app/streamlit_app.py

# Benchmark vs baselines
make benchmark              # Full benchmark (NB, GLMM, IF, LOF) at multiple attack rates
make benchmark-quick        # Quick single-rate benchmark
make robustness             # Robustness analysis (drift, cold-start, attack rate sweep)
make eda                    # Pedagogical EDA pipeline explainer

# Visualization
make viz-all                # All visualizations
make viz-report             # Full PDF report
```

## Architecture

### Two-Stage Pipeline

1. **Detection (`src/bsad/`)** — Hierarchical NB model that learns entity-specific baselines
2. **Triage (`src/triage/`)** — Transforms anomaly scores into actionable SOC workflows

### Module Structure

The key design decision: domain logic lives in focused single-purpose modules; `steps.py` and `pipeline.py` are thin orchestration layers only.

```
src/bsad/
├── data_generator.py   # GeneratorConfig + synthetic data functions (public API)
├── features/           # FeatureConfig + feature engineering (public API)
│   └── __init__.py
├── model.py            # ModelConfig + build/fit/diagnose (public API)
├── scoring/            # Posterior-based anomaly scoring
│   └── __init__.py
├── evaluation/         # Detection + operational metrics
│   └── __init__.py
├── steps.py            # Thin shims: adapts Settings → module configs, calls modules
├── pipeline.py         # Pipeline orchestrator + PipelineState
├── config.py           # Settings dataclass (paths + all params in one place)
├── io.py               # I/O helpers + RunMetadata dataclass
├── cli.py              # Typer CLI (thin wrapper over Pipeline)
├── baselines.py        # Competing models for fair benchmarking
├── calibration.py      # ECE, reliability diagrams, coverage
└── unsw_adapter.py     # UNSW-NB15 real-dataset adapter
src/triage/
├── risk_score.py       # Composite risk score (anomaly + uncertainty + novelty)
├── calibrate_thresholds.py  # Alert budget calibration (fixed_alerts/recall/fpr)
├── ranking_metrics.py  # Operational metrics: precision@k, recall@k, alerts/1k
└── entity_context.py   # Entity history enrichment for analyst decision support
```

### Key Module APIs

**`bsad.data_generator`**
- `GeneratorConfig` — generation params (n_users, n_days, attack_rate, etc.)
- `generate_synthetic_data(config)` → `(events_df, attacks_df)`
- `generate_baseline_events(config, rng)` → `pd.DataFrame`
- `inject_brute_force_attack(df, config, rng)` → `(df, records)`

**`bsad.features`**
- `FeatureConfig` — entity_column, window_size, include_temporal
- `create_time_windows(events_df, config)` → aggregated `pd.DataFrame`
- `add_temporal_features(windowed_df)` → `pd.DataFrame`
- `add_entity_features(windowed_df, entity_column)` → `pd.DataFrame`
- `encode_entity_ids(windowed_df, entity_column)` → `(df, mapping)`
- `build_modeling_table(events_df, config)` → `(modeling_df, metadata)`
- `get_model_arrays(modeling_df)` → `dict[str, np.ndarray]`

**`bsad.model`**
- `ModelConfig` — n_samples, n_chains=4, n_tune, priors
- `build_hierarchical_negbinom_model(y, entity_idx, n_entities, config)` → `pm.Model`
- `fit_model(model, config)` → `az.InferenceData`
- `get_model_diagnostics(trace)` → `dict` (r_hat_max, divergences, converged)

**`bsad.scoring`**
- `compute_scores(y, trace, entity_idx)` → `dict` (anomaly_score, score_std, bounds)
- `compute_intervals(trace, entity_idx)` → `dict` (predicted_mean, lower, upper)
- `create_scored_df(modeling_df, scores, intervals)` → scored `pd.DataFrame`

**`bsad.evaluation`**
- `compute_recall_at_k(y_true, scores, k)` → `float`
- `compute_precision_at_k(y_true, scores, k)` → `float`
- `compute_pr_auc(y_true, scores)` → `float`
- `compute_roc_auc(y_true, scores)` → `float`
- `compute_all_metrics(y_true, scores, k_values)` → `dict`
- `format_metrics_report(metrics)` → `str`

**`bsad.io`**
- `RunMetadata` — captures timestamp, seed, git_commit, config_snapshot
- `RunMetadata.from_settings(settings)` — builds from Settings object
- `save_run_metadata(metadata, path)` — saves as JSON

### Data Flow

```
Raw events → build_modeling_table() → modeling_df
                                           ↓
                               get_model_arrays() → {y, entity_idx, n_entities}
                                           ↓
                    build_hierarchical_negbinom_model() → pm.Model
                                           ↓
                               fit_model() → trace (ArviZ InferenceData)
                                           ↓
                          compute_scores(y, trace, entity_idx) → scores dict
                                           ↓
                    create_scored_df(modeling_df, scores, intervals) → scored_df
```

### Model Structure

Hierarchical Negative Binomial with partial pooling:
- Population: `mu ~ Exponential(rate)`, `alpha ~ HalfNormal`
- Entity: `theta[e] ~ Gamma(mu*alpha, alpha)` — entity-specific rates with shrinkage
- Observation: `y ~ NegativeBinomial(mu=theta[entity_idx], alpha=phi)`

Anomaly score: `-log p(y | posterior)` via log-sum-exp over all posterior samples.

## Key Design Patterns

- **Focused modules**: Each module owns one concern. `steps.py` adapts `Settings` to module-specific configs.
- **`steps.py` as shim**: The pipeline's external API (used by `pipeline.py` and `cli.py`) stays stable; internals moved to focused modules.
- **`PipelineState` dataclass**: Holds all intermediate artifacts (`events_df`, `trace`, `scored_df`, etc.)
- **`RunMetadata`**: Snapshot of config + git hash saved with every run for reproducibility.
- **Parquet + NetCDF4**: DataFrames as parquet, MCMC traces as `.nc` via ArviZ InferenceData.

## Data Leakage Caution

`add_entity_features()` computes entity baseline statistics (mean, std) from the full dataframe passed to it. For temporal train/test splits, call `build_modeling_table()` separately on train and test splits to avoid test-window data leaking into entity baselines.

## Important Constraints

- Designed for **count data + entity structure + rare events (<5%)**
- Not suitable for: multivariate continuous features, classification (>10% attack rate), real-time (<100ms)
- MCMC training takes hours; scoring is fast once trained
- `geo_anomaly` and `device_anomaly` are only detectable via count elevation; the NB model does not see location or device features
- `baselines.py` models are count-data-specific (fair comparison); generic IF/LOF are reference only
