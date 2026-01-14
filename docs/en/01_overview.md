# Bayesian Security Anomaly Detection - Overview

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Why Bayesian Methods?](#why-bayesian-methods)
4. [System Architecture](#system-architecture)
5. [Key Components](#key-components)

---

## Introduction

**Bayesian Security Anomaly Detection (BSAD)** is a probabilistic machine learning system designed to identify anomalous patterns in security event logs. Unlike traditional rule-based or threshold-based approaches, BSAD leverages Bayesian inference to:

1. **Quantify uncertainty** in anomaly predictions
2. **Share information** across entities through hierarchical modeling
3. **Adapt** to entity-specific behavioral baselines
4. **Provide interpretable** probability-based scores

This documentation provides a comprehensive guide to understanding the theoretical foundations, implementation details, and practical usage of the system.

---

## Problem Statement

### The Challenge of Security Log Analysis

Modern security operations centers (SOCs) face an overwhelming volume of event data:

- **Scale**: Large organizations generate millions of security events daily
- **Imbalance**: Actual attacks represent a tiny fraction (<1%) of all events
- **Heterogeneity**: Different users/systems have vastly different "normal" behavior
- **Evolution**: Attack patterns and normal behavior change over time

### Traditional Approaches and Their Limitations

| Approach | Method | Limitations |
|----------|--------|-------------|
| **Rule-based** | Static thresholds (e.g., >100 logins/hour = alert) | Doesn't adapt to entity-specific baselines; high false positives |
| **Statistical** | Z-scores, standard deviation bands | Assumes Gaussian distributions; doesn't handle entity heterogeneity |
| **ML Classification** | Supervised models (Random Forest, etc.) | Requires labeled training data; point estimates without uncertainty |
| **Isolation Forest** | Unsupervised anomaly detection | No probabilistic interpretation; hard to tune |

### Our Approach: Hierarchical Bayesian Modeling

BSAD addresses these limitations through:

1. **Hierarchical structure**: Learns population-level patterns while respecting individual variation
2. **Negative Binomial likelihood**: Handles overdispersed count data common in security logs
3. **Full posterior inference**: Provides uncertainty estimates, not just point predictions
4. **Principled anomaly scoring**: Based on posterior predictive probability

---

## Why Bayesian Methods?

### Advantages for Anomaly Detection

#### 1. Uncertainty Quantification

Traditional ML gives point estimates. Bayesian inference gives probability distributions:

```
Traditional: "This entity has anomaly score 7.5"
Bayesian:    "This entity has anomaly score 7.5 (90% CI: 5.2-9.8)"
```

This distinction matters operationally:
- High score with tight CI → High confidence, prioritize investigation
- High score with wide CI → Uncertain, gather more data before acting

#### 2. Handling Sparse Data

Security data is inherently sparse—many entities have limited history. Bayesian hierarchical models handle this through **partial pooling**:

![Partial Pooling Explained](../images/partial_pooling_explained.png)
*Partial pooling: sparse entities shrink to population mean, dense entities keep their own rate*

- Entities with **abundant data**: Estimates driven by their own observations
- Entities with **sparse data**: Estimates "shrink" toward population average

This prevents both:
- Overfitting to noise in small samples
- Ignoring entity-specific patterns in large samples

#### 3. Prior Knowledge Incorporation

Bayesian methods allow incorporating domain knowledge:

```python
# We know event rates are positive and typically moderate
mu ~ Exponential(0.1)  # Prior: mean rate around 10 events/window
```

#### 4. Coherent Probability Framework

All inferences follow probability theory. Anomaly scores have a principled interpretation:

```
anomaly_score = -log p(y_observed | model)
```

Higher score = observation less likely under learned model = more anomalous.

### The Bayesian Paradigm

At its core, Bayesian inference follows Bayes' theorem:

```
p(θ|y) = p(y|θ) × p(θ) / p(y)

posterior ∝ likelihood × prior
```

Where:
- **p(θ)** = Prior: What we believe about parameters before seeing data
- **p(y|θ)** = Likelihood: How probable is our data given parameters
- **p(θ|y)** = Posterior: Updated beliefs after seeing data

---

## System Architecture

### High-Level Pipeline

![Pipeline Architecture](../images/pipeline_architecture.png)
*End-to-end pipeline: from raw data to ranked anomalies with uncertainty*

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Event Logs │────▶│    Features     │────▶│  Bayesian Model │
│   (synthetic)   │     │  Engineering    │     │    Training     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Evaluation    │◀────│ Anomaly Scoring │◀────│    Posterior    │
│   & Reporting   │     │                 │     │    Samples      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Component Responsibilities

| Component | Module | Responsibility |
|-----------|--------|----------------|
| Data Generation | `data_generator.py` | Create synthetic security logs with known attack patterns |
| Feature Engineering | `features.py` | Aggregate events into entity-window features |
| Model Definition | `model.py` | Define hierarchical Bayesian model structure |
| Inference | `model.py` | Fit model using MCMC (NUTS sampler) |
| Scoring | `scoring.py` | Compute anomaly scores from posterior predictive |
| Evaluation | `evaluation.py` | Calculate PR-AUC, Recall@K metrics |
| Visualization | `visualization.py` | Generate diagnostic plots |
| CLI Interface | `cli.py` | User-facing command-line tools |

---

## Key Components

### 1. Synthetic Data Generator

Generates realistic security event logs with four attack patterns:

- **Brute Force**: High-frequency login attempts from single IP
- **Credential Stuffing**: Moderate attempts across many users from single IP
- **Geographic Anomaly**: Access from unusual locations
- **Device Anomaly**: Multiple new device fingerprints

### 2. Feature Engineering

Transforms raw events into modeling-ready features:

- Time-window aggregation (hourly/daily)
- Entity-level statistics (mean, variance)
- Temporal features (hour, day-of-week)
- Categorical encodings

### 3. Hierarchical Bayesian Model

Negative Binomial model with partial pooling:

```
μ ~ Exponential(0.1)        # Population mean
α ~ HalfNormal(2)           # Concentration
θ_entity ~ Gamma(μα, α)     # Entity rates (partial pooling)
φ ~ HalfNormal(2)           # Overdispersion
y ~ NegativeBinomial(θ, φ)  # Observations
```

### 4. MCMC Inference

Uses No-U-Turn Sampler (NUTS) for efficient posterior exploration:

- Automatic step-size tuning
- Adaptive trajectory lengths
- Diagnostic checks (R-hat, ESS, divergences)

### 5. Anomaly Scoring

Posterior predictive scoring:

```
score = -log p(y_obs | posterior predictive)
```

Incorporates full posterior uncertainty in score calculation.

### 6. Evaluation Metrics

Metrics appropriate for rare event detection:

- **PR-AUC**: Precision-Recall Area Under Curve
- **Recall@K**: Fraction of attacks in top K scores
- **Precision@K**: Fraction of top K that are attacks

---

## Next Steps

- [Theoretical Foundations](02_theoretical_foundations.md): Deep dive into Bayesian statistics and MCMC
- [Model Architecture](03_model_architecture.md): Detailed model specification
- [Implementation Guide](04_implementation_guide.md): Code walkthrough
- [API Reference](05_api_reference.md): Complete function documentation
