# Case Study: Bayesian Anomaly Detection for Security Event Logs

## Project Overview

**Project Name:** Bayesian Security Anomaly Detection (BSAD)
**Domain:** Cybersecurity / Machine Learning
**Techniques:** Bayesian Inference, Hierarchical Modeling, MCMC
**Stack:** Python, PyMC, Pandas, ArviZ, Typer CLI

---

## The Problem

Security teams at organizations of all sizes face an overwhelming challenge: **identifying malicious activity hidden within millions of legitimate events**. Traditional rule-based systems and simple threshold approaches fail in practice:

| Challenge | Impact |
|-----------|--------|
| **High false positive rates** | Alert fatigue; analysts ignore warnings |
| **No uncertainty quantification** | Can't prioritize investigations |
| **One-size-fits-all thresholds** | Miss targeted attacks on low-activity users |
| **Poor generalization** | Overfit to entities with sparse history |

Consider a real scenario: A user who typically generates 10 login events per hour suddenly generates 50. Is this an attack? With a fixed threshold of 100, this anomaly goes undetected. But for this specific user, 50 events represents a 5x increase—highly suspicious.

---

## The Solution

BSAD implements a **hierarchical Bayesian approach** that addresses these challenges through principled probabilistic modeling:

### Key Innovation: Partial Pooling

Instead of treating each user independently (complete separation) or applying global thresholds (complete pooling), we use **partial pooling**:

```
Entity-specific behavior ← Learned from individual history
                        ← Regularized by population statistics
```

This means:
- **High-activity users** get personalized baselines from their own data
- **Low-activity users** borrow strength from the population, avoiding overfitting
- **The model automatically decides** how much to pool based on data availability

### Uncertainty-Aware Scoring

Our anomaly scores aren't arbitrary numbers—they're derived from **posterior predictive probabilities**:

```
anomaly_score = -log P(observed_count | posterior)
```

A score of 8.5 means "this observation is extremely unlikely given everything we know about this entity and the population." The probabilistic foundation means we can:

- Provide credible intervals alongside point estimates
- Quantify model confidence for each prediction
- Make principled decisions under uncertainty

---

## Technical Approach

### Model Architecture

```
Hierarchical Negative Binomial Model
====================================

Population Level:
  μ ~ Exponential(0.1)      # Population mean rate
  α ~ HalfNormal(2)         # Concentration parameter

Entity Level (partial pooling):
  θ_entity ~ Gamma(μ, α)    # Entity-specific rate

Observation Level:
  y_count ~ NegBinomial(θ_entity, overdispersion)
```

**Why Negative Binomial?** Security event counts exhibit overdispersion (variance > mean) due to bursty behavior. The Negative Binomial naturally handles this, unlike Poisson models.

### Attack Patterns Detected

The system identifies four common attack signatures:

| Pattern | Detection Signal |
|---------|-----------------|
| **Brute Force** | Extreme event burst from single source |
| **Credential Stuffing** | Elevated activity across multiple targets |
| **Geo Anomaly** | Activity from unusual locations |
| **Device Anomaly** | New device fingerprints appearing |

---

## Results

### Performance Metrics

On synthetic data with 2% attack rate:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **PR-AUC** | 0.847 | Strong precision-recall tradeoff |
| **Recall@50** | 0.412 | 41% of attacks in top 50 alerts |
| **Recall@100** | 0.623 | 62% of attacks in top 100 alerts |

### Why These Metrics?

For rare events like security attacks, **PR-AUC is more meaningful than ROC-AUC**:

- ROC-AUC can be misleadingly high (0.95+) even with poor precision
- PR-AUC directly reflects the precision/recall tradeoff analysts face
- Recall@K answers the operational question: "How many attacks will I catch if I investigate K alerts per day?"

---

## Implementation Highlights

### Clean Pipeline Architecture

The codebase prioritizes **simplicity and readability**:

```
src/bsad/
├── config.py      # All settings in one dataclass
├── io.py          # File I/O helpers
├── steps.py       # Pure functions for each step
├── pipeline.py    # Orchestration (the ONE coordinator)
└── cli.py         # Thin CLI layer
```

**Design Principle:** Steps are pure functions that don't call each other. The Pipeline class controls all orchestration.

### Reproducibility

Every run is reproducible:

```bash
# Same command always produces same results
bsad demo --seed 42 --n-entities 200 --n-days 30
```

All random state is controlled through explicit seeds.

### Diagnostic Tools

Built-in MCMC diagnostics ensure model quality:

- **R-hat < 1.01**: Chains have converged
- **ESS > 400**: Sufficient effective samples
- **Divergences = 0**: No numerical issues

---

## Business Value

### For Security Teams

| Benefit | Impact |
|---------|--------|
| **Reduced alert fatigue** | Entity-specific baselines = fewer false positives |
| **Prioritized investigations** | Uncertainty quantification guides analyst focus |
| **Explainable alerts** | "This user's activity is 3σ above their normal" |

### For Organizations

| Benefit | Impact |
|---------|--------|
| **Better resource allocation** | Focus analyst time on high-confidence alerts |
| **Audit trail** | Probabilistic scores provide defensible decisions |
| **Adaptability** | Model automatically adjusts to entity behavior changes |

---

## Skills Demonstrated

This project showcases:

### Machine Learning & Statistics
- Bayesian inference and MCMC sampling
- Hierarchical/multilevel modeling
- Probabilistic programming with PyMC
- Model diagnostics and convergence analysis

### Software Engineering
- Clean, maintainable architecture
- Pure functional pipeline design
- Comprehensive CLI with Typer
- Reproducible experiments

### Domain Knowledge
- Security event log analysis
- Attack pattern recognition
- Evaluation metrics for imbalanced classification

### Communication
- Technical documentation (you're reading it!)
- Visualization of complex probabilistic concepts
- Clear separation of concerns for different audiences

---

## Try It Yourself

```bash
# Clone and install
git clone https://github.com/yourusername/bayesian-security-anomaly-detection.git
cd bayesian-security-anomaly-detection
pip install -e ".[dev]"

# Run complete demo
bsad demo --output-dir outputs/

# View results
cat outputs/metrics.json
ls outputs/plots/
```

---

## Future Directions

| Enhancement | Benefit |
|-------------|---------|
| **Temporal modeling** | Capture behavior drift over time |
| **Multi-feature extension** | Include bytes, endpoints, session duration |
| **Online learning** | Incremental updates as new data arrives |
| **Variational inference** | Scale to millions of entities |

---

## Contact

For questions about this project or collaboration opportunities, please reach out via GitHub Issues or LinkedIn.

---

*This case study is part of a portfolio demonstrating applied machine learning for cybersecurity.*
