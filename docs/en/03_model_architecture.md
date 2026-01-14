# Model Architecture

## Table of Contents
1. [Model Specification](#model-specification)
2. [Prior Selection Rationale](#prior-selection-rationale)
3. [Parameterization Details](#parameterization-details)
4. [Model Implementation](#model-implementation)
5. [Inference Configuration](#inference-configuration)
6. [Model Diagnostics](#model-diagnostics)

---

## Model Specification

### Graphical Model

```
                    ┌─────────┐
                    │   μ     │  Population mean rate
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │  θ_1   │ │  θ_2   │ │  θ_N   │  Entity-specific rates
         └────┬───┘ └────┬───┘ └────┬───┘
              │          │          │
              │          │          │          ┌─────────┐
              │          │          │          │   φ     │  Overdispersion
              ▼          ▼          ▼          └────┬────┘
         ┌────────┐ ┌────────┐ ┌────────┐         │
         │  y_1   │ │  y_2   │ │  y_N   │◀────────┘
         └────────┘ └────────┘ └────────┘  Observed counts

    Hyperpriors: μ ~ Exp(0.1), α ~ HalfNormal(2)
    Entity priors: θ_i ~ Gamma(μα, α)
    Likelihood: y_i ~ NegBinom(θ_i, φ)
```

### Mathematical Specification

**Level 1: Hyperpriors**
```
μ ~ Exponential(λ = 0.1)
α ~ HalfNormal(σ = 2)
φ ~ HalfNormal(σ = 2)
```

**Level 2: Entity Parameters**
```
θ_i ~ Gamma(shape = μ·α, rate = α)    for i = 1, ..., N_entities
```

**Level 3: Observations**
```
y_ij ~ NegativeBinomial(mu = θ_i, alpha = φ)    for j = 1, ..., n_i
```

Where:
- μ = population mean event rate
- α = concentration parameter (controls entity variance)
- φ = overdispersion parameter
- θ_i = entity i's mean event rate
- y_ij = observed events for entity i in window j

### Induced Distributions

**Marginal prior on θ_i:**
```
E[θ_i] = μ
Var(θ_i) = μ/α
```

Higher α means entities are more similar to each other (less heterogeneity).

**Marginal distribution of y_ij:**
```
E[y_ij] = θ_i
Var(y_ij) = θ_i + θ_i²/φ
```

The variance exceeds the mean, capturing overdispersion.

---

## Prior Selection Rationale

### Population Mean μ ~ Exponential(0.1)

**Rationale:**
- Mean of Exponential(0.1) = 10 events/window
- 95% prior mass in [0.25, 30]
- Weakly informative: allows data to dominate while preventing extreme values

**Alternative considered:**
```
μ ~ Gamma(2, 0.2)  # More concentrated around 10
```

We chose Exponential for simplicity and minimal assumptions.

### Concentration α ~ HalfNormal(2)

**Rationale:**
- HalfNormal(2) gives 95% mass in [0, 4]
- α < 1: High entity heterogeneity
- α > 2: Moderate pooling
- Allows data to determine appropriate pooling level

**Effect of α:**
```
α = 0.1  → CV(θ) ≈ 3.16 (very heterogeneous)
α = 1.0  → CV(θ) ≈ 1.00 (moderate heterogeneity)
α = 10   → CV(θ) ≈ 0.32 (fairly homogeneous)
```

Where CV = coefficient of variation = σ/μ.

### Overdispersion φ ~ HalfNormal(2)

**Rationale:**
- Controls variance-to-mean ratio
- φ → ∞: Negative Binomial → Poisson
- φ = 1: Variance = 2×mean
- Allows flexible overdispersion fitting

### Prior Predictive Checks

Before fitting, we can sample from priors to verify sensibility:

```python
with model:
    prior_samples = pm.sample_prior_predictive(samples=1000)

# Check: Are simulated counts reasonable?
# Expect: Mostly 0-50 range, occasional higher values
```

---

## Parameterization Details

### Negative Binomial Parameterization

PyMC uses (μ, α) parameterization:

```
NegativeBinomial(mu=μ, alpha=α)

P(y=k) = Γ(k+α)/(Γ(α)k!) × (α/(α+μ))^α × (μ/(α+μ))^k
```

This differs from the (n, p) parameterization in scipy:
```
scipy: n = α, p = α/(α+μ)
pymc:  mu = μ, alpha = α
```

### Gamma Parameterization for θ

We parameterize the Gamma so that:
- E[θ] = μ (inherits population mean)
- Var(θ) = μ/α (controlled by concentration)

```python
# Shape-rate parameterization
θ ~ Gamma(alpha=μ*α, beta=α)

# This gives:
# E[θ] = (μ*α)/α = μ
# Var(θ) = (μ*α)/α² = μ/α
```

### Index Variables

For efficient computation, we use index variables:

```python
# entity_idx: maps each observation to its entity
# Shape: (n_observations,)
# Values: integers in [0, n_entities-1]

# Usage in model:
θ[entity_idx]  # Broadcasts entity rates to observations
```

---

## Model Implementation

### PyMC Model Code

```python
def build_hierarchical_negbinom_model(
    y: np.ndarray,
    entity_idx: np.ndarray,
    n_entities: int,
    config: ModelConfig,
) -> pm.Model:
    """
    Build hierarchical Negative Binomial model.

    Parameters
    ----------
    y : array of shape (n_obs,)
        Event counts per observation
    entity_idx : array of shape (n_obs,)
        Entity index for each observation
    n_entities : int
        Number of unique entities
    config : ModelConfig
        Model configuration
    """
    coords = {
        "entity": np.arange(n_entities),
        "obs": np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:
        # === Data ===
        entity_idx_data = pm.Data("entity_idx", entity_idx, dims="obs")
        y_data = pm.Data("y_obs", y, dims="obs")

        # === Hyperpriors ===
        # Population mean rate
        mu = pm.Exponential("mu", lam=config.mu_prior_rate)

        # Concentration (controls entity heterogeneity)
        alpha = pm.HalfNormal("alpha", sigma=config.alpha_prior_sd)

        # === Entity-level rates ===
        # Partial pooling: θ_i ~ Gamma with E[θ] = μ
        theta = pm.Gamma(
            "theta",
            alpha=mu * alpha,  # shape parameter
            beta=alpha,         # rate parameter
            dims="entity",
        )

        # === Overdispersion ===
        phi = pm.HalfNormal("phi", sigma=config.overdispersion_prior_sd)

        # === Likelihood ===
        pm.NegativeBinomial(
            "y",
            mu=theta[entity_idx_data],  # entity-specific rate
            alpha=phi,                   # shared overdispersion
            observed=y_data,
            dims="obs",
        )

    return model
```

### Dimension Handling

The model uses PyMC's coordinate system for clarity:

```python
coords = {
    "entity": np.arange(n_entities),  # Entity dimension
    "obs": np.arange(len(y)),         # Observation dimension
}

# theta has shape (n_entities,) with "entity" dimension
# y has shape (n_obs,) with "obs" dimension
```

### Vectorized Likelihood

The likelihood is fully vectorized:

```python
# Instead of:
for i in range(n_entities):
    for j in range(n_obs_i):
        y[i,j] ~ NegBinom(theta[i], phi)

# We write:
y ~ NegBinom(mu=theta[entity_idx], alpha=phi)
# Where theta[entity_idx] broadcasts to shape (n_obs,)
```

---

## Inference Configuration

### Default Settings

```python
@dataclass
class ModelConfig:
    # Sampling
    n_samples: int = 2000    # Posterior draws per chain
    n_tune: int = 1000       # Warmup/adaptation samples
    n_chains: int = 4        # Independent chains
    target_accept: float = 0.9  # NUTS acceptance rate
    random_seed: int = 42

    # Priors
    mu_prior_rate: float = 0.1
    alpha_prior_sd: float = 2.0
    overdispersion_prior_sd: float = 2.0
```

### Sampling Process

```python
with model:
    trace = pm.sample(
        draws=config.n_samples,
        tune=config.n_tune,
        chains=config.n_chains,
        target_accept=config.target_accept,
        random_seed=seed,
        cores=min(config.n_chains, os.cpu_count()),
        return_inferencedata=True,
    )
```

**Phases:**
1. **Tuning (1000 iterations)**: Adapt step size and mass matrix
2. **Sampling (2000 iterations)**: Generate posterior samples
3. **Post-processing**: Compute diagnostics, add posterior predictive

### Posterior Predictive Sampling

After fitting, generate predictions:

```python
with model:
    trace.extend(pm.sample_posterior_predictive(trace))
```

This adds `posterior_predictive` group to trace with simulated y values.

---

## Model Diagnostics

### Convergence Checks

```python
def get_model_diagnostics(trace: az.InferenceData) -> dict:
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])

    return {
        "r_hat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "converged": bool(summary["r_hat"].max() < 1.05),
    }
```

### Diagnostic Thresholds

| Metric | Good | Warning | Problem |
|--------|------|---------|---------|
| R-hat | < 1.01 | 1.01-1.05 | > 1.05 |
| ESS bulk | > 400 | 100-400 | < 100 |
| ESS tail | > 400 | 100-400 | < 100 |
| Divergences | 0 | 1-10 | > 10 |

### Troubleshooting

**High R-hat:**
- Increase `n_samples` and `n_tune`
- Check for multimodal posterior
- Reparameterize model

**Low ESS:**
- Increase `n_samples`
- Check for high autocorrelation
- Consider non-centered parameterization

**Divergences:**
- Increase `target_accept` (e.g., 0.95, 0.99)
- Reparameterize (non-centered)
- Use stronger priors

### Non-Centered Parameterization

If divergences occur, consider non-centered parameterization:

```python
# Centered (default):
theta ~ Gamma(mu*alpha, alpha)

# Non-centered:
theta_raw ~ Gamma(alpha, alpha)  # E[theta_raw] = 1
theta = mu * theta_raw           # Rescale
```

This can improve sampling geometry for difficult posteriors.

---

## Model Extensions

### Potential Extensions

1. **Temporal dynamics**: Add autoregressive component
   ```
   theta_t ~ Normal(rho * theta_{t-1}, sigma)
   ```

2. **Covariates**: Include features in rate
   ```
   log(theta_i) = X_i @ beta + epsilon_i
   ```

3. **Multiple observation types**: Joint model for counts and bytes
   ```
   y_count ~ NegBinom(theta_count, phi_count)
   y_bytes ~ LogNormal(theta_bytes, sigma_bytes)
   ```

4. **Zero-inflation**: For highly sparse data
   ```
   y ~ ZeroInflatedNegBinom(psi, mu, alpha)
   ```

---

## Next: [Implementation Guide](04_implementation_guide.md)
