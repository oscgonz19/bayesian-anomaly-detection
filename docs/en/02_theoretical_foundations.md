# Theoretical Foundations

## Table of Contents
1. [Bayesian Statistics Fundamentals](#bayesian-statistics-fundamentals)
2. [Count Data Distributions](#count-data-distributions)
3. [Hierarchical Models](#hierarchical-models)
4. [Markov Chain Monte Carlo (MCMC)](#markov-chain-monte-carlo-mcmc)
5. [Anomaly Detection Theory](#anomaly-detection-theory)

---

## Bayesian Statistics Fundamentals

### Bayes' Theorem

The foundation of Bayesian inference is Bayes' theorem, which relates conditional probabilities:

```
P(θ|y) = P(y|θ) × P(θ) / P(y)
```

In the context of statistical inference:

| Term | Name | Interpretation |
|------|------|----------------|
| P(θ\|y) | **Posterior** | Probability of parameters given observed data |
| P(y\|θ) | **Likelihood** | Probability of data given parameters |
| P(θ) | **Prior** | Probability of parameters before seeing data |
| P(y) | **Marginal Likelihood** | Normalizing constant (evidence) |

Since P(y) is constant with respect to θ, we often write:

```
posterior ∝ likelihood × prior
```

### Frequentist vs. Bayesian Interpretation

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Parameters** | Fixed, unknown constants | Random variables with distributions |
| **Data** | Random (varies across samples) | Fixed (observed once) |
| **Inference** | Point estimates + confidence intervals | Full posterior distribution |
| **Interpretation** | "95% CI means: if we repeated this experiment many times, 95% of CIs would contain the true value" | "95% credible interval means: 95% probability the parameter lies in this range given our data" |

### Prior Distributions

Priors encode our beliefs before seeing data. Common choices:

#### Weakly Informative Priors
Regularize inference without dominating the likelihood:

```python
# For positive rate parameters
μ ~ Exponential(0.1)  # Mean = 10, allows wide range

# For scale parameters
σ ~ HalfNormal(2)     # Positive, concentrated near 0-4
```

#### Conjugate Priors
When prior and posterior are in the same family:

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known σ) | Normal | Normal |
| Negative Binomial | Beta-Negative Binomial | Beta-Negative Binomial |

#### Why Priors Matter

1. **Regularization**: Prevent overfitting by constraining parameter space
2. **Incorporation of domain knowledge**: Security experts know typical event rates
3. **Handling sparse data**: Prior dominates when data is limited, preventing extreme estimates

### Posterior Computation

The posterior is often intractable analytically:

```
P(θ|y) = P(y|θ) × P(θ) / ∫ P(y|θ) P(θ) dθ
                         ↑
                    Often intractable
```

Solutions:
1. **Conjugate priors**: Closed-form posterior (limited applicability)
2. **Variational inference**: Approximate posterior with simpler distribution
3. **MCMC**: Generate samples from posterior (our approach)

---

## Count Data Distributions

### The Poisson Distribution

For count data, the Poisson distribution is a natural starting point:

```
P(y=k|λ) = (λ^k × e^(-λ)) / k!
```

Properties:
- Support: y ∈ {0, 1, 2, ...}
- Mean: E[y] = λ
- Variance: Var(y) = λ

**Limitation**: Variance equals mean (equidispersion). Security data is typically **overdispersed** (variance > mean).

### The Negative Binomial Distribution

Handles overdispersion by introducing a dispersion parameter:

```
P(y=k|μ,α) = Γ(k+α) / (Γ(α)k!) × (α/(α+μ))^α × (μ/(α+μ))^k
```

Where:
- μ = mean
- α = dispersion parameter (higher = less overdispersion)

Properties:
- Mean: E[y] = μ
- Variance: Var(y) = μ + μ²/α

As α → ∞, Negative Binomial → Poisson.

### Why Negative Binomial for Security Data?

Security event counts exhibit:

1. **Overdispersion**: Variance exceeds mean due to:
   - Bursty attack patterns
   - User behavior heterogeneity
   - Seasonal effects

2. **Zero-inflation**: Many entity-windows have zero events

3. **Heavy tails**: Occasional extreme values (attacks)

Example from typical security data:
```
Mean events/window:     8.3
Variance:              47.2
Variance/Mean ratio:    5.7  (overdispersed)
```

The Negative Binomial naturally captures this through its flexible variance structure.

### Alternative: Poisson-Gamma Mixture

The Negative Binomial can be derived as a Poisson with Gamma-distributed rate:

```
λ ~ Gamma(α, α/μ)
y | λ ~ Poisson(λ)

Marginalizing out λ:
y ~ NegativeBinomial(μ, α)
```

This interpretation is useful for hierarchical models where entity-specific rates come from a population distribution.

---

## Hierarchical Models

### The Hierarchical Paradigm

In hierarchical (multilevel) models, parameters themselves have distributions:

```
Level 3 (Hyperpriors):    μ, α ~ population hyperpriors
                              ↓
Level 2 (Entity params):  θ_i ~ population distribution(μ, α)
                              ↓
Level 1 (Data):           y_ij ~ likelihood(θ_i)
```

### Partial Pooling

Hierarchical models implement **partial pooling**—a compromise between:

| Approach | Description | Problem |
|----------|-------------|---------|
| **No pooling** | Estimate separate θ_i for each entity | Overfits with sparse data |
| **Complete pooling** | Single θ for all entities | Ignores entity heterogeneity |
| **Partial pooling** | Entity θ_i drawn from population distribution | Best of both worlds |

#### Mathematical Formulation

Entity rates are drawn from a population distribution:

```
θ_i ~ Gamma(μα, α)
```

Where:
- μ = population mean rate
- α = concentration (higher = less variation across entities)

The posterior for each θ_i becomes:

```
θ_i | y_i, μ, α ∝ Likelihood(y_i|θ_i) × Gamma(θ_i|μα, α)
```

Entities with:
- **Many observations**: Likelihood dominates, θ_i reflects entity-specific data
- **Few observations**: Prior dominates, θ_i shrinks toward population mean μ

### Shrinkage Illustration

Consider three entities with different data amounts:

```
Entity A: 1000 events observed, sample mean = 15
Entity B: 10 events observed, sample mean = 15
Entity C: 2 events observed, sample mean = 15
Population mean μ = 10

Posterior means (partial pooling):
Entity A: ~14.8 (mostly own data)
Entity B: ~12.1 (mixture)
Entity C: ~10.4 (shrunk toward population)
```

### Why Hierarchical Models for Security?

1. **Natural structure**: Users/IPs form a population with shared characteristics
2. **Borrowing strength**: New entities benefit from population-level learning
3. **Adaptive baselines**: Each entity gets personalized "normal" baseline
4. **Uncertainty quantification**: Sparse data = wider credible intervals

### Our Hierarchical Model

```
# Hyperpriors (population level)
μ ~ Exponential(0.1)      # Population mean event rate
α ~ HalfNormal(2)         # Concentration (entity variability)

# Entity-level parameters
θ_i ~ Gamma(μα, α)        # Entity-specific rate, for i = 1,...,N

# Overdispersion
φ ~ HalfNormal(2)         # Shared across entities

# Observations
y_ij ~ NegativeBinomial(θ_i, φ)  # Events for entity i, window j
```

---

## Markov Chain Monte Carlo (MCMC)

### The Sampling Problem

We need samples from the posterior:

```
θ^(1), θ^(2), ..., θ^(S) ~ P(θ|y)
```

Direct sampling is usually impossible. MCMC constructs a Markov chain whose stationary distribution is the target posterior.

### Markov Chain Basics

A Markov chain is a sequence where each state depends only on the previous:

```
P(θ^(t) | θ^(t-1), θ^(t-2), ..., θ^(1)) = P(θ^(t) | θ^(t-1))
```

With appropriate transition kernel, the chain converges to a stationary distribution.

### Metropolis-Hastings Algorithm

The foundational MCMC algorithm:

```
1. Initialize θ^(0)
2. For t = 1, 2, ..., S:
   a. Propose θ* from proposal distribution q(θ*|θ^(t-1))
   b. Compute acceptance probability:
      α = min(1, [P(θ*|y) × q(θ^(t-1)|θ*)] / [P(θ^(t-1)|y) × q(θ*|θ^(t-1))])
   c. With probability α: θ^(t) = θ* (accept)
      Otherwise: θ^(t) = θ^(t-1) (reject)
```

### Hamiltonian Monte Carlo (HMC)

Improves on random-walk Metropolis by using gradient information:

1. **Augment** parameter space with momentum variables
2. **Simulate Hamiltonian dynamics** to propose distant states
3. **Accept/reject** based on Hamiltonian conservation

Advantages:
- Explores parameter space more efficiently
- Less correlation between samples
- Scales better to high dimensions

### No-U-Turn Sampler (NUTS)

NUTS (our sampler) automatically tunes HMC:

- **Adaptive trajectory length**: Simulates until trajectory "turns around"
- **Automatic step size**: Dual averaging during warmup
- **No hand-tuning**: Eliminates difficult hyperparameter selection

```python
# PyMC uses NUTS by default
trace = pm.sample(
    draws=2000,      # Posterior samples
    tune=1000,       # Warmup for adaptation
    chains=4,        # Independent chains
    target_accept=0.9  # Acceptance rate target
)
```

### Convergence Diagnostics

#### R-hat (Gelman-Rubin Statistic)

Compares within-chain and between-chain variance:

```
R-hat = √(Var(combined) / Var(within-chain))
```

- R-hat ≈ 1.0: Chains have converged to same distribution
- R-hat > 1.05: Potential convergence issues

#### Effective Sample Size (ESS)

Accounts for autocorrelation in MCMC samples:

```
ESS = S / (1 + 2 × Σ_k ρ_k)
```

Where ρ_k is lag-k autocorrelation.

Guidelines:
- ESS > 400 for reliable posterior summaries
- ESS > 1000 for tail probabilities

#### Divergences

HMC-specific diagnostic. Divergences indicate:
- Posterior geometry problems
- Step size too large
- Pathological posterior shape

Solutions:
- Increase `target_accept` (smaller steps)
- Reparameterize model
- Use non-centered parameterization

---

## Anomaly Detection Theory

### What is an Anomaly?

An anomaly (outlier) is an observation that deviates significantly from expected behavior. In our probabilistic framework:

```
Anomaly = observation with low probability under learned model
```

### Posterior Predictive Distribution

The posterior predictive for new observation ỹ:

```
P(ỹ|y) = ∫ P(ỹ|θ) P(θ|y) dθ
```

This integrates out parameter uncertainty, giving predictions that account for model uncertainty.

Approximated via Monte Carlo:

```
P(ỹ|y) ≈ (1/S) × Σ_s P(ỹ|θ^(s))
```

### Anomaly Scoring

Our anomaly score is the negative log posterior predictive probability:

```
score(y_obs) = -log P(y_obs | y_train)
             = -log [ (1/S) × Σ_s P(y_obs | θ^(s)) ]
```

Properties:
- **Higher score = more anomalous**: Less probable under model
- **Incorporates uncertainty**: Averaged over posterior samples
- **Interpretable**: Based on probability theory

### Log-Sum-Exp Trick

For numerical stability, we compute:

```
score = -log Σ_s exp(log_lik_s) + log(S)

Using log-sum-exp:
logsumexp(x) = max(x) + log(Σ exp(x - max(x)))
```

### Why This Scoring Works

1. **Probabilistic foundation**: Directly measures "surprise" under model
2. **Handles uncertainty**: An observation might score high if:
   - It's far from the mean, OR
   - The model is confident the mean is elsewhere
3. **Entity-specific**: Each entity's expected behavior is learned from data
4. **Automatic calibration**: Scores are comparable across entities

### Score Uncertainty

We also report score uncertainty from posterior variation:

```python
# For each posterior sample s
score_s = -log P(y_obs | θ^(s))

# Summary statistics
score_mean = mean(score_s)
score_std = std(score_s)
score_95CI = [percentile(score_s, 2.5), percentile(score_s, 97.5)]
```

High score + low uncertainty → confident anomaly detection
High score + high uncertainty → flagged but investigate further

---

## References

1. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

2. McElreath, R. (2020). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press.

3. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

4. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.

5. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3), 1-58.

6. Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.

---

## Next: [Model Architecture](03_model_architecture.md)
