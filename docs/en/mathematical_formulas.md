# Mathematical Formulas: Complete Statistical Specification

## Overview

This document provides the complete mathematical specification of the Bayesian Security Anomaly Detection model. It is intended for statisticians, quantitative researchers, and those who want to understand the theoretical foundations.

---

## 1. Model Specification

### 1.1 Notation

| Symbol | Description |
|--------|-------------|
| $i$ | Entity index, $i \in \{1, \ldots, N\}$ |
| $t$ | Time window index, $t \in \{1, \ldots, T\}$ |
| $y_{it}$ | Observed event count for entity $i$ in window $t$ |
| $\theta_i$ | Entity-specific rate parameter |
| $\mu$ | Population mean rate |
| $\alpha$ | Concentration parameter (pooling strength) |
| $\phi$ | Overdispersion parameter |

### 1.2 Hierarchical Model

The full hierarchical model is:

$$
\begin{aligned}
\mu &\sim \text{Exponential}(\lambda = 0.1) \\
\alpha &\sim \text{HalfNormal}(\sigma = 2) \\
\phi &\sim \text{HalfNormal}(\sigma = 1) \\
\theta_i &\sim \text{Gamma}(\text{shape} = \mu\alpha, \text{rate} = \alpha) \\
y_{it} &\sim \text{NegativeBinomial}(\mu = \theta_i, \alpha = \phi)
\end{aligned}
$$

---

## 2. Prior Distributions

### 2.1 Population Mean ($\mu$)

$$\mu \sim \text{Exponential}(\lambda = 0.1)$$

**Properties:**
- Support: $\mu > 0$
- Mean: $\mathbb{E}[\mu] = 1/\lambda = 10$
- Variance: $\text{Var}[\mu] = 1/\lambda^2 = 100$

**PDF:**
$$p(\mu) = \lambda e^{-\lambda \mu} = 0.1 e^{-0.1\mu}$$

**Rationale:** Weakly informative prior allowing population mean rates from near-zero to 50+ events per window.

### 2.2 Concentration Parameter ($\alpha$)

$$\alpha \sim \text{HalfNormal}(\sigma = 2)$$

**Properties:**
- Support: $\alpha > 0$
- Mean: $\mathbb{E}[\alpha] = \sigma\sqrt{2/\pi} \approx 1.60$
- Mode: $\alpha = 0$

**PDF:**
$$p(\alpha) = \frac{2}{\sigma\sqrt{2\pi}} \exp\left(-\frac{\alpha^2}{2\sigma^2}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{\alpha^2}{8}\right)$$

**Rationale:** Allows both strong pooling (large $\alpha$) and weak pooling (small $\alpha$), letting data determine the appropriate shrinkage.

### 2.3 Overdispersion Parameter ($\phi$)

$$\phi \sim \text{HalfNormal}(\sigma = 1)$$

**Properties:**
- Support: $\phi > 0$
- Controls variance relative to Poisson

**PDF:**
$$p(\phi) = \frac{2}{\sqrt{2\pi}} \exp\left(-\frac{\phi^2}{2}\right)$$

**Rationale:** Security event counts typically show moderate overdispersion; this prior concentrates mass on reasonable values.

---

## 3. Entity-Level Distribution

### 3.1 Gamma Prior for Entity Rates

$$\theta_i \sim \text{Gamma}(\text{shape} = \mu\alpha, \text{rate} = \alpha)$$

**Properties:**
$$
\begin{aligned}
\mathbb{E}[\theta_i] &= \frac{\mu\alpha}{\alpha} = \mu \\
\text{Var}[\theta_i] &= \frac{\mu\alpha}{\alpha^2} = \frac{\mu}{\alpha}
\end{aligned}
$$

**PDF:**
$$p(\theta_i | \mu, \alpha) = \frac{\alpha^{\mu\alpha}}{\Gamma(\mu\alpha)} \theta_i^{\mu\alpha - 1} e^{-\alpha\theta_i}$$

### 3.2 Partial Pooling Interpretation

The concentration parameter $\alpha$ controls the degree of pooling:

| $\alpha$ Value | Effect | Entity Rates |
|----------------|--------|--------------|
| $\alpha \to 0$ | No pooling | $\theta_i$ vary freely |
| $\alpha \approx 1$ | Moderate pooling | Balanced shrinkage |
| $\alpha \to \infty$ | Complete pooling | $\theta_i \to \mu$ |

**Shrinkage Factor:**
$$\text{Shrinkage} = \frac{\alpha}{\alpha + n_i}$$

where $n_i$ is the number of observations for entity $i$.

---

## 4. Likelihood Function

### 4.1 Negative Binomial Distribution

$$y_{it} \sim \text{NegativeBinomial}(\mu = \theta_i, \alpha = \phi)$$

**Parameterization (mean-dispersion):**
$$p(y | \mu, \alpha) = \binom{y + \alpha - 1}{y} \left(\frac{\alpha}{\alpha + \mu}\right)^\alpha \left(\frac{\mu}{\alpha + \mu}\right)^y$$

**Properties:**
$$
\begin{aligned}
\mathbb{E}[y] &= \mu = \theta_i \\
\text{Var}[y] &= \mu + \frac{\mu^2}{\alpha} = \theta_i + \frac{\theta_i^2}{\phi}
\end{aligned}
$$

### 4.2 Overdispersion Ratio

$$\text{Overdispersion} = \frac{\text{Var}[y]}{\mathbb{E}[y]} = 1 + \frac{\mu}{\phi}$$

For $\phi = 1, \mu = 10$: Overdispersion = 11 (variance is 11x the mean)

### 4.3 Why Not Poisson?

Poisson assumes $\text{Var}[y] = \mathbb{E}[y]$. Security logs exhibit:
- Bursty behavior (variance >> mean)
- Heavy tails (extreme counts more common)

The Negative Binomial generalizes Poisson with an extra dispersion parameter.

---

## 5. Posterior Distribution

### 5.1 Joint Posterior

By Bayes' theorem:

$$p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) \propto p(\mathbf{y} | \boldsymbol{\theta}, \phi) \cdot p(\boldsymbol{\theta} | \mu, \alpha) \cdot p(\mu) \cdot p(\alpha) \cdot p(\phi)$$

**Expanded:**
$$
p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) \propto
\left[\prod_{i,t} p(y_{it} | \theta_i, \phi)\right]
\left[\prod_i p(\theta_i | \mu, \alpha)\right]
p(\mu) \cdot p(\alpha) \cdot p(\phi)
$$

### 5.2 Log-Posterior (for MCMC)

$$
\log p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) = \sum_{i,t} \log p(y_{it} | \theta_i, \phi) + \sum_i \log p(\theta_i | \mu, \alpha) + \log p(\mu) + \log p(\alpha) + \log p(\phi) + C
$$

### 5.3 Why MCMC is Necessary

The posterior integral is **intractable**:

$$p(\mu, \alpha, \phi, \boldsymbol{\theta} | \mathbf{y}) = \frac{p(\mathbf{y} | \boldsymbol{\theta}, \phi) p(\boldsymbol{\theta} | \mu, \alpha) p(\mu) p(\alpha) p(\phi)}{\int \int \int \int p(\mathbf{y} | \boldsymbol{\theta}', \phi') p(\boldsymbol{\theta}' | \mu', \alpha') p(\mu') p(\alpha') p(\phi') \, d\boldsymbol{\theta}' d\mu' d\alpha' d\phi'}$$

The denominator requires integrating over $(N + 3)$-dimensional space where $N$ is the number of entities.

**MCMC Solution**: Sample from the posterior instead of computing it analytically.

---

## 5A. Markov Chain Monte Carlo Theory

### 5A.1 Markov Chains

A **Markov chain** is a sequence $\{X_0, X_1, X_2, \ldots\}$ satisfying the Markov property:

$$P(X_{t+1} | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} | X_t) = T(X_{t+1} | X_t)$$

**Stationary Distribution**: A distribution $\pi$ is stationary if:

$$\pi(x') = \int T(x' | x) \pi(x) \, dx$$

**Ergodic Theorem**: For an ergodic chain with stationary distribution $\pi$:

$$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} f(X_t) = \mathbb{E}_\pi[f(X)] = \int f(x) \pi(x) \, dx$$

**Key Insight**: We can estimate expectations under $\pi$ by averaging over a single long chain.

### 5A.2 MCMC Diagnostics

**R-hat (Gelman-Rubin Statistic)**:

For $C$ chains of length $N$ each:

$$\hat{R} = \sqrt{\frac{\widehat{\text{Var}}^+(\theta)}{W}}$$

where $W$ is within-chain variance and $\widehat{\text{Var}}^+$ is pooled variance estimate.

**Interpretation**:
- $\hat{R} \approx 1$: Chains have converged
- $\hat{R} > 1.01$: Chains exploring different regions
- $\hat{R} > 1.1$: Serious convergence problems

**Effective Sample Size (ESS)**:

$$\text{ESS} = \frac{MN}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

where $\rho_k$ is autocorrelation at lag $k$.

**We use NUTS** (No-U-Turn Sampler), an adaptive HMC variant that automatically tunes step size.

---

## 6. Anomaly Scoring

### 6.1 Posterior Predictive Distribution

For a new observation $y^*$ from entity $i$:

$$p(y^* | \mathbf{y}) = \int p(y^* | \theta_i, \phi) \cdot p(\theta_i, \phi | \mathbf{y}) \, d\theta_i \, d\phi$$

**Monte Carlo Approximation:**
$$p(y^* | \mathbf{y}) \approx \frac{1}{S} \sum_{s=1}^{S} p(y^* | \theta_i^{(s)}, \phi^{(s)})$$

where $(\theta_i^{(s)}, \phi^{(s)})$ are posterior samples.

### 6.2 Anomaly Score Definition

$$\text{score}(y_{it}) = -\log p(y_{it} | \mathbf{y}_{-it})$$

**Interpretation:**
- Score is the "surprise" of observing $y_{it}$
- Higher score = less likely under the model
- Scale: natural log (nats)

### 6.3 Numerical Computation

Using log-sum-exp for stability:

$$
\log p(y | \mathbf{y}) = \log \left(\frac{1}{S} \sum_{s=1}^{S} p(y | \theta^{(s)}, \phi^{(s)})\right)
= \text{logsumexp}\left(\log p(y | \theta^{(s)}, \phi^{(s)})\right) - \log S
$$

---

## 7. Credible Intervals

### 7.1 Posterior Predictive Intervals

For entity $i$, the $(1-\alpha)$ credible interval $[L_i, U_i]$:

$$P(L_i \leq y^*_i \leq U_i | \mathbf{y}) = 1 - \alpha$$

**Computation:**
1. Draw $\theta_i^{(s)}, \phi^{(s)}$ from posterior
2. Generate $y^{*(s)} \sim \text{NegBinomial}(\theta_i^{(s)}, \phi^{(s)})$
3. $L_i = \text{quantile}(y^{*(1:S)}, \alpha/2)$
4. $U_i = \text{quantile}(y^{*(1:S)}, 1 - \alpha/2)$

### 7.2 Interval Width as Uncertainty

$$\text{Uncertainty}_i = U_i - L_i$$

Entities with:
- **Narrow intervals**: Confident predictions (lots of data)
- **Wide intervals**: Uncertain predictions (sparse data → more pooling)

---

## 8. Evaluation Metrics

### 8.1 Precision-Recall Metrics

**Precision:**
$$\text{Precision} = \frac{TP}{TP + FP} = P(\text{Attack} | \text{Flagged})$$

**Recall:**
$$\text{Recall} = \frac{TP}{TP + FN} = P(\text{Flagged} | \text{Attack})$$

**PR-AUC:**
$$\text{PR-AUC} = \int_0^1 P(r) \, dr$$

where $P(r)$ is precision at recall level $r$.

### 8.2 Recall@K

$$\text{Recall@K} = \frac{|\{\text{attacks in top } K\}|}{|\{\text{all attacks}\}|}$$

**Operational interpretation:** If analysts investigate $K$ alerts per day, what fraction of attacks do they catch?

### 8.3 Baseline Comparison

**Random baseline PR-AUC:**
$$\text{PR-AUC}_{\text{random}} = \frac{|\{\text{attacks}\}|}{|\{\text{all observations}\}|} = \text{attack rate}$$

Our model should significantly exceed this baseline.

---

## 9. Model Properties

### 9.1 Conjugacy

The Gamma-Negative Binomial model is not fully conjugate, but:
- Gamma is conjugate to Gamma (for hierarchical rates)
- Enables efficient Gibbs steps if desired

### 9.2 Exchangeability

Entities are exchangeable a priori:
$$p(\theta_1, \ldots, \theta_N | \mu, \alpha) = p(\theta_{\pi(1)}, \ldots, \theta_{\pi(N)} | \mu, \alpha)$$

for any permutation $\pi$.

### 9.3 Posterior Consistency

As data increases, the posterior concentrates on true parameters:
$$p(\theta_i | \mathbf{y}) \xrightarrow{n_i \to \infty} \delta_{\theta_i^*}$$

---

## 10. Extensions

### 10.1 Temporal Component

Add autoregressive structure:
$$\theta_{i,t} = \rho \theta_{i,t-1} + (1-\rho)\mu_i + \epsilon_{it}$$

### 10.2 Multi-Feature Model

Extend to multivariate counts:
$$\mathbf{y}_{it} \sim \text{MultivariateNegBinom}(\boldsymbol{\theta}_i, \boldsymbol{\Sigma})$$

### 10.3 Variational Approximation

Replace MCMC with variational inference:
$$q^*(\boldsymbol{\theta}) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q || p(\boldsymbol{\theta} | \mathbf{y}))$$

---

## Summary Table

| Component | Distribution | Parameters |
|-----------|--------------|------------|
| Population mean | $\mu \sim \text{Exp}(0.1)$ | $\mathbb{E}[\mu] = 10$ |
| Concentration | $\alpha \sim \text{HalfNormal}(2)$ | Controls pooling |
| Overdispersion | $\phi \sim \text{HalfNormal}(1)$ | Controls variance |
| Entity rate | $\theta_i \sim \text{Gamma}(\mu\alpha, \alpha)$ | $\mathbb{E}[\theta_i] = \mu$ |
| Observation | $y_{it} \sim \text{NegBinom}(\theta_i, \phi)$ | $\mathbb{E}[y] = \theta_i$ |
| Anomaly score | $-\log p(y | \text{posterior})$ | Higher = more anomalous |

---

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis, 3rd Edition*. Chapter 5 (Hierarchical Models).

2. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*.

3. Hilbe, J. M. (2011). *Negative Binomial Regression, 2nd Edition*.

4. Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
