# Bayesian vs Classical Inference

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLASSICAL (FREQUENTIST) APPROACH                      │
└─────────────────────────────────────────────────────────────────────────┘

    Data (y)  ──────────────────────────┐
                                        ▼
                            ┌───────────────────────┐
                            │  Maximum Likelihood   │
                            │    Estimation (MLE)   │
                            └───────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │   Point Estimate θ̂    │
                            │   (Single Value)      │
                            └───────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  Confidence Interval  │
                            │   (from sampling      │
                            │    distribution)      │
                            └───────────────────────┘

    ❌ LIMITATIONS:
    • No incorporation of prior knowledge
    • Point estimates don't quantify uncertainty
    • Confidence intervals are about sampling, not parameters
    • Struggles with sparse data (overfitting)
    • No regularization mechanism


┌─────────────────────────────────────────────────────────────────────────┐
│                         BAYESIAN APPROACH                                │
└─────────────────────────────────────────────────────────────────────────┘

    Prior P(θ)     Data (y)
         │              │
         └──────┬───────┘
                ▼
    ┌───────────────────────┐
    │   Bayes' Theorem      │
    │                       │
    │  P(θ|y) ∝ P(y|θ)P(θ) │
    │  ─────────────────    │
    │       P(y)            │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Posterior P(θ|y)     │
    │  (Full Distribution)  │
    │                       │
    │      ▁▂▃▅▇▅▃▂▁       │
    │     ╱         ╲      │
    │    ╱           ╲     │
    │   ╱             ╲    │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  Credible Interval    │
    │  (95% of posterior    │
    │   mass lies here)     │
    └───────────────────────┘

    ✅ ADVANTAGES:
    • Incorporates prior knowledge/constraints
    • Full uncertainty quantification
    • Natural regularization (priors prevent overfitting)
    • Works well with sparse data
    • Interpretable: "95% probability θ is in [a, b]"


┌─────────────────────────────────────────────────────────────────────────┐
│                      EXAMPLE: ANOMALY DETECTION                          │
└─────────────────────────────────────────────────────────────────────────┘

CLASSICAL:
    Entity with 5 observations: [2, 3, 2, 50, 3]
    → Mean = 12, Std = 21
    → Threshold = Mean + 2×Std = 54
    → Result: Nothing flagged (50 < 54)
    ❌ Overestimates due to the outlier itself!

BAYESIAN:
    Prior: θ ~ Gamma(μα, α) where μ = 5 (population mean)
    Likelihood: y ~ NegBin(θ, φ)
    → Posterior: Regularized toward μ = 5
    → Predicted mean ≈ 7 (shrunk from 12)
    → 95% CI: [3, 15]
    → Observation of 50 is far outside CI
    ✅ Correctly flags as anomaly!


┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY DIFFERENCES                                  │
└─────────────────────────────────────────────────────────────────────────┘

| Aspect              | Classical              | Bayesian                  |
|---------------------|------------------------|---------------------------|
| **Output**          | Point estimate         | Full distribution         |
| **Uncertainty**     | Confidence interval    | Credible interval         |
| **Interpretation**  | Sampling-based         | Probability statement     |
| **Prior Knowledge** | Not used               | Incorporated via priors   |
| **Sparse Data**     | Unreliable estimates   | Regularized via pooling   |
| **Computation**     | Analytic (fast)        | MCMC (slower)             |
| **Use Case**        | Large datasets         | Any size, especially small|
