# Documentation - English

## Bayesian Security Anomaly Detection (BSAD)

Welcome to the BSAD documentation. This guide covers everything from theoretical foundations to practical implementation.

---

## Quick Navigation

### Getting Started
- **[Tutorial](06_tutorial.md)** - Step-by-step guide to using BSAD

### Understanding the System
- **[Overview](01_overview.md)** - System introduction and architecture
- **[Theoretical Foundations](02_theoretical_foundations.md)** - Bayesian statistics, MCMC, hierarchical models
- **[Model Architecture](03_model_architecture.md)** - Detailed model specification

### Implementation
- **[Implementation Guide](04_implementation_guide.md)** - Code walkthrough
- **[API Reference](05_api_reference.md)** - Complete function documentation

---

## Document Summary

| Document | Content | Audience |
|----------|---------|----------|
| Overview | Problem statement, system architecture, key components | All users |
| Theoretical Foundations | Bayesian inference, MCMC, Negative Binomial distribution | Data scientists, researchers |
| Model Architecture | Mathematical specification, PyMC implementation | ML engineers |
| Implementation Guide | Data generation, feature engineering, scoring | Developers |
| API Reference | Function signatures, parameters, examples | Developers |
| Tutorial | Installation, quick start, troubleshooting | All users |

---

## Reading Order

### For New Users
1. [Tutorial](06_tutorial.md) - Get running quickly
2. [Overview](01_overview.md) - Understand what the system does
3. [API Reference](05_api_reference.md) - Learn the CLI and functions

### For Data Scientists
1. [Theoretical Foundations](02_theoretical_foundations.md) - Understand the math
2. [Model Architecture](03_model_architecture.md) - Understand the model
3. [Implementation Guide](04_implementation_guide.md) - Understand the code

### For Security Professionals
1. [Overview](01_overview.md) - Problem statement and approach
2. [Tutorial](06_tutorial.md) - Run the demo
3. [Implementation Guide](04_implementation_guide.md) - Attack patterns section

---

## Key Concepts

### Bayesian Approach
- Prior distributions encode assumptions
- Posterior combines prior with observed data
- Uncertainty is quantified, not hidden

### Hierarchical Model
- Partial pooling shares information across entities
- Sparse entities borrow strength from population
- Data-rich entities follow their own patterns

### Anomaly Scoring
- Score = -log p(observation | model)
- Higher score = less probable = more anomalous
- Uncertainty in scores reflects model confidence

---

## Contributing

Found an error or want to improve the documentation?
Please submit an issue or pull request on GitHub.
