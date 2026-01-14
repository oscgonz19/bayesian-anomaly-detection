<div align="center">

# 🛡️ BSAD: Bayesian Security Anomaly Detection

**Rare-event detection for security count data using hierarchical Bayesian modeling**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMC](https://img.shields.io/badge/PyMC-5.10+-orange.svg)](https://www.pymc.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![UNSW-NB15](https://img.shields.io/badge/dataset-UNSW--NB15-purple.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

[Start Here 🗺️](#-project-overview--navigation) •
[The Problem](#-the-problem) •
[When to Use](#-when-to-use-bsad) •
[Results](#-results) •
[Quick Start](#-quick-start)

[**🇪🇸 Versión en Español**](README_ES.md)

</div>

---

## 🎯 One-Line Summary

**BSAD detects rare COUNT ANOMALIES per ENTITY with uncertainty quantification—achieving +30 PR-AUC points over classical methods in its domain.**

---

## 🗺️ Project Overview & Navigation

This project demonstrates **when and why** to use Bayesian hierarchical modeling for anomaly detection, using UNSW-NB15 network traffic data as a comprehensive case study.

### 📚 Three Learning Paths

Choose your journey based on your needs:

| Path | Start Here | What You'll Learn |
|------|-----------|-------------------|
| **🎓 Theory & Practice** | [`01_end_to_end_walkthrough.ipynb`](notebooks/01_end_to_end_walkthrough.ipynb) | Complete BSAD tutorial: Bayesian inference, MCMC, hierarchical models, with synthetic data |
| **📊 Real Data Application** | [`02_unsw_nb15_real_data.ipynb`](notebooks/02_unsw_nb15_real_data.ipynb) | UNSW-NB15 transformation from classification (64% attacks) to rare-event detection (1-5% attacks) |
| **⚖️ Method Selection** | [`03_model_comparison.ipynb`](notebooks/03_model_comparison.ipynb) | When BSAD wins (+30 PR-AUC) vs when classical methods win |

### 📖 Deep Dives

| Document | Purpose |
|----------|---------|
| [`docs/assets/unsw_nb15_dataset_description.md`](docs/assets/unsw_nb15_dataset_description.md) | **What are network flows?** Comprehensive dataset documentation explaining why context matters |
| [`docs/assets/model_comparison.md`](docs/assets/model_comparison.md) | Decision framework: BSAD vs Isolation Forest vs One-Class SVM vs LOF |
| [`docs/assets/posterior_predictive_scoring.md`](docs/assets/posterior_predictive_scoring.md) | How BSAD scores work: `-log P(y \| posterior)` |

### 🎯 Quick Decision: Should I Use BSAD?

**✅ YES** if your data has **ALL** of:
- COUNT data (integers: logins, requests, packets)
- Entity structure (users, IPs, services, devices)
- Rare anomalies (<5% attack rate)
- Overdispersion (Variance >> Mean)

**❌ NO** if you have:
- Continuous multivariate features → Use **Isolation Forest** or **One-Class SVM**
- High attack rates (>10%) → This is classification, use **Random Forest** or **XGBoost**
- No entity structure → Use classical anomaly detection

---

## ❌ The Problem

### Not All Anomaly Detection is Equal

There are **two fundamentally different problems** often confused as "anomaly detection":

| Aspect | Classification (Wrong for BSAD) | Rare-Event Detection (BSAD Domain) |
|--------|--------------------------------|-----------------------------------|
| **Attack Rate** | 50-70% | <5% |
| **Data Type** | Feature vectors | COUNT data |
| **Structure** | Independent samples | Entity hierarchies |
| **Example** | Network flow classification | Login attempts per user |
| **Best Tool** | Random Forest, SVM | **BSAD** |

### The Critical Insight

**BSAD is a SPECIALIST, not a generalist.**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ❌ WRONG USE CASE (Classification)                            │
│   ─────────────────────────────────────                         │
│   Dataset: 64% attacks, 36% normal                              │
│   Problem: "Is this flow malicious?"                            │
│   Best Tool: Random Forest, XGBoost, Neural Networks            │
│                                                                 │
│   ✅ CORRECT USE CASE (Rare-Event Detection)                    │
│   ──────────────────────────────────────────                    │
│   Dataset: 2% attacks, 98% normal                               │
│   Problem: "Is this user's activity count unusual?"             │
│   Best Tool: BSAD (Hierarchical Bayesian)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ When to Use BSAD

### Decision Framework

```
                    ┌─────────────────────────────────────┐
                    │     What type of data do you have?  │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
        ┌─────────────────────┐           ┌─────────────────────┐
        │  COUNT DATA         │           │  FEATURE VECTORS    │
        │  (integers)         │           │  (continuous)       │
        └─────────────────────┘           └─────────────────────┘
                    │                                   │
                    ▼                                   ▼
        ┌─────────────────────┐           ┌─────────────────────┐
        │  Entity structure?  │           │  Use Classical:     │
        │  (users, IPs, etc)  │           │  • Isolation Forest │
        └─────────────────────┘           │  • One-Class SVM    │
                    │                     │  • LOF              │
          ┌────────┴────────┐             └─────────────────────┘
          ▼                 ▼
     ┌─────────┐      ┌─────────────┐
     │   YES   │      │     NO      │
     │ → BSAD  │      │ → Classical │
     └─────────┘      └─────────────┘
```

### BSAD Checklist

Use BSAD when **ALL** of these apply:

- [x] **COUNT data**: Events, requests, packets, logins (integers)
- [x] **Entity structure**: Users, IPs, services, devices
- [x] **RARE anomalies**: Attack rate < 5%
- [x] **Overdispersion**: Variance >> Mean
- [x] **Need uncertainty**: Confidence intervals required

### Use Cases

| Domain | Entity | Count Variable | Anomaly Type |
|--------|--------|----------------|--------------|
| **SOC** | User ID | Login attempts/hour | Brute force |
| **API Security** | Endpoint | Requests/minute | Rate abuse |
| **Network** | Source IP | Connections/window | Port scanning |
| **IoT** | Device ID | Messages/interval | Botnet C&C |
| **Cloud Costs** | Service | Hourly spend | Resource abuse |

---

## 📊 Case Study: UNSW-NB15

### The Dataset

**UNSW-NB15** is a widely-used network intrusion detection dataset from the Australian Centre for Cyber Security.

> **📖 Full Dataset Description**: See [`docs/assets/unsw_nb15_dataset_description.md`](docs/assets/unsw_nb15_dataset_description.md) for comprehensive documentation on what network flows are, dataset structure, and why context matters.

| Property | Original | Problem |
|----------|----------|---------|
| Records | 257,673 flows | |
| Attack Rate | **64%** | ❌ This is CLASSIFICATION |
| Features | 49 features | ❌ Not count data natively |
| Entities | None explicit | ❌ No hierarchy (but implicit in `proto_service`) |

**Critical Understanding**: UNSW-NB15 contains *network flows*, not packets. Each row is a complete communication story between two machines. The dataset has implicit entity structure through traffic types (`proto_service`), which can be exploited for Bayesian modeling.

### Our Transformation: Rare-Attack Regime

We created proper anomaly detection datasets by resampling:

```
Original (64% attacks)  →  Rare-Attack Regime
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        ├─ 1% attacks (939 samples)
Keep ALL normals        ├─ 2% attacks (1,897 samples)
Subsample attacks   →   └─ 5% attacks (4,894 samples)
```

**Files Created:**
- `data/unsw_nb15_rare_attack_1pct.parquet`
- `data/unsw_nb15_rare_attack_2pct.parquet`
- `data/unsw_nb15_rare_attack_5pct.parquet`

### Why This Matters

| Regime | Attack Rate | Nature | BSAD Performance |
|--------|-------------|--------|------------------|
| Classification | 64% | Attacks are NORMAL | ❌ Poor fit |
| Rare-Event | 1-5% | Attacks are ANOMALIES | ✅ Excels |

---

## 🏆 Results

### Scenario A: Count Data with Entity Structure (BSAD Domain)

**Setup**: 50 entities, 200 time windows, rare anomalies (1-5%)

```
📊 PR-AUC Results:
                      1%      2%      5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BSAD (Bayesian)    0.985   0.989   0.985  👑 WINNER
Isolation Forest   0.631   0.672   0.683
One-Class SVM      0.570   0.697   0.651
LOF                0.031   0.034   0.100

📈 BSAD Advantage: +30 PR-AUC points over best classical
```

### Scenario B: Multivariate Features (Classical Domain)

**Setup**: UNSW-NB15 with 8 continuous features

```
📊 PR-AUC Results (5% attack rate):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One-Class SVM      0.052  👑 WINNER
Isolation Forest   0.025
LOF                0.015
BSAD (Bayesian)    0.005  (outside its domain)
```

### Key Insight

| Scenario | Winner | Advantage |
|----------|--------|-----------|
| Count data + Entities | **BSAD** | +30 PR-AUC pts |
| Multivariate features | **Classical** | Better fit |

**BSAD is a specialist that dominates in its domain.**

---

## 🔬 How BSAD Works

### The Model

```
Hierarchical Negative Binomial Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Population Level:
    μ ~ Exponential(λ)        # Global mean rate
    α ~ HalfNormal(σ_α)       # Pooling strength

Entity Level (partial pooling):
    θ[e] ~ Gamma(μα, α)       # Entity-specific rate
                              # Sparse entities → regularized to μ
                              # Dense entities → individualized

Observation Level:
    φ ~ HalfNormal(σ_φ)       # Overdispersion
    y[e,t] ~ NegBinomial(θ[e], φ)  # Count observation
```

### Anomaly Scoring

```
score = -log P(y | posterior)

For each observation:
1. Get posterior samples: θ^(s), φ^(s) for s = 1..S
2. Compute: P(y | θ^(s), φ^(s)) for each sample
3. Average: P(y | posterior) = (1/S) Σ P(y | θ^(s), φ^(s))
4. Score: -log(P) → Higher = more anomalous
```

### Why It Works

| Feature | Benefit |
|---------|---------|
| **Entity-specific θ[e]** | Different baselines per user/IP |
| **Hierarchical pooling** | Sparse entities regularized |
| **Negative Binomial** | Handles overdispersion (Var >> Mean) |
| **Full posterior** | Uncertainty quantification |

---

## 🚀 Quick Start

### Installation

```bash
# Clone
git clone git@github.com:oscgonz19/bayesian-anomaly-detection.git
cd bayesian-anomaly-detection

# Install
pip install -e ".[dev]"

# Verify
python -c "from bsad import Pipeline; print('OK')"
```

### Run Demo

```bash
# Generate synthetic data and train model
bsad demo --output-dir outputs/

# Or with Python
from bsad import Pipeline, Settings

settings = Settings(n_entities=200, n_days=30, attack_rate=0.02)
pipeline = Pipeline(settings)
pipeline.run_all()
```

### Explore Notebooks

**See the [📚 Three Learning Paths](#-project-overview--navigation) section above for detailed guidance on which notebook to start with.**

| Notebook | Key Concepts | Output |
|----------|--------------|--------|
| **01. End-to-End Walkthrough** | Bayesian inference, MCMC, hierarchical models, partial pooling, posterior predictive checks | Synthetic data demo with full theory |
| **02. UNSW-NB15 Real Data** | Statistical regimes (64% → 1-5%), network flows, overdispersion, entity structure, rare-attack transformation | Demonstrates why BSAD needs proper anomaly detection setup |
| **03. Model Comparison** | Scenario A (BSAD wins), Scenario B (Classical wins), uncertainty quantification, entity baselines | Head-to-head: +30 PR-AUC advantage in BSAD's domain |

**Visual Outputs Created:**
- 📊 `outputs/eda_case_study/` - 5 comprehensive EDA visualizations
- 📈 `outputs/rare_attack_comparison/` - Model comparison charts
- 🎯 All results demonstrate: **BSAD is a specialist, not a generalist**

---

## 📁 Project Structure

```
bayesian-security-anomaly-detection/
├── src/bsad/
│   ├── config.py          # Settings configuration
│   ├── steps.py           # Pure functions (data, model, scoring)
│   ├── pipeline.py        # Orchestration
│   ├── cli.py             # Command-line interface
│   └── unsw_adapter.py    # UNSW-NB15 data adapter
├── notebooks/
│   ├── 01_end_to_end_walkthrough.ipynb
│   ├── 02_unsw_nb15_real_data.ipynb
│   └── 03_model_comparison.ipynb
├── data/
│   ├── unsw_nb15_rare_attack_1pct.parquet
│   ├── unsw_nb15_rare_attack_2pct.parquet
│   └── unsw_nb15_rare_attack_5pct.parquet
├── outputs/
│   ├── eda_case_study/    # EDA visualizations
│   └── rare_attack_comparison/  # Comparison results
├── docs/
│   ├── en/                # English documentation
│   ├── es/                # Spanish documentation
│   └── assets/            # Visual guides
└── README.md
```

---

## 📚 Documentation

### Visual Guides

- [**Model Comparison**](docs/assets/model_comparison.md) - When to use BSAD vs Classical
- [**Bayesian vs Classical**](docs/assets/bayesian_vs_classical.md) - Why Bayesian wins for rare events
- [**Posterior Predictive Scoring**](docs/assets/posterior_predictive_scoring.md) - How scoring works

### EDA Visualizations

See `outputs/eda_case_study/`:
- `01_what_bsad_solves.png` - The problem BSAD addresses
- `02_unsw_nb15_analysis.png` - Dataset analysis
- `03_rare_attack_transformation.png` - Creating proper regime
- `04_model_comparison.png` - Results
- `05_summary_dashboard.png` - Complete summary

### Technical Documentation

- [Technical Report](docs/en/technical_report.md) - Full methodology
- [Mathematical Formulas](docs/en/mathematical_formulas.md) - Statistical specification

---

## 🧠 Key Takeaways

1. **BSAD is a SPECIALIST** for count-based, entity-structured, rare-event detection

2. **+30 PR-AUC points** advantage over classical methods in its domain

3. **Not for classification** - if attack rate >10%, use classical methods

4. **Statistical regime matters** more than the dataset itself

5. **Uncertainty quantification** enables confident decision-making

---

## 📖 Citation

```bibtex
@software{bsad2024,
  author = {Gonzalez, Oscar},
  title = {BSAD: Bayesian Security Anomaly Detection},
  year = {2024},
  url = {https://github.com/oscgonz19/bayesian-anomaly-detection}
}
```

---

## 🙏 Acknowledgments

- **PyMC** - Probabilistic programming framework
- **UNSW-NB15** - Network intrusion dataset
- **ArviZ** - Bayesian visualization

---

<div align="center">

**BSAD: The right tool for rare-event detection**

[⭐ Star this repo](https://github.com/oscgonz19/bayesian-anomaly-detection) | [📊 View Case Study](outputs/eda_case_study/)

</div>
