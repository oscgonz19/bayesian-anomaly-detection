<div align="center">

# 🛡️ BSAD: Bayesian Security Anomaly Detection

**Rare-event detection for security count data using hierarchical Bayesian modeling**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMC](https://img.shields.io/badge/PyMC-5.10+-orange.svg)](https://www.pymc.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![UNSW-NB15](https://img.shields.io/badge/dataset-UNSW--NB15-purple.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

[🔐 Security Problem](#-security-problem-addressed) •
[📚 Navigation](#-project-overview--navigation) •
[✅ When to Use](#-when-to-use-bsad) •
[❌ When NOT](#-when-not-to-use-bsad) •
[🧠 SOC Use Case](#-operational-use-case-soc-environment) •
[🚀 Quick Start](#-quick-start)

[**🇪🇸 Versión en Español**](README_ES.md)

</div>

---

## 🎯 The 30-Second Pitch

This project explores how **hierarchical Bayesian models** can be used for **behavioral anomaly detection** in network traffic. Instead of classifying attacks, it models what is **normal for each type of network activity** and flags **statistically improbable behavior** under rare-event regimes—particularly useful for detecting **low-and-slow threats** that evade traditional detectors.

**Key Result**: +30 PR-AUC points over classical methods when applied to count-based, entity-structured rare-event data.

<div align="center">

![BSAD Summary Dashboard](outputs/eda_case_study/05_summary_dashboard.png)
*Complete overview: The problem, solution, and results*

</div>

---

## 🔐 Security Problem Addressed

### What Threats Does This Detect?

This project focuses on **behavioral anomaly detection** in network environments, targeting threats that evade signature-based systems:

| Threat Type | Why Traditional Systems Miss It | Why BSAD Catches It |
|-------------|----------------------------------|---------------------|
| **Low-and-Slow Beaconing** | Spread over time, no single spike | Entity-specific baselines detect subtle deviations |
| **Insider Misuse** | Authorized access, normal protocols | Count patterns reveal unusual behavior for that user/service |
| **Long-term Reconnaissance** | APT-style gradual scanning | Rare-event regime optimized for <5% attack rates |
| **Zero-Day Exploits** | No known signatures | Behavioral deviation from established baselines |
| **Data Exfiltration** | Looks like normal traffic | Unusual packet/byte counts for specific protocol_service |

### The Core Security Insight

**Traditional signature-based detection**: "Does this match a known attack pattern?"
**BSAD approach**: "Is this behavior statistically improbable for this entity?"

Example:
- DNS query generating **50 packets** → 🚨 **Highly anomalous** (DNS normally 2-3 packets)
- HTTP session generating **50 packets** → ✅ **Normal** (HTTP typically 100+ packets)

**The same count means different things in different contexts.**

<div align="center">

![What BSAD Solves](outputs/eda_case_study/01_what_bsad_solves.png)
*Understanding what BSAD detects: count spikes, overdispersion, and entity-specific baselines*

</div>

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

## ❌ When NOT to Use BSAD

### This Approach is NOT Intended For

Be honest about limitations. BSAD is a specialist tool—use classical methods when:

| Problem Type | Why BSAD Fails | Use Instead |
|--------------|----------------|-------------|
| **Malware Classification** | Not designed for binary/multi-class classification | Random Forest, XGBoost, Deep Learning |
| **Signature-Based Detection** | No signature matching capability | YARA, Snort, Suricata |
| **Multivariate Feature Anomalies** | Designed for COUNT data, not feature vectors | Isolation Forest, One-Class SVM |
| **High Attack Rates (>10%)** | Rare-event assumptions break down | This becomes classification—use supervised learning |
| **Real-Time Detection (<100ms)** | MCMC inference is computationally intensive | Rule-based systems, pre-trained models |
| **No Entity Structure** | Requires grouping variable (users, IPs, services) | Global anomaly detection (LOF, Isolation Forest) |

### Example: See Notebook 03

**Scenario B** in [`03_model_comparison.ipynb`](notebooks/03_model_comparison.ipynb) explicitly demonstrates when BSAD performs **worse** than classical methods (multivariate features: PR-AUC 0.005 vs 0.052).

**This honesty is a feature, not a bug.** Professional data scientists know when their tools don't apply.

---

## 🧠 Operational Use Case: SOC Environment

### How This Would Work in Production

In a Security Operations Center (SOC), BSAD would be used as part of a **behavioral analytics layer**:

```
┌─────────────────────────────────────────────────────────┐
│  TRADITIONAL TIER (Signature-Based)                    │
│  ├─ Known malware signatures                           │
│  ├─ CVE-based exploit detection                        │
│  └─ Rule-based alerts                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  BEHAVIORAL TIER (BSAD)                                 │
│  ├─ Establish per-service/protocol baselines           │
│  ├─ Score deviations with uncertainty quantification   │
│  ├─ Prioritize under rare-event regimes                │
│  └─ Reduce false positives from static thresholds      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  ANALYST WORKFLOW                                       │
│  ├─ High-confidence anomalies (narrow credible interval)│
│  ├─ Context-aware: "unusual FOR this service"          │
│  └─ Reduced alert fatigue vs global thresholds         │
└─────────────────────────────────────────────────────────┘
```

### Operational Benefits

| Challenge | Traditional Approach | BSAD Approach |
|-----------|---------------------|---------------|
| **Alert Fatigue** | Global thresholds generate many false positives | Entity-specific baselines reduce noise |
| **Prioritization** | All anomalies treated equally | Uncertainty-aware risk scores |
| **Context Loss** | "100 packets is anomalous" (for what?) | "100 packets is anomalous for DNS" |
| **Rare Events** | Static thresholds miss subtle deviations | Optimized for <5% attack rates |
| **New Services** | No baseline until enough data | Partial pooling borrows strength from similar entities |

### Example Alert

```
🚨 High-Confidence Anomaly Detected

Entity: udp_dns (DNS traffic)
Observed Count: 47 packets in window
Expected: 2.3 packets (90% CI: [1.8, 2.9])
Anomaly Score: 28.4 (top 0.1%)
Confidence: High (narrow interval)

Recommendation: Investigate potential DNS tunneling or exfiltration
```

---

## 📈 Impact: BSAD vs Classical Methods

### Beyond PR-AUC: Strategic Comparison

| Aspect | Classical Methods (IF, OCSVM, LOF) | BSAD (Hierarchical Bayesian) |
|--------|-----------------------------------|------------------------------|
| **Rare-Event Detection** | Unstable with <5% attack rates | ✅ Designed for rare events |
| **Interpretability** | Black-box decision boundaries | ✅ Probabilistic, entity-specific baselines |
| **Uncertainty Awareness** | Point estimates only | ✅ Full posterior distributions |
| **Entity-Level Context** | Global anomaly detection | ✅ "Normal for user A ≠ normal for user B" |
| **SOC Prioritization** | Difficult (all scores equal weight) | ✅ Confidence intervals guide triage |
| **Training Speed** | ✅ Fast (minutes) | Slow (hours with MCMC) |
| **Inference Speed** | ✅ Real-time capable | Slow (not for <100ms requirements) |
| **Data Requirements** | Moderate | High (needs count data + entity structure) |

### When Each Wins

```
BSAD Wins:
  ✓ Count data + entity structure
  ✓ Rare anomalies (<5%)
  ✓ Need uncertainty quantification
  ✓ Operational context matters

Classical Wins:
  ✓ Multivariate continuous features
  ✓ No entity structure
  ✓ Speed critical (real-time)
  ✓ Prototyping/exploration
```

<div align="center">

![Head-to-Head Comparison](outputs/comparison/head_to_head_comparison.png)
*BSAD dominates in its domain (Scenario A), classical methods win elsewhere (Scenario B)*

</div>

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

<div align="center">

![Regime Transformation](outputs/unsw/regime_transformation.png)
*From classification (68% attacks) to true anomaly detection (1-5% attacks)*

</div>

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

## 📊 Results

### Scenario A: Count Data with Entity Structure (BSAD Domain)

**Setup**: 50 entities, 200 time windows, rare anomalies (1-5%)

```
Operational Performance (1-5% attack rate):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method              FPR@90%recall   Alerts/Day*   Maintains precision
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BSAD (Bayesian)         ~5%            ~50         ✓ under extreme rarity
Isolation Forest       ~35%           ~350         degrades at <2%
One-Class SVM          ~40%           ~400         degrades at <2%
LOF                    ~90%           ~900         unusable

* Simulated: 1000 events/day, targeting 90% attack detection
```

**Key finding**: BSAD maintains operable precision under extreme rarity while classical methods collapse into false positive noise.

### Scenario B: Multivariate Features (Classical Domain)

**Setup**: UNSW-NB15 with 8 continuous features (NOT count data)

```
Result: Classical methods outperform BSAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is EXPECTED. BSAD is designed for count data.
Use Isolation Forest or One-Class SVM here.
```

### Key Insight

| Data Type | Best Approach | Why |
|-----------|---------------|-----|
| Count data + Entities | **BSAD** | Entity baselines + overdispersion handling |
| Multivariate features | **Classical** | Geometric distance works well |

**BSAD is a specialist, not a generalist. Use the right tool for the job.**

### ⚠️ What BSAD Does NOT Detect

**Critical limitation**: BSAD only detects anomalies that alter event counts.

| Attack Type | BSAD Detection | Why |
|-------------|----------------|-----|
| **Brute force / scanning** | ✅ Detects | Count spike |
| **Beaconing / C2** | ✅ Detects | Unusual periodicity in counts |
| **Data exfiltration** | ✅ Detects | Unusual byte/packet counts |
| **Lateral movement (constant volume)** | ❌ Misses | No count change |
| **Payload-based exploits** | ❌ Misses | Content, not volume |
| **Credential theft** | ❌ Misses | Semantic, not statistical |
| **Zero-day with normal traffic patterns** | ❌ Misses | Looks normal statistically |

```
BSAD is a RATE anomaly detector, not a CONTENT anomaly detector.
It complements signature-based and payload-inspection systems.
```

### Computational Reality

| Aspect | Reality | Implication |
|--------|---------|-------------|
| **Training** | Hours (MCMC sampling) | Offline, batch process |
| **Scoring** | Milliseconds (lookup) | Online capable |
| **Retraining** | Weekly/monthly | Not real-time adaptive |
| **Scale** | ~100-1000 entities | Not for millions of unique IPs |

```
Architecture: Train OFFLINE → Score ONLINE → Retrain periodically

This is viable for:
  ✓ Batch SOC analytics
  ✓ Periodic baseline updates
  ✗ Inline IDS (use signatures)
  ✗ Real-time streaming (use simpler models)
```

### Why This Matters (Honest Framing)

**BSAD is not a general intrusion detector.**
It's a **probabilistic baseline model per entity** designed to:
- Reduce noise in rare-event regimes
- Prioritize alerts with statistical evidence
- Provide uncertainty quantification

```
The right framing for interviews:

"BSAD doesn't replace classical methods.
 It works as a probabilistic normalization layer
 that learns what's normal FOR EACH ENTITY
 and only raises alerts with strong statistical evidence—
 critical in SOCs suffering from alert fatigue."
```

### BSAD's Real Advantages (Not Captured by PR-AUC)

1. **No labeled data required** — works when you don't have attack examples
2. **Entity-specific baselines** — "normal for A ≠ normal for B"
3. **Uncertainty quantification** — know when to trust the score
4. **Interpretable** — "3σ above entity baseline" vs black-box score

### BSAD Unique Capabilities

<div align="center">

![Uncertainty Quantification](outputs/comparison/uncertainty_quantification.png)
*BSAD provides confidence intervals, not just point estimates*

</div>

<div align="center">

![Entity-Specific Baselines](outputs/comparison/entity_baselines.png)
*Each entity learns its own baseline rate with uncertainty bounds*

</div>

---

## 🔬 How BSAD Works

### The Model

<div align="center">

![Hierarchical Model Diagram](docs/images/hierarchical_model_diagram.png)
*Three-level hierarchical structure: Population → Entity → Observation*

</div>

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

<div align="center">

![Scoring Explanation](docs/images/scoring_explanation.png)
*How anomaly scores are computed: from distribution to ranked output with uncertainty*

</div>

### Why It Works

| Feature | Benefit |
|---------|---------|
| **Entity-specific θ[e]** | Different baselines per user/IP |
| **Hierarchical pooling** | Sparse entities regularized |
| **Negative Binomial** | Handles overdispersion (Var >> Mean) |
| **Full posterior** | Uncertainty quantification |

<div align="center">

![Partial Pooling Explained](docs/images/partial_pooling_explained.png)
*Partial pooling: sparse entities shrink to population mean, dense entities keep their own rate*

</div>

<div align="center">

![Overdispersion Analysis](outputs/unsw/overdispersion_analysis.png)
*Security data shows overdispersion: Variance >> Mean (all points above the Poisson line)*

</div>

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

## 👨‍💻 Relevant Professional Roles

This project demonstrates skills and approaches relevant for:

| Role | How This Project Applies |
|------|--------------------------|
| **Security Data Scientist** | Behavioral analytics, rare-event modeling, uncertainty quantification for threat detection |
| **Detection Engineer (Behavioral Analytics)** | Entity-specific baselines, anomaly scoring under rare-event regimes, SOC integration |
| **NDR / SOC Analytics Engineer** | Network behavior modeling, alert prioritization, context-aware detection |
| **Applied Bayesian ML Engineer** | Hierarchical modeling, MCMC implementation, posterior predictive scoring |
| **Threat Detection Researcher** | Novel detection methodologies, evaluation under realistic attack rates, honest method comparison |

### Skills Demonstrated

- ✅ **Domain Expertise**: Network security, intrusion detection, threat intelligence
- ✅ **Statistical Rigor**: Bayesian inference, MCMC, hierarchical models, model diagnostics
- ✅ **Engineering Pragmatism**: When to use vs when NOT to use specialized methods
- ✅ **Operational Thinking**: SOC workflows, alert fatigue, prioritization strategies
- ✅ **Research Quality**: Honest evaluation, two-scenario comparison, documented limitations

---

## 📁 Project Structure

<div align="center">

![Pipeline Architecture](docs/images/pipeline_architecture.png)
*End-to-end pipeline: from raw data to ranked anomalies with uncertainty*

</div>

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
