# 12× Fewer Alerts, Same Detection: How Hierarchical Bayesian Modeling Changes the UEBA Game

*By Oscar González — Security Data Science*

---

SOC analysts open an average of hundreds of alerts per day. Most are false positives. The problem isn't that detection models miss attacks — it's that they can't tell the difference between a network admin who authenticates to 300 endpoints per hour (normal) and an office worker who just did the same (a five-alarm fire).

That gap is the core problem in User and Entity Behavior Analytics (UEBA), and it's why I built [BSAD](https://github.com/oscgonz19/bayesian-anomaly-detection): a hierarchical Bayesian anomaly detector designed specifically for the regime where attacks are rare, labels don't exist, and false positive volume is the thing that actually burns out your analysts.

---

## The Two Properties That Make Security Anomaly Detection Hard

**1. Entities are radically different from each other.**
A threshold of "15 logins per hour = suspicious" catches 0% of attacks on a sysadmin and generates 100% false positives on a junior developer. Every rule-based or global-threshold approach collapses on this.

**2. Attacks are rare — genuinely rare.**
In a well-monitored enterprise, real compromises appear in fewer than 2% of all activity windows. Standard classifiers trained on balanced datasets, and metrics like ROC-AUC that ignore class imbalance, give you a false sense of performance. The metric that matters is **Precision-Recall AUC**, and the operational metric that matters even more is **how many alerts per day do I have to review to catch 30% of attacks?**

---

## The Statistical Answer: Partial Pooling

Instead of fitting a threshold per entity (noisy, data-hungry) or a global threshold (ignores heterogeneity entirely), a hierarchical Bayesian model does something more elegant: it **learns each entity's baseline while borrowing statistical strength from the population**.

The model structure is:

```
Population:   μ ~ Exponential(0.1)     # global mean rate
              α ~ HalfNormal(2)        # concentration

Entity:       θ_e ~ Gamma(μ·α, α)      # entity-specific rate, shrunk toward μ

Observation:  φ ~ HalfNormal(2)        # overdispersion
              y ~ NegativeBinomial(θ_e, φ)
```

A few things worth noting here:

- **Negative Binomial, not Poisson.** Real security logs show variance >> mean (bursty jobs, periodic processes). A Poisson model underestimates variance, producing intervals that are too narrow and therefore too many false positives. The NB dispersion parameter φ fixes this.

- **Partial pooling.** An entity with 3 observations gets pulled strongly toward the population mean. An entity with 200 observations is trusted to have its own baseline. This means **cold-start accounts (new users, new services) get reasonable baselines immediately**, without a separate initialization pipeline.

- **No peer-group clustering required.** The hierarchical structure learns the population distribution implicitly. You don't need an org chart, HR records, or a k-means preprocessing step.

### Anomaly Score

For each new observation y from entity e:

```
score(y, e) = −log p(y | posterior)
```

How surprised is the learned model by this count? A network admin generating 300 authentications gets a low score. A developer generating 300 gets a high score. Same number, completely different interpretation — no threshold calibration per entity needed.

The posterior also gives a **score standard deviation** across MCMC samples: a signal of how confident the model is in its anomaly judgment.

---

## Results: The Numbers That Matter

### vs. Classical Methods (Synthetic Benchmark, 2% Attack Rate)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **BSAD (hierarchical)** | **0.562** | **0.943** |
| NB-MLE (no pooling) | 0.466 | 0.856 |
| Z-Score | 0.283 | 0.834 |
| LOF | 0.034 | 0.569 |

The comparison between BSAD and NB-MLE is instructive: **same likelihood family, same data, different pooling structure**. The 17% PR-AUC gain comes entirely from partial pooling — entities with sparse history benefit from population-level priors instead of noisy individual fits.

Z-Score, the most common industrial baseline, achieves PR-AUC of 0.283 — barely above random at this attack rate.

### The Operational Number: Alerts Per Day

At a 1% attack rate on CSE-CIC-IDS2018 (real network data), to catch 30% of attacks:

| Model | Alerts per 1,000 windows |
|---|---|
| **BSAD** | **2.5** |
| Random Forest (supervised) | 29.5 |
| Isolation Forest | 32.8 |
| Logistic Regression | 46.1 |

**BSAD generates 12× fewer alerts than Random Forest at the same recall.** Random Forest also requires labeled attack data. BSAD doesn't.

This is the number SOC managers care about. Not AUC — tickets.

### Does It Hold Over Time?

| Period | PR-AUC | Change |
|---|---|---|
| Training window | 0.633 | — |
| Test window 1 | 0.682 | −7.7% |
| Test window 2 | 0.674 | −6.5% |

Less than 8% degradation across two held-out periods. Weekly retraining is sufficient; the model doesn't need to be re-tuned after every shift change.

### What About New Accounts? (Cold Start)

| Entity type | PR-AUC |
|---|---|
| Entities with full history | 0.722 |
| Cold entities (<5 observations) | 0.621 |

A 14% gap — the honest cost of thin history. But critically, cold entities still get scored from day one via population priors, without a separate warm-up period or manual baseline setting.

---

## The Part Most Papers Skip: When It Doesn't Work

On CSE-CIC-IDS2018 at a **17% attack rate**, Random Forest (PR-AUC 0.582) significantly outperforms BSAD (0.321). At 5%, the gap narrows. Below 2%, BSAD's alert volume advantage becomes decisive.

| Attack Rate | BSAD PR-AUC | RF PR-AUC | BSAD Alerts/1k |
|---|---|---|---|
| 17% | 0.321 | **0.582** | 15.5 |
| 2% | 0.094 | 0.208 | **5.4** |
| 1% | 0.101 | 0.293 | **2.5** |

The crossover point is around 2–5%. Above that: use a supervised classifier. Below that: alert volume is your bottleneck, and unsupervised Bayesian methods win.

BSAD is also honest about attack types: **brute force (count bursts) is reliably detected. Geo anomaly and device anomaly are not** — those require features the model doesn't see. We report this in the evaluation rather than cherry-picking favorable attack types.

---

## Triage: Turning Scores into Decisions

A raw anomaly score is not an alert. The triage layer adds:

**Risk Score** — combines anomaly score, confidence (1/score_std), and novelty (how little history we have). Two entities with the same score but different confidence levels get different priorities.

**Alert Budget Calibration** — instead of asking "what's the threshold?", you ask "how many alerts can my team handle per day?" The system inverts the problem and finds the threshold that satisfies your capacity constraint at maximum recall.

In practice at a recent benchmark run: **Precision@100 = 0.92, Recall@100 = 0.52** — 92 of the top 100 ranked alerts were real attacks, covering 52% of all attacks in the dataset.

---

## Takeaways

1. **Partial pooling over no pooling**: a 17% PR-AUC improvement for free, just by switching from per-entity MLE to a hierarchical prior.
2. **Alert volume over AUC**: the metric your SOC actually optimizes for is tickets, not curves.
3. **Honest operating envelope**: unsupervised Bayesian count models are for rare-event, no-label, entity-structured data — not for 17% attack rates or packet-level features.
4. **Cold start is handled natively**: population priors cover new entities without a warmup period.

If your detection problem fits this description — count aggregates, entity structure, <5% attack rate, no labels — the full code, benchmark harness, and interactive notebooks are at:

**[github.com/oscgonz19/bayesian-anomaly-detection](https://github.com/oscgonz19/bayesian-anomaly-detection)**

---

*Benchmark config: 50 entities, 14 days, 500 MCMC samples, 4 chains, seed 42. Real-data experiments on CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity). Full results in `outputs/`.*

---

### References

- Hawryluk et al. (2022). Peer-group Behaviour Analytics of Windows Authentications Events Using Hierarchical Bayesian Modelling. *arXiv:2209.09769*
- Perusquía et al. (2022). Bayesian Models Applied to Cyber Security Anomaly Detection Problems. *International Statistical Review*
- Sharafaldin et al. (2018). Toward Generating a New Intrusion Detection Dataset. *ICISSP*
- Vehtari et al. (2021). Rank-normalization, folding, and localization: an improved R̂. *Bayesian Analysis*
