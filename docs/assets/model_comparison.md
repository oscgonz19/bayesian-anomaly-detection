# Model Comparison: BSAD vs Classical Methods

## Key Insight: Different Tools for Different Problems

**BSAD is NOT a general-purpose anomaly detector.** It's designed for a specific problem class.

## Problem Class Comparison

| Aspect | BSAD Domain | Classical Methods Domain |
|--------|-------------|-------------------------|
| **Data Type** | COUNT data (integers) | Feature vectors (continuous) |
| **Structure** | Entity-based hierarchies | Independent samples |
| **Anomaly Type** | Count SPIKES per entity | Multivariate outliers |
| **Example** | Login attempts per user | Network flow features |
| **Overdispersion** | Required (Var >> Mean) | Not relevant |

## Experimental Results

### Scenario A: Entity-Based Count Data (BSAD Domain)

**Setup**: 50 entities, 200 time windows each, rare anomalies (1-5%)

```
📊 PR-AUC Results:
                      1%      2%      5%
────────────────────────────────────────
BSAD (Bayesian)    0.985   0.989   0.985  👑
Isolation Forest   0.631   0.672   0.683
One-Class SVM      0.570   0.697   0.651
LOF                0.031   0.034   0.100

📈 BSAD Advantage: +29 to +35 PR-AUC points
```

**Why BSAD Wins Here:**
- Entity-specific baselines capture individual behavior
- Negative Binomial handles overdispersion naturally
- Hierarchical structure shares information across entities
- Anomalies are COUNT SPIKES (exactly what BSAD detects)

### Scenario B: Multivariate Feature Data (Classical Domain)

**Setup**: UNSW-NB15 network flows with 8+ features (bytes, packets, rate, etc.)

```
📊 PR-AUC Results (5% attack rate):
                      PR-AUC
────────────────────────────
One-Class SVM         0.052  👑
Isolation Forest      0.025
LOF                   0.015
BSAD (Bayesian)       0.005

📈 Classical methods win when attacks are multivariate
```

**Why Classical Wins Here:**
- Attacks manifest in multiple features (bytes, duration, rate)
- No meaningful entity structure for count aggregation
- SVM/IF can model high-dimensional decision boundaries
- Anomalies are FEATURE PATTERNS, not count spikes

## Decision Framework

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
        │  (users, IPs, etc)  │           │  - Isolation Forest │
        └─────────────────────┘           │  - One-Class SVM    │
                    │                     │  - LOF              │
          ┌────────┴────────┐             └─────────────────────┘
          ▼                 ▼
    ┌──────────┐     ┌──────────────┐
    │   YES    │     │     NO       │
    └──────────┘     └──────────────┘
          │                 │
          ▼                 ▼
    ┌──────────┐     ┌──────────────┐
    │ Use BSAD │     │ Use Poisson/ │
    │          │     │ NegBin only  │
    └──────────┘     └──────────────┘
```

## BSAD Use Cases (Where It Excels)

1. **User Activity Monitoring**
   - Login attempts per user per hour
   - API requests per account per day
   - File access counts per employee

2. **Network Security**
   - Connection attempts per source IP
   - DNS queries per internal host
   - Failed auth attempts per account

3. **IoT/OT Security**
   - Sensor readings per device
   - Commands per PLC per minute
   - Messages per MQTT topic

4. **Application Security**
   - Errors per endpoint per hour
   - Cache misses per service
   - Queue depth per worker

## When NOT to Use BSAD

❌ Raw network flow features (bytes, duration, flags)
❌ No entity structure in data
❌ Anomalies are pattern-based, not spike-based
❌ Need for real-time, low-latency detection
❌ Very high-dimensional feature spaces

## Summary Table

| Criterion | Use BSAD | Use Classical |
|-----------|----------|---------------|
| Data type | Counts | Continuous |
| Entity structure | Required | Not needed |
| Anomaly type | Count spikes | Multivariate patterns |
| Interpretability | High (θ per entity) | Low-Medium |
| Uncertainty | Full posteriors | Point estimates |
| Speed | Slower (MCMC) | Fast |
| Best PR-AUC | +30 points in domain | Better outside domain |

## Conclusion

**BSAD is a specialist, not a generalist.**

- In its domain (entity-based count data), it achieves **+29-35 PR-AUC points** over classical methods
- Outside its domain (multivariate features), classical methods perform better
- Choose based on your data structure, not just because "Bayesian sounds better"
