# Uncertainty Quantification vs Classical Thresholds

```
┌─────────────────────────────────────────────────────────────────────────┐
│             CLASSICAL THRESHOLD APPROACH (DETERMINISTIC)                 │
└─────────────────────────────────────────────────────────────────────────┘

    Historical Data → Compute Statistics
    
    [2, 5, 4, 6, 3, 5, 4] → μ = 4.14, σ = 1.35
    
    Threshold = μ + 2σ = 4.14 + 2.7 = 6.84
    
    ┌──────────────────────────────────────────────────┐
    │            Decision Boundary                     │
    │                                                  │
    │   ║                              │               │
    │   ║     NORMAL                   │   ANOMALY     │
    │   ║                              │               │
    │ ──╫──────────────────────────────┼────────────── │
    │   0      2      4      6     6.84│    10    15   │
    │   ║                              │               │
    │   ║  y=6 → OK                    │ y=8 → FLAG    │
    └──────────────────────────────────────────────────┘
    
    ❌ PROBLEMS:
    • Hard binary decision (no confidence measure)
    • Ignores uncertainty in parameter estimates
    • Same threshold for all entities
    • Sensitive to outliers in training data
    • No notion of "how anomalous"


┌─────────────────────────────────────────────────────────────────────────┐
│           BAYESIAN APPROACH (PROBABILISTIC WITH UNCERTAINTY)             │
└─────────────────────────────────────────────────────────────────────────┘

    Historical Data → Posterior Distribution
    
    [2, 5, 4, 6, 3, 5, 4] + Prior → P(θ | data)
    
    ┌──────────────────────────────────────────────────┐
    │         Posterior Predictive Distribution        │
    │                                                  │
    │            ╱‾‾‾╲                                │
    │           ╱     ╲                               │
    │          ╱       ╲                              │
    │         ╱         ╲___                          │
    │   ─────╱             ╲_____________             │
    │   0    2    4    6    8   10   12  15          │
    │        ├────────┤                               │
    │        90% CI                                   │
    │      [2.1, 7.8]                                 │
    └──────────────────────────────────────────────────┘
    
    ✅ ADVANTAGES:
    • Continuous score (not binary)
    • Quantifies uncertainty
    • Adapts per entity
    • Robust to sparse data
    • Interpretable probability


┌─────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE: TWO USERS, SAME OBSERVATION                  │
└─────────────────────────────────────────────────────────────────────────┘

USER A (Well-Characterized, 100 observations)
    Historical: [4,5,5,6,4,5,4,5,6,5, ...]
    
    Posterior Predictive (NARROW uncertainty):
    
         ║            ╱‾╲
         ║           ╱   ╲
         ║          ╱     ╲
         ║   ──────╱       ╲────────
         ║  0   2   4   6   8   10
         ║         ├─┤
         ║        90% CI
         ║      [3.8, 6.2]
    
    NEW OBSERVATION: y = 10
    → Far outside CI → Score = 7.2 → 🔴 HIGHLY ANOMALOUS


USER B (Sparse Data, 5 observations)
    Historical: [4, 5, 6, 3, 5]
    
    Posterior Predictive (WIDE uncertainty):
    
         ║       ╱‾‾‾‾‾‾‾╲
         ║      ╱         ╲
         ║     ╱           ╲
         ║   ─╱             ╲────
         ║  0   2   4   6   8   10  12
         ║      ├──────────┤
         ║       90% CI
         ║     [1.5, 9.5]
    
    NEW OBSERVATION: y = 10
    → Just outside CI → Score = 4.1 → 🟡 MODERATELY UNUSUAL


KEY INSIGHT: Same observation (y=10), different scores!
• User A: Confident baseline → Detects small deviations
• User B: Uncertain baseline → More conservative


┌─────────────────────────────────────────────────────────────────────────┐
│                 UNCERTAINTY PROPAGATION IN ACTION                        │
└─────────────────────────────────────────────────────────────────────────┘

SCENARIO: Entity with evolving behavior

    Time Period 1 (Days 1-7):
    Events: [2,3,2,3,2]
    
    ┌─────────────────────────┐
    │    Posterior            │
    │      ╱‾╲               │
    │     ╱   ╲              │
    │   ─╱     ╲─            │
    │   0  2  4  6            │
    │     ├─┤                │
    │   90% CI               │
    │  [1.5, 4.2]            │
    └─────────────────────────┘
    
    NEW: y=8 → Score = 6.8 → 🔴 FLAG
    

    Time Period 2 (Days 8-21, more data):
    Events: [2,3,2,3,2,8,7,9,8,7,8,9]
    
    ┌─────────────────────────┐
    │    Updated Posterior    │
    │              ╱‾╲       │
    │             ╱   ╲      │
    │   ─────────╱     ╲─    │
    │   0  2  4  6  8  10     │
    │            ├──┤         │
    │          90% CI         │
    │        [4.8, 9.1]       │
    └─────────────────────────┘
    
    NEW: y=8 → Score = 2.1 → 🟢 NORMAL
    
    → Uncertainty ADAPTS as we gather more evidence!
    → Baseline shifted due to behavioral change
    → Bayesian model tracks this naturally


┌─────────────────────────────────────────────────────────────────────────┐
│                    THRESHOLD TUNING: OLD WAY vs NEW WAY                  │
└─────────────────────────────────────────────────────────────────────────┘

OLD WAY (Classical):
    
    "Let's try threshold = μ + 2σ"
     → Too many false positives
    
    "OK, try threshold = μ + 3σ"
     → Missing attacks
    
    "Let's do μ + 2.5σ"
     → Still not great, and arbitrary!
    
    ❌ Manual, iterative, domain-specific


NEW WAY (Bayesian):
    
    P(y | data) → anomaly_score(y) → Rank by score
    
    Pick operational threshold based on capacity:
    • "Investigate top 50 alerts/day" → Take top 50 by score
    • "Alert if score > 6" → Calibrated to probability ~0.0025
    
    ✅ Automatic, principled, interpretable


┌─────────────────────────────────────────────────────────────────────────┐
│              VISUAL: CLASSICAL vs BAYESIAN DECISION MAKING               │
└─────────────────────────────────────────────────────────────────────────┘

CLASSICAL: Binary decision at fixed threshold
    
    Confidence?
        │
    100%│ ████████████████│
        │                 │
     50%│                 │
        │                 │
      0%│                 │
        └─────────────────┼──────────→ Observed Value
               Threshold  │
                       
    "It's either OK or NOT OK, no middle ground"


BAYESIAN: Continuous confidence based on posterior

    Confidence?
        │            ╱‾╲
    100%│           ╱   ╲
        │          ╱     ╲___
     50%│         ╱          ╲___
        │   _____╱               ╲_____
      0%│                            
        └────────────────────────────────→ Observed Value
               0    2    4    6    8   10
        
    Score:  1.5  2.1  2.8  4.2  6.7  9.1
    
    "Observation at 6 is moderately unusual (score=4.2)"
    "Observation at 10 is highly unusual (score=9.1)"


┌─────────────────────────────────────────────────────────────────────────┐
│                      REAL-WORLD IMPACT                                   │
└─────────────────────────────────────────────────────────────────────────┘

CASE 1: SOC Analyst Workflow

Classical Threshold:
    • 500 alerts/day
    • 450 false positives (90% FP rate)
    • Analyst fatigue → Miss real attacks
    
Bayesian Scoring:
    • Rank by score, investigate top 50
    • 42 true positives, 8 false positives (16% FP rate)
    • Focus analyst time on real threats
    

CASE 2: Automated Response

Classical:
    • Fixed threshold triggers auto-block
    • One false positive → Legitimate user locked out
    • Customer complains, revenue loss
    
Bayesian:
    • Tiered response based on score:
      - Score 6-7: Log + notify
      - Score 7-8: Rate limit
      - Score 8+: Auto-block
    • Lower scores → Less aggressive action
    • Fewer customer complaints


┌─────────────────────────────────────────────────────────────────────────┐
│                           KEY TAKEAWAYS                                  │
└─────────────────────────────────────────────────────────────────────────┘

1. UNCERTAINTY IS A FEATURE, NOT A BUG
   → Classical: Ignores uncertainty → Overconfident
   → Bayesian: Quantifies uncertainty → Calibrated

2. ONE SIZE DOESN'T FIT ALL
   → Classical: Same threshold for all entities
   → Bayesian: Entity-specific baselines + uncertainty

3. INTERPRETABILITY MATTERS
   → Classical: "It exceeded the threshold" (arbitrary)
   → Bayesian: "This has 0.2% probability" (meaningful)

4. ADAPTATION IS AUTOMATIC
   → Classical: Manual retuning when behavior changes
   → Bayesian: Posterior updates with new data

5. OPERATIONAL FLEXIBILITY
   → Classical: Fixed threshold
   → Bayesian: Score-based ranking → Choose top-K or threshold
