# UNSW-NB15 Dataset Description

## Overview

The **UNSW-NB15 dataset** is a network intrusion detection dataset created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS) at the University of New South Wales. It contains modern network traffic captures with both normal activities and nine families of attack behaviors.

**Key Reference**: Moustafa, N. & Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)*. Military Communications and Information Systems Conference (MilCIS).

---

## Dataset Composition

| Component | Training Set | Testing Set | Total |
|-----------|--------------|-------------|-------|
| Records | 175,341 | 82,332 | 257,673 |
| Features | 49 | 49 | 49 |
| Normal | 56,000 (37%) | 37,000 (45%) | 93,000 |
| Attacks | 119,341 (63%) | 45,332 (55%) | 164,673 |

**Critical Observation**: The original dataset has a 64% attack rate, making it a **classification problem** rather than a true anomaly detection scenario. In production security environments, attack rates are typically <1-5%.

---

## What is a Network Flow?

**UNSW-NB15 is a collection of network flows, not individual packets.**

### Definition

A **network flow** is an aggregated record of communication between two endpoints over a short time period. Each flow represents a distinct conversation characterized by:

- **Who**: Source and destination IP addresses
- **How**: Protocol and service used
- **Duration**: Time window of the communication
- **Behavior**: Packet counts, byte transfers, timing patterns

### Conceptual Example

```
Flow 1: Machine A → Machine B via UDP/DNS for 0.5 seconds
        (3 packets, 180 bytes, normal DNS query)

Flow 2: Machine C → Machine D via TCP/HTTP for 2.3 seconds
        (1,247 packets, 850 KB, potential data exfiltration)

Flow 3: Machine E → Machine F via TCP/SMTP for 0.8 seconds
        (127 packets, 45 KB, normal email transmission)
```

**Each row = one complete communication story.**

---

## Dataset Structure

Each flow record contains four logical blocks of information:

### A. Communication Identity (Who talks to whom)

| Feature | Description | Example Values |
|---------|-------------|----------------|
| `srcip` | Source IP address | 192.168.1.10 |
| `dstip` | Destination IP address | 8.8.8.8 |
| `sport` | Source port | 54321 |
| `dport` | Destination port | 53 |
| `proto` | Protocol | TCP, UDP, ARP, ICMP |

### B. Traffic Type (What kind of conversation)

| Feature | Description | Example Values |
|---------|-------------|----------------|
| `proto` | Network protocol | tcp, udp, arp, ospf |
| `service` | Application service | http, dns, smtp, ssh, ftp, `-` |
| `state` | Connection state | FIN, INT, CON, REQ |

**Key Insight**: The combination `proto_service` defines the **traffic context**, which is critical for determining what constitutes "normal" behavior.

### C. Behavioral Characteristics (How intense was it)

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `spkts` | Source packets | Count variable (key for BSAD) |
| `dpkts` | Destination packets | Count variable |
| `sbytes` | Source bytes | Size of data sent |
| `dbytes` | Destination bytes | Size of data received |
| `dur` | Duration (seconds) | How long the flow lasted |
| `rate` | Packets/second | Communication intensity |
| `sload`, `dload` | Source/dest load | Throughput metrics |
| `sintpkt`, `dintpkt` | Inter-packet time | Timing patterns |

### D. Security Labels (Ground truth)

| Feature | Description | Values |
|---------|-------------|--------|
| `label` | Binary attack indicator | 0 = Normal, 1 = Attack |
| `attack_cat` | Attack category | Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms |

**Note**: These labels are ground truth for evaluation, not available in production deployments.

---

## The Heterogeneous Nature of Network Traffic

### The Context Dependency Problem

**UNSW-NB15 is not just a table of features; it is a collection of heterogeneous communication processes, each with its own notion of normality.**

#### Example: The Meaning of "50 packets"

| Traffic Type | 50 Packets Means |
|--------------|------------------|
| **ARP** | Normal (broadcast resolution) |
| **DNS** | Anomalous (DNS should be 2-3 packets) |
| **HTTP** | Irrelevant (typical web page loads 100s) |
| **SSH** | Depends (interactive vs file transfer) |

**Key Insight**: A numeric value has no intrinsic meaning without its traffic context.

### Entity Structure is Implicit

The dataset contains implicit **entity-level structure** through traffic type:

```
proto_service = Entity Type
├── udp_dns      → DNS queries (fast, small, predictable)
├── tcp_http     → Web traffic (variable, large, bursty)
├── tcp_smtp     → Email (medium, structured)
├── arp_-        → Address resolution (tiny, broadcast)
└── tcp_ssh      → Secure shell (encrypted, variable)
```

Each entity has:
- **Different baseline behavior**
- **Different variance patterns**
- **Different attack signatures**

---

## Two Valid Perspectives on UNSW-NB15

The dataset can be used in two fundamentally different ways, depending on the research question:

### Perspective 1: Multivariate Feature Space (Classical)

**View**: Each flow is an independent feature vector.

**Approach**:
- Use all 49 features as inputs
- Ignore traffic type structure
- Treat each row as exchangeable
- Apply global decision boundary

**Best Methods**:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor

**Results**: Classical methods excel (PR-AUC ~0.05-0.15 on multivariate features)

**Question Answered**: "Is this flow's feature profile unusual compared to all other flows?"

---

### Perspective 2: Count Processes with Entity Structure (Bayesian)

**View**: Flows are realizations of heterogeneous count processes.

**Approach**:
- Select count variable (e.g., `spkts`)
- Define entity (e.g., `proto_service`)
- Learn entity-specific baselines
- Detect deviations within context

**Best Methods**:
- BSAD (Hierarchical Negative Binomial)
- Bayesian methods with partial pooling

**Results**: BSAD excels on synthetic count data (+30 PR-AUC points)

**Question Answered**: "Is this count unusual for *this type* of traffic?"

---

## Data Transformation for BSAD

To apply BSAD to UNSW-NB15, the following transformations are required:

### 1. Entity Definition

```python
entity = proto + '_' + service
# Examples: 'udp_dns', 'tcp_http', 'arp_-'
```

### 2. Count Variable Selection

```python
count_variable = 'spkts'  # Source packets
# Alternatives: 'dpkts', 'sbytes', 'dbytes'
```

### 3. Rare-Attack Regime Creation

**Problem**: Original 64% attack rate breaks anomaly detection assumptions.

**Solution**: Subsample attacks to create realistic regimes:

```python
# Keep all normal flows (93,000)
# Subsample attacks to achieve target rates

unsw_nb15_rare_attack_1pct.parquet  →  1% attacks (939 attacks)
unsw_nb15_rare_attack_2pct.parquet  →  2% attacks (1,897 attacks)
unsw_nb15_rare_attack_5pct.parquet  →  5% attacks (4,894 attacks)
```

**Rationale**: Real-world SOC environments see <1% attack rates, making rare-attack regimes more realistic for anomaly detection research.

### 4. Window Aggregation (Optional)

```python
# Aggregate consecutive flows into time windows
# This creates entity-window observations

df_windowed = aggregate_flows_to_windows(
    df,
    entity_col='proto_service',
    count_col='spkts',
    window_size=200  # flows per window
)
```

**Result**: Each observation becomes:
- Entity: `proto_service` combination
- Count: Sum of `spkts` in window
- Label: Window has attack if any flow is malicious

---

## Statistical Properties

### Overdispersion

UNSW-NB15 count features exhibit strong **overdispersion** (Variance >> Mean), a key requirement for Negative Binomial modeling:

| Feature | Mean | Variance | Var/Mean Ratio |
|---------|------|----------|----------------|
| `spkts` | 19.7 | 18,175.2 | **923.0** |
| `dpkts` | 15.3 | 9,847.1 | **643.6** |
| `sbytes` | 3,684.5 | 127M | **34,481.2** |
| `dbytes` | 10,890.3 | 2.1B | **192,863.8** |

**Implication**: Poisson models are inadequate; Negative Binomial is the appropriate distribution.

### Entity Variation

Traffic types show high heterogeneity in baseline rates:

```
Mean spkts by entity:
  tcp_http:    85.4 packets/flow
  udp_dns:      2.1 packets/flow
  tcp_smtp:    42.7 packets/flow
  arp_-:        1.8 packets/flow
```

**Implication**: Hierarchical models with entity-specific parameters are necessary.

---

## Attack Categories

The dataset includes 9 attack families:

| Attack Type | Description | Prevalence |
|-------------|-------------|------------|
| **Generic** | Techniques against block ciphers, hash functions | 40,000 |
| **Exploits** | Known vulnerability exploitation | 33,393 |
| **Fuzzers** | Random data injection to cause crashes | 18,184 |
| **DoS** | Denial of Service attacks | 12,264 |
| **Reconnaissance** | Network scanning and probing | 10,491 |
| **Analysis** | Port scanning, spam, HTML files | 2,000 |
| **Backdoor** | Remote access trojans | 1,746 |
| **Shellcode** | Code injection attacks | 1,133 |
| **Worms** | Self-replicating malware | 130 |

**Note**: Attack distribution is highly imbalanced, with Generic and Exploits dominating.

---

## Key Takeaways

1. **UNSW-NB15 records network flows, not packets**
   - Each row = one complete communication story
   - Flows have identity, type, behavior, and labels

2. **Traffic context is critical**
   - Numeric values have no meaning without traffic type
   - "Normal" is entity-specific, not global

3. **Two valid analysis paradigms**
   - Multivariate: Ignore structure → Classical methods
   - Count-based: Exploit structure → Bayesian methods

4. **Original dataset is classification, not anomaly detection**
   - 64% attack rate breaks anomaly assumptions
   - Rare-attack regimes (1-5%) are needed for realistic evaluation

5. **Strong statistical properties for count modeling**
   - Extreme overdispersion (Var/Mean up to 923x)
   - Entity heterogeneity requires hierarchical models
   - Negative Binomial is the natural distribution

---

## Recommended Usage

### For BSAD (Count-Based Anomaly Detection)

✅ **Use when**:
- Research question is: "Is this count unusual for this traffic type?"
- Need entity-specific baselines
- Want uncertainty quantification
- Have rare-attack regime data

✅ **Configuration**:
```python
entity = 'proto_service'
count = 'spkts'
attack_rate = 0.01  # Use rare-attack resampled data
```

### For Classical Methods (Multivariate Anomaly Detection)

✅ **Use when**:
- Research question is: "Is this flow's profile unusual globally?"
- Want fast detection without training
- Multivariate feature patterns matter
- Entity structure is unknown or irrelevant

✅ **Configuration**:
```python
features = ['sbytes', 'dbytes', 'spkts', 'dpkts',
            'dur', 'rate', 'sload', 'dload']
method = IsolationForest()
```

---

## Limitations and Considerations

1. **Temporal dependencies ignored**
   - Flows are treated as independent
   - Sequential attack patterns not captured

2. **No entity metadata**
   - No information about specific IPs or users
   - Limited contextual information

3. **Simulated traffic**
   - Generated in controlled environment
   - May not reflect production network diversity

4. **Attack label granularity**
   - Binary label (0/1) at flow level
   - Attack category distribution is imbalanced

5. **High baseline attack rate**
   - Original 64% attacks unrealistic for production
   - Requires resampling for anomaly detection research

---

## Citation

If you use UNSW-NB15 in your research, please cite:

```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 Military Communications and Information Systems Conference (MilCIS)},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

---

## Summary Statement

> **UNSW-NB15 is not merely a table of features; it is a collection of heterogeneous communication processes, each with its own notion of normality. The dataset's value lies not just in what it contains, but in how we choose to model its inherent structure.**

Understanding this distinction is critical for choosing the appropriate anomaly detection methodology and interpreting results meaningfully.
