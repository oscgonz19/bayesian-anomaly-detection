# Dataset Card: BSAD Datasets

## Overview

BSAD supports three types of datasets:

1. **Synthetic Data** - Generated for demos and testing
2. **UNSW-NB15** - Real network intrusion dataset
3. **CSE-CIC-IDS2018** - Real intrusion detection dataset

---

## 1. Synthetic Data

### Description
Programmatically generated security event logs with realistic attack patterns.

### Features Used
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Event timestamp |
| `user_id` | string | Entity identifier (e.g., `user_0042`) |
| `ip_address` | string | Source IP address |
| `endpoint` | string | API endpoint (e.g., `/api/v1/login`) |
| `status_code` | int | HTTP status code (200, 401, 403, 500) |
| `location` | string | Geographic location |
| `device_fingerprint` | string | Device identifier |
| `bytes_transferred` | int | Bytes in response |
| `is_attack` | bool | Ground truth label |
| `attack_type` | string | Type of attack or "none" |

### Aggregated Features (modeling_df)
| Column | Description |
|--------|-------------|
| `event_count` | Events per entity-window (TARGET) |
| `unique_ips` | Distinct IPs in window |
| `unique_endpoints` | Distinct endpoints |
| `unique_devices` | Distinct device fingerprints |
| `unique_locations` | Distinct geolocations |
| `bytes_total` | Total bytes transferred |
| `failed_count` | Events with error status codes |

### Attack Types
- **brute_force**: 50-200 login attempts from single IP
- **credential_stuffing**: Multiple users targeted from same IP
- **geo_anomaly**: Access from suspicious locations (TOR, VPN, sanctioned countries)
- **device_anomaly**: Many new device fingerprints

### Limitations
- Synthetic patterns may not capture all real-world attack complexity
- Attack injection is random, may create unrealistic scenarios
- No temporal seasonality beyond weekend effects

---

## 2. UNSW-NB15

### Source
[UNSW Canberra Cyber](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Description
Network traffic dataset with modern attack types, generated using the IXIA PerfectStorm tool.

### Features Used by BSAD
| Original Column | BSAD Usage | Notes |
|-----------------|------------|-------|
| `srcip` | Entity ID | Source IP as entity |
| `spkts` | Count target | Source packets (count data) |
| `dpkts` | Feature | Destination packets |
| `sbytes` | Feature | Source bytes |
| `dbytes` | Feature | Destination bytes |
| `attack_cat` | Label | Attack category |
| `label` | Binary label | 0=normal, 1=attack |

### Attack Categories
- Normal (benign traffic)
- Generic, Exploits, Fuzzers, DoS, Reconnaissance
- Analysis, Backdoor, Shellcode, Worms

### Preprocessing
1. Filter to rows with valid `srcip`
2. Aggregate by `srcip` (entity) and time window
3. Target variable: `spkts` (source packet count)
4. Remove rows with zero counts

### Limitations
- High attack rate (~44%) in original data - NOT rare-event regime
- Need subsampling to create realistic <5% attack rate
- No true timestamps, only relative order
- May not represent modern attack patterns (2015 data)

---

## 3. CSE-CIC-IDS2018

### Source
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html)

### Description
Collaborative intrusion detection dataset with labeled network flows.

### Features Used by BSAD
| Column | BSAD Usage | Notes |
|--------|------------|-------|
| `Src IP` | Entity ID | Source IP as entity |
| `Fwd Pkts` | Count target | Forward packets |
| `Flow Pkts/s` | Feature | Packets per second |
| `Flow Byts/s` | Feature | Bytes per second |
| `Label` | Label | Attack type or "Benign" |

### Attack Types
- Benign
- DoS attacks-GoldenEye, DoS attacks-Slowloris, DoS attacks-SlowHTTPTest
- DDoS attacks-LOIC-HTTP, DDoS attack-HOIC
- Brute Force (SSH, FTP)
- Infiltration
- Bot

### Preprocessing
1. Handle missing values (`Infinity` → NaN → median impute)
2. Aggregate by source IP and time window
3. Target: forward packet count
4. Binary label: Benign=0, any attack=1

### Limitations
- Large file sizes (>100MB per day)
- Some flows have missing/infinite values
- Attack distribution varies by day
- Synthetic traffic may not represent real environments

---

## General Data Requirements for BSAD

### Necessary Structure
- **Count data**: Non-negative integers (packets, events, requests)
- **Entity structure**: Repeated observations per entity
- **Rare events**: Attack rate ideally <5% for BSAD advantage
- **Overdispersion**: Variance >> Mean (typical in security data)

### What BSAD Does NOT Handle Well
- Continuous features without discretization
- Very high attack rates (>10%)
- Single-observation entities
- Real-time requirements (<100ms)
- Content-based attacks (SQLi, XSS patterns)

---

## Leakage Checklist

### ✅ No Leakage Present
- [ ] Test data is temporally after training data
- [ ] Entity baselines computed only on training data
- [ ] No future information in features
- [ ] Ground truth labels not used in feature engineering
- [ ] Model hyperparameters not tuned on test set

### Validation
Run temporal split validation:
```bash
python scripts/benchmark.py --output outputs/benchmark
```

This ensures train/test split respects temporal ordering.

---

## Reproducibility

### Seeds
All experiments use fixed seeds:
- Data generation: `Settings.random_seed = 42`
- MCMC sampling: Derived from main seed
- Train/test splits: Same seed

### Versioning
- Dataset processing code in `src/bsad/unsw_adapter.py`
- Settings in `src/bsad/config.py`
- Benchmark protocol in `scripts/benchmark.py`
