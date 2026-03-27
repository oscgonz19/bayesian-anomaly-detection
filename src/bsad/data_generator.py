"""
Synthetic security event log generator.

Generates entity-structured count data with injected attack patterns
suitable for training the hierarchical Negative Binomial anomaly detector.

WHAT THIS MODULE MODELS
-----------------------
- Entity-specific daily event counts drawn from Poisson processes
  with user-level rate heterogeneity (lognormal mixing)
- Four attack types: brute_force, credential_stuffing, geo_anomaly, device_anomaly

WHAT IT DOES NOT MODEL
-----------------------
- geo_anomaly and device_anomaly are injected as metadata; they only
  raise event_count if the attacker generates enough events to perturb
  the daily total. The hierarchical NB model detects them only through
  count elevation, not through location/device features.
- Continuous covariate structure (bytes, endpoint diversity) is generated
  but is not modeled by the NB likelihood—it is contextual metadata only.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

AttackType = Literal["brute_force", "credential_stuffing", "geo_anomaly", "device_anomaly", "none"]


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation.

    Separated from model and feature config so that generation parameters
    can be varied independently in benchmarks and robustness studies.
    """

    # Entity / time dimensions
    n_users: int = 200
    n_days: int = 30
    n_ips: int = 100
    n_endpoints: int = 50

    # Baseline event rate (lognormal mixing across users)
    events_per_user_day_mean: float = 5.0
    events_per_user_day_std: float = 3.0  # unused in generation (lognormal sigma=0.5), kept for docs

    # Attack prevalence
    attack_rate: float = 0.02

    # Attack intensity parameters (min, max range)
    brute_force_multiplier: tuple[int, int] = (50, 200)
    credential_stuffing_users: tuple[int, int] = (10, 30)
    credential_stuffing_events_per_user: tuple[int, int] = (3, 10)
    device_anomaly_new_devices: tuple[int, int] = (3, 8)

    # Reproducibility
    random_seed: int = 42


def generate_synthetic_data(config: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic security event logs with injected attack patterns.

    Args:
        config: Generation parameters.

    Returns:
        events_df: Row-per-event table with columns:
            timestamp, user_id, ip_address, endpoint, status_code,
            location, device_fingerprint, bytes_transferred,
            is_attack (bool), attack_type (str).
        attacks_df: Ground-truth attack metadata (one row per injected attack).
    """
    rng = np.random.default_rng(config.random_seed)
    events_df = generate_baseline_events(config, rng)
    all_attack_records: list[dict] = []

    n_entity_windows = config.n_users * config.n_days
    n_attacks = max(1, int(n_entity_windows * config.attack_rate))

    attack_types: list[AttackType] = ["brute_force", "credential_stuffing", "geo_anomaly", "device_anomaly"]
    attack_distribution = rng.choice(attack_types, size=n_attacks)

    for attack_type in attack_distribution:
        if attack_type == "brute_force":
            events_df, records = inject_brute_force_attack(events_df, config, rng)
        elif attack_type == "credential_stuffing":
            events_df, records = _inject_credential_stuffing(events_df, config, rng)
        elif attack_type == "geo_anomaly":
            events_df, records = _inject_geo_anomaly(events_df, config, rng)
        else:
            events_df, records = _inject_device_anomaly(events_df, config, rng)
        all_attack_records.extend(records)

    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    attacks_df = pd.DataFrame(all_attack_records)

    return events_df, attacks_df


def generate_baseline_events(config: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate benign baseline events for all users across all days.

    Each user has a stable Poisson rate drawn from a lognormal distribution,
    modelling the realistic heterogeneity in user activity levels.
    Weekend activity is dampened by 70%.
    """
    users = [f"user_{i:04d}" for i in range(config.n_users)]
    ips = [f"ip_{i:04d}" for i in range(config.n_ips)]
    endpoints = [f"/api/v1/{e}" for e in ["login", "logout", "data", "users", "admin", "reports"]]
    endpoints += [f"/api/v1/resource_{i}" for i in range(config.n_endpoints - len(endpoints))]

    user_rates = rng.lognormal(
        mean=np.log(config.events_per_user_day_mean), sigma=0.5, size=config.n_users
    )
    user_primary_ip = {u: rng.choice(ips) for u in users}
    locations = ["US-East", "US-West", "EU-West", "EU-Central", "APAC"]
    user_primary_location = {u: rng.choice(locations) for u in users}
    user_devices = {
        u: [_fingerprint(u, i, rng) for i in range(rng.integers(1, 4))]
        for u in users
    }

    start_date = datetime(2024, 1, 1)
    events = []

    for day_offset in range(config.n_days):
        current_date = start_date + timedelta(days=day_offset)
        dow_multiplier = 1.0 if current_date.weekday() < 5 else 0.3

        for user_idx, user in enumerate(users):
            n_events = rng.poisson(user_rates[user_idx] * dow_multiplier)
            for _ in range(n_events):
                hour = int(rng.beta(2, 2) * 14 + 7) % 24
                timestamp = current_date.replace(
                    hour=hour,
                    minute=rng.integers(0, 60),
                    second=rng.integers(0, 60),
                )
                ip = user_primary_ip[user] if rng.random() < 0.9 else rng.choice(ips)
                location = (
                    user_primary_location[user] if rng.random() < 0.95 else rng.choice(locations)
                )
                device = rng.choice(user_devices[user])
                endpoint_weights = [0.3] + [0.7 / (len(endpoints) - 1)] * (len(endpoints) - 1)
                endpoint = rng.choice(endpoints, p=endpoint_weights)
                status = rng.choice(
                    [200, 201, 400, 401, 403, 500], p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01]
                )
                events.append({
                    "timestamp": timestamp,
                    "user_id": user,
                    "ip_address": ip,
                    "endpoint": endpoint,
                    "status_code": status,
                    "location": location,
                    "device_fingerprint": device,
                    "bytes_transferred": int(rng.lognormal(6, 1)),
                    "is_attack": False,
                    "attack_type": "none",
                })

    return pd.DataFrame(events)


def inject_brute_force_attack(
    df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject a brute-force login attack targeting a single user.

    Detectable by the NB model: large burst of events on one entity-day.
    """
    users = df["user_id"].unique()
    target_user = rng.choice(users)
    attack_ip = f"attack_ip_{rng.integers(1000, 9999)}"
    n_events = rng.integers(*config.brute_force_multiplier)
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))
    attack_hour = rng.integers(0, 24)

    attack_events = []
    for i in range(n_events):
        timestamp = attack_date.replace(
            hour=attack_hour,
            minute=rng.integers(0, 60),
            second=rng.integers(0, 60),
        )
        status = 200 if i == n_events - 1 else rng.choice([401, 403], p=[0.9, 0.1])
        attack_events.append({
            "timestamp": timestamp,
            "user_id": target_user,
            "ip_address": attack_ip,
            "endpoint": "/api/v1/login",
            "status_code": status,
            "location": rng.choice(["Unknown", "TOR", "VPN"]),
            "device_fingerprint": _fingerprint("attacker", 0, rng),
            "bytes_transferred": int(rng.lognormal(5, 0.5)),
            "is_attack": True,
            "attack_type": "brute_force",
        })

    record = {
        "attack_type": "brute_force",
        "target_entity": target_user,
        "source_ip": attack_ip,
        "window_start": attack_date.replace(hour=attack_hour),
        "n_events": n_events,
    }
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


# ---------------------------------------------------------------------------
# Internal injectors (not part of the public test API)
# ---------------------------------------------------------------------------

def _inject_credential_stuffing(
    df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject credential stuffing: one attacker IP hits many users.

    May be detectable per-user if event count elevates enough; primarily
    a multi-entity pattern that the current single-entity model cannot
    correlate across users. Detection is per-entity only.
    """
    users = df["user_id"].unique()
    attack_ip = f"attack_ip_{rng.integers(1000, 9999)}"
    n_target_users = rng.integers(*config.credential_stuffing_users)
    target_users = rng.choice(users, size=min(n_target_users, len(users)), replace=False)
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))

    attack_events = []
    for target_user in target_users:
        n_events = rng.integers(*config.credential_stuffing_events_per_user)
        for _ in range(n_events):
            timestamp = attack_date.replace(
                hour=rng.integers(0, 24),
                minute=rng.integers(0, 60),
                second=rng.integers(0, 60),
            )
            attack_events.append({
                "timestamp": timestamp,
                "user_id": target_user,
                "ip_address": attack_ip,
                "endpoint": "/api/v1/login",
                "status_code": rng.choice([401, 200], p=[0.85, 0.15]),
                "location": rng.choice(["Unknown", "Proxy"]),
                "device_fingerprint": _fingerprint("stuffing", 0, rng),
                "bytes_transferred": int(rng.lognormal(5, 0.5)),
                "is_attack": True,
                "attack_type": "credential_stuffing",
            })

    record = {
        "attack_type": "credential_stuffing",
        "target_entity": list(target_users),
        "source_ip": attack_ip,
        "window_start": attack_date,
        "n_events": len(attack_events),
    }
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


def _inject_geo_anomaly(
    df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject access from anomalous geolocations.

    NOTE: The hierarchical NB model does NOT model location features.
    This attack type is detectable only if the associated events cause a
    meaningful count elevation on the target entity-day. Low-volume
    geo_anomalies will NOT be reliably detected by count-based scoring.
    """
    users = df["user_id"].unique()
    target_user = rng.choice(users)
    anomalous_locations = ["North-Korea", "Iran", "Unknown-VPN", "TOR-Exit", "Suspicious-Proxy"]
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))
    n_events = rng.integers(5, 20)

    attack_events = []
    for i in range(n_events):
        timestamp = attack_date.replace(
            hour=rng.integers(0, 24),
            minute=rng.integers(0, 60),
            second=rng.integers(0, 60),
        )
        attack_events.append({
            "timestamp": timestamp,
            "user_id": target_user,
            "ip_address": f"geo_attack_ip_{rng.integers(1000, 9999)}",
            "endpoint": rng.choice(["/api/v1/data", "/api/v1/admin", "/api/v1/reports"]),
            "status_code": 200,
            "location": rng.choice(anomalous_locations),
            "device_fingerprint": _fingerprint("geo_attacker", i, rng),
            "bytes_transferred": int(rng.lognormal(8, 1)),
            "is_attack": True,
            "attack_type": "geo_anomaly",
        })

    record = {
        "attack_type": "geo_anomaly",
        "target_entity": target_user,
        "window_start": attack_date,
        "n_events": n_events,
    }
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


def _inject_device_anomaly(
    df: pd.DataFrame,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Inject access from new/unseen devices.

    NOTE: The hierarchical NB model does NOT model device features.
    Detection is count-based only; new-device signals are not explicitly
    modeled. This attack type serves as a realistic false-negative
    case for the count model in benchmarks.
    """
    users = df["user_id"].unique()
    target_user = rng.choice(users)
    n_new_devices = rng.integers(*config.device_anomaly_new_devices)
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))

    attack_events = []
    for device_idx in range(n_new_devices):
        n_events = rng.integers(2, 8)
        new_device = _fingerprint(f"new_device_{target_user}", device_idx, rng)
        for _ in range(n_events):
            timestamp = attack_date.replace(
                hour=rng.integers(0, 24),
                minute=rng.integers(0, 60),
                second=rng.integers(0, 60),
            )
            attack_events.append({
                "timestamp": timestamp,
                "user_id": target_user,
                "ip_address": f"device_ip_{rng.integers(1000, 9999)}",
                "endpoint": rng.choice(["/api/v1/login", "/api/v1/data"]),
                "status_code": 200,
                "location": rng.choice(["US-East", "US-West", "EU-West"]),
                "device_fingerprint": new_device,
                "bytes_transferred": int(rng.lognormal(6, 1)),
                "is_attack": True,
                "attack_type": "device_anomaly",
            })

    record = {
        "attack_type": "device_anomaly",
        "target_entity": target_user,
        "window_start": attack_date,
        "n_events": len(attack_events),
        "n_new_devices": n_new_devices,
    }
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


def _fingerprint(user: str, idx: int, rng: np.random.Generator) -> str:
    seed_str = f"{user}_{idx}_{rng.integers(0, 10000)}"
    return hashlib.md5(seed_str.encode()).hexdigest()[:16]
