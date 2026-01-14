"""
Pipeline steps as pure functions.

Each function does ONE thing. Steps do NOT call each other.
The Pipeline class orchestrates the flow.
"""

import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from bsad.config import Settings

# Type alias for attack types
AttackType = Literal["brute_force", "credential_stuffing", "geo_anomaly", "device_anomaly", "none"]


# =============================================================================
# Step 1: Generate Synthetic Data
# =============================================================================


def generate_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic security event logs with attack patterns.

    Returns:
        events_df: All events (benign + attacks)
        attacks_df: Attack metadata for ground truth
    """
    rng = np.random.default_rng(settings.random_seed)

    # Generate baseline events
    events_df = _generate_baseline_events(settings, rng)
    all_attack_records = []

    # Calculate number of attacks
    n_entity_windows = settings.n_entities * settings.n_days
    n_attacks = max(1, int(n_entity_windows * settings.attack_rate))

    # Distribute attacks across types
    attack_types = ["brute_force", "credential_stuffing", "geo_anomaly", "device_anomaly"]
    attack_distribution = rng.choice(attack_types, size=n_attacks)

    for attack_type in attack_distribution:
        if attack_type == "brute_force":
            events_df, records = _inject_brute_force(events_df, settings, rng)
        elif attack_type == "credential_stuffing":
            events_df, records = _inject_credential_stuffing(events_df, settings, rng)
        elif attack_type == "geo_anomaly":
            events_df, records = _inject_geo_anomaly(events_df, settings, rng)
        else:
            events_df, records = _inject_device_anomaly(events_df, settings, rng)
        all_attack_records.extend(records)

    # Sort by timestamp
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    attacks_df = pd.DataFrame(all_attack_records)

    return events_df, attacks_df


def _generate_baseline_events(settings: Settings, rng: np.random.Generator) -> pd.DataFrame:
    """Generate benign baseline events."""
    users = [f"user_{i:04d}" for i in range(settings.n_entities)]
    ips = [f"ip_{i:04d}" for i in range(settings.n_ips)]
    endpoints = [f"/api/v1/{e}" for e in ["login", "logout", "data", "users", "admin", "reports"]]
    endpoints += [f"/api/v1/resource_{i}" for i in range(settings.n_endpoints - len(endpoints))]

    # User-specific rates
    user_rates = rng.lognormal(mean=np.log(settings.events_per_user_day_mean), sigma=0.5, size=settings.n_entities)
    user_primary_ip = {u: rng.choice(ips) for u in users}
    locations = ["US-East", "US-West", "EU-West", "EU-Central", "APAC"]
    user_primary_location = {u: rng.choice(locations) for u in users}
    user_devices = {u: [_fingerprint(u, i, rng) for i in range(rng.integers(1, 4))] for u in users}

    start_date = datetime(2024, 1, 1)
    events = []

    for day_offset in range(settings.n_days):
        current_date = start_date + timedelta(days=day_offset)
        dow_multiplier = 1.0 if current_date.weekday() < 5 else 0.3

        for user_idx, user in enumerate(users):
            n_events = rng.poisson(user_rates[user_idx] * dow_multiplier)

            for _ in range(n_events):
                hour = int(rng.beta(2, 2) * 14 + 7) % 24
                timestamp = current_date.replace(hour=hour, minute=rng.integers(0, 60), second=rng.integers(0, 60))
                ip = user_primary_ip[user] if rng.random() < 0.9 else rng.choice(ips)
                location = user_primary_location[user] if rng.random() < 0.95 else rng.choice(locations)
                device = rng.choice(user_devices[user])
                endpoint_weights = [0.3] + [0.7 / (len(endpoints) - 1)] * (len(endpoints) - 1)
                endpoint = rng.choice(endpoints, p=endpoint_weights)
                status = rng.choice([200, 201, 400, 401, 403, 500], p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01])

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


def _fingerprint(user: str, idx: int, rng: np.random.Generator) -> str:
    seed_str = f"{user}_{idx}_{rng.integers(0, 10000)}"
    return hashlib.md5(seed_str.encode()).hexdigest()[:16]


def _inject_brute_force(df: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> tuple[pd.DataFrame, list]:
    users = df["user_id"].unique()
    target_user = rng.choice(users)
    attack_ip = f"attack_ip_{rng.integers(1000, 9999)}"
    n_events = rng.integers(*settings.brute_force_multiplier)
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))
    attack_hour = rng.integers(0, 24)

    attack_events = []
    for i in range(n_events):
        timestamp = attack_date.replace(hour=attack_hour, minute=rng.integers(0, 60), second=rng.integers(0, 60))
        status = 200 if i == n_events - 1 else rng.choice([401, 403], p=[0.9, 0.1])
        attack_events.append({
            "timestamp": timestamp, "user_id": target_user, "ip_address": attack_ip,
            "endpoint": "/api/v1/login", "status_code": status,
            "location": rng.choice(["Unknown", "TOR", "VPN"]),
            "device_fingerprint": _fingerprint("attacker", 0, rng),
            "bytes_transferred": int(rng.lognormal(5, 0.5)),
            "is_attack": True, "attack_type": "brute_force",
        })

    record = {"attack_type": "brute_force", "target_entity": target_user, "source_ip": attack_ip,
              "window_start": attack_date.replace(hour=attack_hour), "n_events": n_events}
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


def _inject_credential_stuffing(df: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> tuple[pd.DataFrame, list]:
    users = df["user_id"].unique()
    attack_ip = f"attack_ip_{rng.integers(1000, 9999)}"
    n_target_users = rng.integers(*settings.credential_stuffing_users)
    target_users = rng.choice(users, size=min(n_target_users, len(users)), replace=False)
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))

    attack_events = []
    for target_user in target_users:
        n_events = rng.integers(*settings.credential_stuffing_events_per_user)
        for i in range(n_events):
            timestamp = attack_date.replace(hour=rng.integers(0, 24), minute=rng.integers(0, 60), second=rng.integers(0, 60))
            attack_events.append({
                "timestamp": timestamp, "user_id": target_user, "ip_address": attack_ip,
                "endpoint": "/api/v1/login", "status_code": rng.choice([401, 200], p=[0.85, 0.15]),
                "location": rng.choice(["Unknown", "Proxy"]),
                "device_fingerprint": _fingerprint("stuffing", 0, rng),
                "bytes_transferred": int(rng.lognormal(5, 0.5)),
                "is_attack": True, "attack_type": "credential_stuffing",
            })

    record = {"attack_type": "credential_stuffing", "target_entity": list(target_users),
              "source_ip": attack_ip, "window_start": attack_date, "n_events": len(attack_events)}
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


def _inject_geo_anomaly(df: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> tuple[pd.DataFrame, list]:
    users = df["user_id"].unique()
    target_user = rng.choice(users)
    anomalous_locations = ["North-Korea", "Iran", "Unknown-VPN", "TOR-Exit", "Suspicious-Proxy"]
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))
    n_events = rng.integers(5, 20)

    attack_events = []
    for i in range(n_events):
        timestamp = attack_date.replace(hour=rng.integers(0, 24), minute=rng.integers(0, 60), second=rng.integers(0, 60))
        attack_events.append({
            "timestamp": timestamp, "user_id": target_user,
            "ip_address": f"geo_attack_ip_{rng.integers(1000, 9999)}",
            "endpoint": rng.choice(["/api/v1/data", "/api/v1/admin", "/api/v1/reports"]),
            "status_code": 200, "location": rng.choice(anomalous_locations),
            "device_fingerprint": _fingerprint("geo_attacker", i, rng),
            "bytes_transferred": int(rng.lognormal(8, 1)),
            "is_attack": True, "attack_type": "geo_anomaly",
        })

    record = {"attack_type": "geo_anomaly", "target_entity": target_user, "window_start": attack_date, "n_events": n_events}
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


def _inject_device_anomaly(df: pd.DataFrame, settings: Settings, rng: np.random.Generator) -> tuple[pd.DataFrame, list]:
    users = df["user_id"].unique()
    target_user = rng.choice(users)
    n_new_devices = rng.integers(*settings.device_anomaly_new_devices)
    dates = df["timestamp"].dt.date.unique()
    attack_date = pd.Timestamp(rng.choice(dates))

    attack_events = []
    for device_idx in range(n_new_devices):
        n_events = rng.integers(2, 8)
        new_device = _fingerprint(f"new_device_{target_user}", device_idx, rng)
        for i in range(n_events):
            timestamp = attack_date.replace(hour=rng.integers(0, 24), minute=rng.integers(0, 60), second=rng.integers(0, 60))
            attack_events.append({
                "timestamp": timestamp, "user_id": target_user,
                "ip_address": f"device_ip_{rng.integers(1000, 9999)}",
                "endpoint": rng.choice(["/api/v1/login", "/api/v1/data"]),
                "status_code": 200, "location": rng.choice(["US-East", "US-West", "EU-West"]),
                "device_fingerprint": new_device, "bytes_transferred": int(rng.lognormal(6, 1)),
                "is_attack": True, "attack_type": "device_anomaly",
            })

    record = {"attack_type": "device_anomaly", "target_entity": target_user, "window_start": attack_date,
              "n_events": len(attack_events), "n_new_devices": n_new_devices}
    return pd.concat([df, pd.DataFrame(attack_events)], ignore_index=True), [record]


# =============================================================================
# Step 2: Build Features
# =============================================================================


def build_features(events_df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, dict]:
    """
    Transform raw events into modeling table with windowed features.

    Returns:
        modeling_df: Feature table ready for model training
        metadata: Dict with entity mapping and feature info
    """
    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["window"] = df["timestamp"].dt.floor(settings.window_size)

    # Aggregate by entity and window
    agg_funcs = {
        "timestamp": "count",
        "ip_address": "nunique",
        "endpoint": "nunique",
        "device_fingerprint": "nunique",
        "location": "nunique",
        "bytes_transferred": "sum",
        "is_attack": "any",
    }

    grouped = df.groupby([settings.entity_column, "window"]).agg(agg_funcs).reset_index()
    grouped.columns = [settings.entity_column, "window", "event_count", "unique_ips", "unique_endpoints",
                       "unique_devices", "unique_locations", "bytes_total", "has_attack"]

    # Failed count
    failed_mask = df["status_code"].isin([400, 401, 403, 404, 500, 502, 503])
    failed_counts = df[failed_mask].groupby([settings.entity_column, "window"]).size().reset_index(name="failed_count")
    grouped = grouped.merge(failed_counts, on=[settings.entity_column, "window"], how="left")
    grouped["failed_count"] = grouped["failed_count"].fillna(0).astype(int)

    # Attack type
    attack_types = df[df["is_attack"]].groupby([settings.entity_column, "window"])["attack_type"].first().reset_index()
    grouped = grouped.merge(attack_types, on=[settings.entity_column, "window"], how="left")
    grouped["attack_type"] = grouped["attack_type"].fillna("none")

    # Temporal features
    if settings.include_temporal:
        grouped["window"] = pd.to_datetime(grouped["window"])
        grouped["hour"] = grouped["window"].dt.hour
        grouped["day_of_week"] = grouped["window"].dt.dayofweek
        grouped["is_weekend"] = grouped["day_of_week"].isin([5, 6]).astype(int)
        grouped["is_business_hours"] = ((grouped["hour"] >= 9) & (grouped["hour"] <= 17) & (~grouped["is_weekend"].astype(bool))).astype(int)

    # Entity-level features
    entity_stats = grouped.groupby(settings.entity_column)["event_count"].agg(["mean", "std"]).reset_index()
    entity_stats.columns = [settings.entity_column, "entity_mean_count", "entity_std_count"]
    entity_stats["entity_std_count"] = entity_stats["entity_std_count"].fillna(1.0)
    grouped = grouped.merge(entity_stats, on=settings.entity_column, how="left")
    grouped["count_zscore"] = (grouped["event_count"] - grouped["entity_mean_count"]) / grouped["entity_std_count"].clip(lower=0.1)

    # Entity encoding
    unique_entities = grouped[settings.entity_column].unique()
    entity_mapping = {entity: idx for idx, entity in enumerate(unique_entities)}
    grouped["entity_idx"] = grouped[settings.entity_column].map(entity_mapping)

    metadata = {
        "entity_column": settings.entity_column,
        "entity_mapping": entity_mapping,
        "n_entities": len(entity_mapping),
        "n_windows": len(grouped),
        "attack_rate": grouped["has_attack"].mean(),
    }

    return grouped, metadata


# =============================================================================
# Step 3: Get Model Arrays
# =============================================================================


def get_model_arrays(modeling_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract numpy arrays for PyMC model."""
    return {
        "y": modeling_df["event_count"].values.astype(np.int64),
        "entity_idx": modeling_df["entity_idx"].values.astype(np.int64),
        "is_attack": modeling_df["has_attack"].values.astype(bool),
        "n_entities": modeling_df["entity_idx"].nunique(),
    }


# =============================================================================
# Step 4: Train Model
# =============================================================================


def train_model(arrays: dict, settings: Settings) -> az.InferenceData:
    """
    Build and fit hierarchical Negative Binomial model.

    Returns MCMC trace as ArviZ InferenceData.
    """
    y = arrays["y"]
    entity_idx = arrays["entity_idx"]
    n_entities = arrays["n_entities"]

    coords = {"entity": np.arange(n_entities), "obs": np.arange(len(y))}

    with pm.Model(coords=coords) as model:
        entity_idx_data = pm.Data("entity_idx", entity_idx, dims="obs")
        y_data = pm.Data("y_obs", y, dims="obs")

        # Priors
        mu = pm.Exponential("mu", lam=settings.mu_prior_rate)
        alpha = pm.HalfNormal("alpha", sigma=settings.alpha_prior_sd)
        theta = pm.Gamma("theta", alpha=mu * alpha, beta=alpha, dims="entity")
        phi = pm.HalfNormal("phi", sigma=settings.overdispersion_prior_sd)

        # Likelihood
        pm.NegativeBinomial("y", mu=theta[entity_idx_data], alpha=phi, observed=y_data, dims="obs")

    # Fit
    rng = np.random.default_rng(settings.random_seed)
    seed = int(rng.integers(0, 2**31))

    with model:
        trace = pm.sample(
            draws=settings.n_samples,
            tune=settings.n_tune,
            chains=settings.n_chains,
            target_accept=settings.target_accept,
            random_seed=seed,
            cores=min(settings.n_chains, os.cpu_count() or 1),
            return_inferencedata=True,
            progressbar=True,
        )
        trace.extend(pm.sample_posterior_predictive(trace, random_seed=seed + 1))

    return trace


# =============================================================================
# Step 5: Compute Scores
# =============================================================================


def compute_scores(y: np.ndarray, trace: az.InferenceData, entity_idx: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute anomaly scores from posterior predictive.

    Score = -log p(y | posterior) - higher means more anomalous.
    """
    theta = trace.posterior["theta"].values
    phi = trace.posterior["phi"].values

    n_chains, n_draws, n_entities = theta.shape
    theta_flat = theta.reshape(-1, n_entities)
    phi_flat = phi.reshape(-1)

    n_samples = theta_flat.shape[0]
    n_obs = len(y)

    log_likelihoods = np.zeros((n_samples, n_obs))

    for s in range(n_samples):
        mu_s = theta_flat[s, entity_idx]
        phi_s = phi_flat[s]
        n_param = phi_s
        p_param = phi_s / (phi_s + mu_s)
        log_likelihoods[s, :] = stats.nbinom.logpmf(y, n=n_param, p=p_param)

    avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples)
    anomaly_scores = -avg_log_lik

    individual_scores = -log_likelihoods
    score_std = np.std(individual_scores, axis=0)
    score_lower = np.percentile(individual_scores, 5, axis=0)
    score_upper = np.percentile(individual_scores, 95, axis=0)

    return {"anomaly_score": anomaly_scores, "score_std": score_std, "score_lower": score_lower, "score_upper": score_upper}


def compute_intervals(trace: az.InferenceData, entity_idx: np.ndarray, credible_mass: float = 0.9) -> dict[str, np.ndarray]:
    """Compute predictive intervals for each observation."""
    if hasattr(trace, "posterior_predictive") and "y" in trace.posterior_predictive:
        ppc = trace.posterior_predictive["y"].values
        ppc_flat = ppc.reshape(-1, ppc.shape[-1])
        alpha = (1 - credible_mass) / 2
        lower = np.percentile(ppc_flat, alpha * 100, axis=0)
        upper = np.percentile(ppc_flat, (1 - alpha) * 100, axis=0)
        mean = np.mean(ppc_flat, axis=0)
    else:
        theta = trace.posterior["theta"].values
        phi = trace.posterior["phi"].values
        theta_flat = theta.reshape(-1, theta.shape[-1])
        phi_flat = phi.reshape(-1)
        means = theta_flat[:, entity_idx]
        mean = np.mean(means, axis=0)
        avg_phi = np.mean(phi_flat)
        variance = mean + mean**2 / avg_phi
        std = np.sqrt(variance)
        z = stats.norm.ppf((1 + credible_mass) / 2)
        lower = np.maximum(0, mean - z * std)
        upper = mean + z * std

    return {"predicted_mean": mean, "predicted_lower": lower, "predicted_upper": upper}


# =============================================================================
# Step 6: Create Scored DataFrame
# =============================================================================


def create_scored_df(modeling_df: pd.DataFrame, scores: dict, intervals: dict) -> pd.DataFrame:
    """Join scores and intervals to modeling table, sorted by anomaly score."""
    result = modeling_df.copy()

    result["anomaly_score"] = scores["anomaly_score"]
    result["score_std"] = scores["score_std"]
    result["score_lower"] = scores["score_lower"]
    result["score_upper"] = scores["score_upper"]

    result["predicted_mean"] = intervals["predicted_mean"]
    result["predicted_lower"] = intervals["predicted_lower"]
    result["predicted_upper"] = intervals["predicted_upper"]

    result["anomaly_rank"] = result["anomaly_score"].rank(ascending=False, method="first").astype(int)
    result["exceeds_interval"] = result["event_count"] > result["predicted_upper"]

    return result.sort_values("anomaly_score", ascending=False)


# =============================================================================
# Step 7: Evaluate
# =============================================================================


def evaluate(scored_df: pd.DataFrame, k_values: list[int] | None = None) -> dict:
    """Compute all evaluation metrics."""
    if k_values is None:
        k_values = [10, 25, 50, 100]

    y_true = scored_df["has_attack"].astype(int).values
    scores = scored_df["anomaly_score"].values

    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "n_observations": len(y_true),
        "n_positives": int(y_true.sum()),
        "attack_rate": float(y_true.mean()),
    }

    for k in k_values:
        if k <= len(y_true):
            n_positives = y_true.sum()
            top_k_idx = np.argsort(scores)[-k:]
            tp_at_k = y_true[top_k_idx].sum()
            metrics[f"recall_at_{k}"] = float(tp_at_k / n_positives) if n_positives > 0 else 0.0
            metrics[f"precision_at_{k}"] = float(tp_at_k / k)

    # PR curve for plotting
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

    return metrics


# =============================================================================
# Step 8: Create Plots
# =============================================================================


def create_plots(scored_df: pd.DataFrame, metrics: dict, trace: az.InferenceData | None, output_dir: Path) -> dict[str, Path]:
    """Generate all visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plots = {}

    # Score distribution
    path = output_dir / "score_distribution.png"
    _plot_score_distribution(scored_df, path)
    plots["score_distribution"] = path

    # PR curve
    path = output_dir / "precision_recall_curve.png"
    _plot_pr_curve(metrics, path)
    plots["precision_recall_curve"] = path

    # Top anomalies
    path = output_dir / "top_anomalies.png"
    _plot_top_anomalies(scored_df, path)
    plots["top_anomalies"] = path

    # Model diagnostics
    if trace is not None:
        path = output_dir / "model_diagnostics.png"
        _plot_diagnostics(trace, path)
        plots["model_diagnostics"] = path

    return plots


def _plot_score_distribution(scored_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    benign = scored_df[~scored_df["has_attack"]]["anomaly_score"]
    attack = scored_df[scored_df["has_attack"]]["anomaly_score"]

    axes[0].hist(benign, bins=50, alpha=0.7, label=f"Benign (n={len(benign):,})", color="steelblue", density=True)
    axes[0].hist(attack, bins=50, alpha=0.7, label=f"Attack (n={len(attack):,})", color="crimson", density=True)
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Anomaly Score Distribution by Class")
    axes[0].legend()

    plot_data = scored_df[["anomaly_score", "has_attack"]].copy()
    plot_data["Class"] = plot_data["has_attack"].map({True: "Attack", False: "Benign"})
    sns.boxplot(data=plot_data, x="Class", y="anomaly_score", hue="Class", ax=axes[1],
                palette={"Benign": "steelblue", "Attack": "crimson"}, legend=False)
    axes[1].set_ylabel("Anomaly Score")
    axes[1].set_title("Score Distribution by Class")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pr_curve(metrics: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if "pr_curve" in metrics:
        ax.plot(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], linewidth=2, color="steelblue")
        ax.fill_between(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], alpha=0.2, color="steelblue")

    ax.axhline(y=metrics.get("attack_rate", 0.02), color="gray", linestyle="--",
               label=f"Baseline (attack rate = {metrics.get('attack_rate', 0.02):.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (PR-AUC = {metrics.get('pr_auc', 0):.3f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_top_anomalies(scored_df: pd.DataFrame, path: Path, n: int = 20) -> None:
    from matplotlib.patches import Patch

    top_df = scored_df.head(n).copy().sort_values("anomaly_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))

    colors = ["crimson" if attack else "steelblue" for attack in top_df["has_attack"]]
    y_pos = np.arange(len(top_df))

    lower_err = np.maximum(0, top_df["anomaly_score"] - top_df["score_lower"])
    upper_err = np.maximum(0, top_df["score_upper"] - top_df["anomaly_score"])

    ax.barh(y_pos, top_df["anomaly_score"], xerr=[lower_err, upper_err], color=colors, alpha=0.8, capsize=3)

    labels = [f"{row['user_id']} ({row['window'].strftime('%m-%d') if hasattr(row['window'], 'strftime') else row['window']})"
              for _, row in top_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"Top {n} Anomalies (Red = Attack, Blue = Benign)")

    legend_elements = [Patch(facecolor="crimson", alpha=0.8, label="Attack"), Patch(facecolor="steelblue", alpha=0.8, label="Benign")]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_diagnostics(trace: az.InferenceData, path: Path) -> None:
    az.plot_trace(trace, var_names=["mu", "alpha", "phi"], figsize=(14, 8), compact=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Step 9: Get Model Diagnostics
# =============================================================================


def get_diagnostics(trace: az.InferenceData) -> dict:
    """Get MCMC convergence diagnostics."""
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])
    return {
        "r_hat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "divergences": int(trace.sample_stats["diverging"].sum().values),
        "converged": bool(summary["r_hat"].max() < 1.05),
    }
