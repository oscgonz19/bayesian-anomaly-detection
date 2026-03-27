"""
Feature engineering for the hierarchical count anomaly detector.

WHAT THE MODEL ACTUALLY USES
-----------------------------
The NB model consumes only two arrays:
  - y (event_count per entity-window, int)
  - entity_idx (integer entity ID)

All other features (unique_ips, unique_devices, temporal, z-score)
are computed here for:
  - Diagnostic / EDA purposes
  - Future model extensions (covariates, multi-signal)
  - Analyst enrichment in the triage layer

They are NOT part of the NB likelihood.

LEAKAGE NOTE
------------
entity_mean_count and count_zscore are computed from the full dataset
passed to build_modeling_table. When using temporal train/test splits,
call build_modeling_table separately on train and test to avoid leakage
from test windows into entity baseline statistics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for the feature engineering step."""

    entity_column: str = "user_id"
    window_size: str = "1D"
    include_temporal: bool = True


def create_time_windows(events_df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """
    Aggregate raw events into entity-window count features.

    Groups by (entity_column, time_window) and computes:
      event_count, unique_ips, unique_endpoints, unique_devices,
      unique_locations, bytes_total, has_attack, attack_type, failed_count.

    The sum of event_count equals len(events_df): every raw event
    contributes exactly once to its entity-window bucket.

    Args:
        events_df: Raw event log with at minimum columns:
            timestamp, <entity_column>, ip_address, endpoint,
            device_fingerprint, location, bytes_transferred,
            status_code, is_attack, attack_type.
        config: Feature configuration.

    Returns:
        DataFrame with one row per (entity, window).
    """
    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["window"] = df["timestamp"].dt.floor(config.window_size)

    agg_funcs = {
        "timestamp": "count",
        "ip_address": "nunique",
        "endpoint": "nunique",
        "device_fingerprint": "nunique",
        "location": "nunique",
        "bytes_transferred": "sum",
        "is_attack": "any",
    }

    grouped = (
        df.groupby([config.entity_column, "window"])
        .agg(agg_funcs)
        .reset_index()
    )
    grouped.columns = [
        config.entity_column, "window",
        "event_count", "unique_ips", "unique_endpoints",
        "unique_devices", "unique_locations", "bytes_total", "has_attack",
    ]

    # Failed status codes
    failed_mask = df["status_code"].isin([400, 401, 403, 404, 500, 502, 503])
    failed_counts = (
        df[failed_mask]
        .groupby([config.entity_column, "window"])
        .size()
        .reset_index(name="failed_count")
    )
    grouped = grouped.merge(failed_counts, on=[config.entity_column, "window"], how="left")
    grouped["failed_count"] = grouped["failed_count"].fillna(0).astype(int)

    # Attack type (first attack type seen in window, if any)
    attack_types = (
        df[df["is_attack"]]
        .groupby([config.entity_column, "window"])["attack_type"]
        .first()
        .reset_index()
    )
    grouped = grouped.merge(attack_types, on=[config.entity_column, "window"], how="left")
    grouped["attack_type"] = grouped["attack_type"].fillna("none")

    return grouped


def add_temporal_features(windowed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day and day-of-week features to a windowed feature table.

    These are NOT used by the current NB model but are available for
    future covariate extensions and analyst dashboards.

    Args:
        windowed_df: Output of create_time_windows.

    Returns:
        DataFrame with added columns: hour, day_of_week, is_weekend,
        is_business_hours.
    """
    df = windowed_df.copy()
    df["window"] = pd.to_datetime(df["window"])
    df["hour"] = df["window"].dt.hour
    df["day_of_week"] = df["window"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = (
        (df["hour"] >= 9) & (df["hour"] <= 17) & (~df["is_weekend"].astype(bool))
    ).astype(int)
    return df


def add_entity_features(windowed_df: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    """
    Add entity-level baseline statistics: mean, std, and z-score.

    WARNING: Computed from the entire dataframe passed in. Do not mix
    training and test windows here—compute separately to avoid leakage.

    Args:
        windowed_df: Output of create_time_windows.
        entity_column: Column name identifying the entity.

    Returns:
        DataFrame with added columns: entity_mean_count, entity_std_count,
        count_zscore.
    """
    df = windowed_df.copy()
    entity_stats = (
        df.groupby(entity_column)["event_count"]
        .agg(["mean", "std"])
        .reset_index()
    )
    entity_stats.columns = [entity_column, "entity_mean_count", "entity_std_count"]
    entity_stats["entity_std_count"] = entity_stats["entity_std_count"].fillna(1.0)
    df = df.merge(entity_stats, on=entity_column, how="left")
    df["count_zscore"] = (
        (df["event_count"] - df["entity_mean_count"])
        / df["entity_std_count"].clip(lower=0.1)
    )
    return df


def encode_entity_ids(
    windowed_df: pd.DataFrame, entity_column: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Assign contiguous integer IDs to entities (0 to n_entities-1).

    The mapping is derived from the order entities appear in the DataFrame.
    For reproducible mappings across train/test splits, pass a pre-built
    mapping to the NB model instead of re-encoding.

    Args:
        windowed_df: Feature table.
        entity_column: Column name identifying the entity.

    Returns:
        (encoded_df, entity_mapping) where entity_mapping maps
        entity name → integer index.
    """
    df = windowed_df.copy()
    unique_entities = df[entity_column].unique()
    entity_mapping = {entity: idx for idx, entity in enumerate(unique_entities)}
    df["entity_idx"] = df[entity_column].map(entity_mapping)
    return df, entity_mapping


def build_modeling_table(
    events_df: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, dict]:
    """
    Full feature engineering pipeline: events → modeling table.

    Composes: create_time_windows → add_temporal_features →
              add_entity_features → encode_entity_ids.

    Args:
        events_df: Raw event log.
        config: Feature configuration.

    Returns:
        modeling_df: Ready-to-model feature table.
        metadata: Dict with keys entity_column, entity_mapping,
                  n_entities, n_windows, attack_rate.
    """
    windowed = create_time_windows(events_df, config)
    if config.include_temporal:
        windowed = add_temporal_features(windowed)
    windowed = add_entity_features(windowed, config.entity_column)
    modeling_df, entity_mapping = encode_entity_ids(windowed, config.entity_column)

    metadata = {
        "entity_column": config.entity_column,
        "entity_mapping": entity_mapping,
        "n_entities": len(entity_mapping),
        "n_windows": len(modeling_df),
        "attack_rate": float(modeling_df["has_attack"].mean()),
    }

    return modeling_df, metadata


def get_model_arrays(modeling_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Extract the minimal numpy arrays required by the NB model.

    Returns:
        dict with keys:
          y          – event counts, int64
          entity_idx – integer entity indices, int64
          is_attack  – ground-truth labels, bool (NOT used in training)
          n_entities – number of unique entities, int
    """
    return {
        "y": modeling_df["event_count"].values.astype(np.int64),
        "entity_idx": modeling_df["entity_idx"].values.astype(np.int64),
        "is_attack": modeling_df["has_attack"].values.astype(bool),
        "n_entities": int(modeling_df["entity_idx"].nunique()),
    }
