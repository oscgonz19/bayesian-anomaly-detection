"""
UNSW-NB15 Dataset Adapter for BSAD Pipeline.

Transforms the UNSW-NB15 network intrusion detection dataset into the format
expected by the Bayesian Security Anomaly Detection pipeline.

Key Mappings:
- Entity: proto + service combination (network flow profile)
- Count Variable: spkts (source packets) - shows strong overdispersion
- Window: Synthetic windows created by batching consecutive flows
- Attack Label: label column (0=normal, 1=attack)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class UNSWSettings:
    """Configuration for UNSW-NB15 adapter."""

    # Paths
    train_path: Path = field(default_factory=lambda: Path("data/UNSW_NB15_training-set.parquet"))
    test_path: Path = field(default_factory=lambda: Path("data/UNSW_NB15_testing-set.parquet"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/unsw"))

    # Entity definition
    entity_columns: list[str] = field(default_factory=lambda: ["proto", "service"])

    # Count variable
    count_column: str = "spkts"

    # Window settings (synthetic windows since no timestamp)
    window_size: int = 1000  # Number of flows per window
    min_entity_windows: int = 3  # Minimum windows per entity for training

    # Model parameters (can override)
    n_samples: int = 1500
    n_tune: int = 1000
    n_chains: int = 2
    target_accept: float = 0.9

    # Priors
    mu_prior_rate: float = 0.05  # Lower rate for higher prior mean (more packets expected)
    alpha_prior_sd: float = 2.0
    overdispersion_prior_sd: float = 3.0  # Higher for more overdispersion

    random_seed: int = 42

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_unsw_data(settings: UNSWSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load UNSW-NB15 training and testing datasets.

    Returns:
        train_df, test_df: DataFrames with raw UNSW-NB15 data
    """
    train_df = pd.read_parquet(settings.train_path)
    test_df = pd.read_parquet(settings.test_path)

    return train_df, test_df


def create_entity(df: pd.DataFrame, settings: UNSWSettings) -> pd.DataFrame:
    """
    Create entity column by combining specified columns.

    Entity = proto_service (e.g., "tcp_http", "udp_dns")
    """
    df = df.copy()
    entity_parts = [df[col].astype(str) for col in settings.entity_columns]
    df["entity"] = entity_parts[0]
    for part in entity_parts[1:]:
        df["entity"] = df["entity"] + "_" + part

    return df


def create_windows(df: pd.DataFrame, settings: UNSWSettings) -> pd.DataFrame:
    """
    Create synthetic time windows by batching consecutive flows.

    Since UNSW-NB15 doesn't have timestamps, we create windows by:
    1. Sorting by entity
    2. Assigning window IDs based on row position within entity
    """
    df = df.copy()

    # Add row number within each entity
    df["entity_row"] = df.groupby("entity").cumcount()

    # Create window ID
    df["window"] = df["entity_row"] // settings.window_size

    return df


def build_modeling_table(df: pd.DataFrame, settings: UNSWSettings) -> tuple[pd.DataFrame, dict]:
    """
    Build the modeling table from UNSW-NB15 data.

    Aggregates flows by entity and window to create features compatible
    with the BSAD pipeline.

    Returns:
        modeling_df: Feature table for model training
        metadata: Dictionary with entity mapping and feature info
    """
    # Create entity and windows
    df = create_entity(df, settings)
    df = create_windows(df, settings)

    # Aggregate by entity and window
    agg_funcs = {
        settings.count_column: "sum",  # Total source packets in window
        "dpkts": "sum",  # Total dest packets
        "sbytes": "sum",  # Total source bytes
        "dbytes": "sum",  # Total dest bytes
        "dur": "sum",  # Total duration
        "rate": "mean",  # Average rate
        "sload": "mean",  # Average source load
        "dload": "mean",  # Average dest load
        "label": "any",  # Has any attack in window
        "attack_cat": lambda x: x[x != "Normal"].iloc[0] if (x != "Normal").any() else "Normal"
    }

    grouped = df.groupby(["entity", "window"]).agg(agg_funcs).reset_index()

    # Rename columns to match BSAD pipeline
    grouped = grouped.rename(columns={
        settings.count_column: "event_count",
        "label": "has_attack",
        "attack_cat": "attack_type"
    })

    # Convert attack type to string
    grouped["attack_type"] = grouped["attack_type"].astype(str)

    # Filter entities with minimum windows
    entity_window_counts = grouped.groupby("entity").size()
    valid_entities = entity_window_counts[entity_window_counts >= settings.min_entity_windows].index
    grouped = grouped[grouped["entity"].isin(valid_entities)].reset_index(drop=True)

    # Entity encoding
    unique_entities = grouped["entity"].unique()
    entity_mapping = {entity: idx for idx, entity in enumerate(unique_entities)}
    grouped["entity_idx"] = grouped["entity"].map(entity_mapping)

    # Add entity-level statistics
    entity_stats = grouped.groupby("entity")["event_count"].agg(["mean", "std"]).reset_index()
    entity_stats.columns = ["entity", "entity_mean_count", "entity_std_count"]
    entity_stats["entity_std_count"] = entity_stats["entity_std_count"].fillna(1.0)
    grouped = grouped.merge(entity_stats, on="entity", how="left")

    # Z-score
    grouped["count_zscore"] = (grouped["event_count"] - grouped["entity_mean_count"]) / grouped["entity_std_count"].clip(lower=0.1)

    # Metadata
    metadata = {
        "entity_column": "entity",
        "entity_mapping": entity_mapping,
        "n_entities": len(entity_mapping),
        "n_windows": len(grouped),
        "attack_rate": grouped["has_attack"].mean(),
        "count_column": settings.count_column,
        "window_size": settings.window_size,
        "source": "UNSW-NB15"
    }

    return grouped, metadata


def get_model_arrays(modeling_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract numpy arrays for PyMC model."""
    return {
        "y": modeling_df["event_count"].values.astype(np.int64),
        "entity_idx": modeling_df["entity_idx"].values.astype(np.int64),
        "is_attack": modeling_df["has_attack"].values.astype(bool),
        "n_entities": modeling_df["entity_idx"].nunique(),
    }


def analyze_overdispersion(df: pd.DataFrame, settings: UNSWSettings) -> dict:
    """
    Analyze overdispersion in the count variable.

    Returns statistics useful for setting prior parameters.
    """
    count_col = settings.count_column

    # Overall statistics
    mean = df[count_col].mean()
    var = df[count_col].var()
    overdispersion_ratio = var / mean if mean > 0 else 0

    # By entity
    entity_stats = df.groupby(create_entity(df, settings)["entity"])[count_col].agg(["mean", "var"])
    entity_stats["overdispersion"] = entity_stats["var"] / entity_stats["mean"]
    entity_stats = entity_stats.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "overall_mean": mean,
        "overall_var": var,
        "overall_overdispersion_ratio": overdispersion_ratio,
        "entity_mean_overdispersion": entity_stats["overdispersion"].mean(),
        "entity_median_overdispersion": entity_stats["overdispersion"].median(),
        "n_entities_analyzed": len(entity_stats),
        "recommendation": {
            "mu_prior_rate": 1 / mean if mean > 0 else 0.1,
            "overdispersion_prior_sd": min(5.0, np.sqrt(overdispersion_ratio) / 10) if overdispersion_ratio > 1 else 2.0
        }
    }


def summary_statistics(modeling_df: pd.DataFrame, metadata: dict) -> None:
    """Print summary statistics for the modeling table."""
    print("=" * 70)
    print("UNSW-NB15 -> BSAD Modeling Table Summary")
    print("=" * 70)
    print(f"\nSource: {metadata['source']}")
    print(f"Window size: {metadata['window_size']} flows per window")
    print(f"\nDimensions:")
    print(f"  Total windows: {metadata['n_windows']:,}")
    print(f"  Unique entities: {metadata['n_entities']}")
    print(f"  Attack rate: {metadata['attack_rate']:.2%}")

    print(f"\nEvent count (aggregated {metadata['count_column']}):")
    print(f"  Mean: {modeling_df['event_count'].mean():.2f}")
    print(f"  Std: {modeling_df['event_count'].std():.2f}")
    print(f"  Variance: {modeling_df['event_count'].var():.2f}")
    print(f"  Var/Mean ratio: {modeling_df['event_count'].var()/modeling_df['event_count'].mean():.2f}")

    print(f"\nAttack distribution:")
    print(modeling_df["attack_type"].value_counts().to_string())

    print("\nTop 10 entities by window count:")
    entity_counts = modeling_df["entity"].value_counts().head(10)
    print(entity_counts.to_string())


# Example usage
if __name__ == "__main__":
    settings = UNSWSettings()
    settings.ensure_dirs()

    # Load data
    print("Loading UNSW-NB15 data...")
    train_df, test_df = load_unsw_data(settings)
    print(f"Training: {train_df.shape}, Testing: {test_df.shape}")

    # Analyze overdispersion
    print("\nAnalyzing overdispersion...")
    overdispersion = analyze_overdispersion(train_df, settings)
    print(f"Overdispersion ratio: {overdispersion['overall_overdispersion_ratio']:.2f}")

    # Build modeling table
    print("\nBuilding modeling table...")
    modeling_df, metadata = build_modeling_table(train_df, settings)

    # Summary
    summary_statistics(modeling_df, metadata)

    # Get arrays for model
    arrays = get_model_arrays(modeling_df)
    print(f"\nModel arrays ready:")
    print(f"  y shape: {arrays['y'].shape}")
    print(f"  entity_idx shape: {arrays['entity_idx'].shape}")
    print(f"  n_entities: {arrays['n_entities']}")
