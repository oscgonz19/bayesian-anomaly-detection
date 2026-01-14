"""
File I/O helpers for BSAD pipeline.

Centralized load/save functions for parquet, NetCDF, and JSON.
"""

import json
from pathlib import Path

import arviz as az
import pandas as pd


# =============================================================================
# Parquet I/O
# =============================================================================


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet file."""
    return pd.read_parquet(path)


# =============================================================================
# Model (NetCDF) I/O
# =============================================================================


def save_model(trace: az.InferenceData, path: Path) -> None:
    """Save ArviZ InferenceData (MCMC trace) to NetCDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(str(path))


def load_model(path: Path) -> az.InferenceData:
    """Load ArviZ InferenceData from NetCDF."""
    return az.from_netcdf(str(path))


# =============================================================================
# JSON I/O
# =============================================================================


def save_json(data: dict, path: Path) -> None:
    """Save dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable items (like numpy arrays in pr_curve)
    clean_data = {k: v for k, v in data.items() if k != "pr_curve"}
    with open(path, "w") as f:
        json.dump(clean_data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load dictionary from JSON file."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Convenience Functions
# =============================================================================


def save_events(events_df: pd.DataFrame, attacks_df: pd.DataFrame, events_path: Path) -> None:
    """
    Save events and attacks metadata to parquet files.

    Handles list columns by converting to strings.
    """
    save_parquet(events_df, events_path)

    # Save attacks metadata alongside
    attacks_path = events_path.parent / (events_path.stem + "_attacks.parquet")

    # Convert list columns to strings for parquet compatibility
    attacks_df_copy = attacks_df.copy()
    if "target_entity" in attacks_df_copy.columns:
        attacks_df_copy["target_entity"] = attacks_df_copy["target_entity"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else str(x)
        )

    save_parquet(attacks_df_copy, attacks_path)
