"""
File I/O helpers for BSAD pipeline.

Centralized load/save functions for parquet, NetCDF, JSON, and run metadata.
"""

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import pandas as pd


# =============================================================================
# Run Metadata
# =============================================================================


@dataclass
class RunMetadata:
    """
    Snapshot of run configuration for reproducibility and auditing.

    Saved alongside model outputs so every artifact is traceable to
    the exact settings and environment that produced it.
    """

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    random_seed: int = 42
    git_commit: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Any) -> "RunMetadata":
        """
        Build RunMetadata from a Settings object.

        Captures git commit if the working directory is inside a git repo.
        """
        git_commit = _get_git_commit()
        # Snapshot only the primitive settings fields (skip Path objects)
        snapshot = {}
        if hasattr(settings, "__dataclass_fields__"):
            for f_name in settings.__dataclass_fields__:
                val = getattr(settings, f_name)
                if isinstance(val, (int, float, str, bool, tuple, list)):
                    snapshot[f_name] = val
        return cls(
            random_seed=getattr(settings, "random_seed", 42),
            git_commit=git_commit,
            config_snapshot=snapshot,
        )

    def save(self, path: Path) -> None:
        """Persist metadata as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


def save_run_metadata(metadata: RunMetadata, path: Path) -> None:
    """Save RunMetadata to JSON at the given path."""
    metadata.save(path)


def _get_git_commit() -> str | None:
    """Return the current HEAD commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


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
