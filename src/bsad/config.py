"""
Configuration for BSAD pipeline.

All settings in one place - paths, model params, generation params.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """All pipeline settings in one place."""

    # ==========================================================================
    # Paths
    # ==========================================================================
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # ==========================================================================
    # Data Generation
    # ==========================================================================
    n_entities: int = 200
    n_ips: int = 100
    n_endpoints: int = 50
    n_days: int = 30
    events_per_user_day_mean: float = 5.0
    events_per_user_day_std: float = 3.0
    attack_rate: float = 0.02

    # Attack pattern parameters
    brute_force_multiplier: tuple[int, int] = (50, 200)
    credential_stuffing_users: tuple[int, int] = (10, 30)
    credential_stuffing_events_per_user: tuple[int, int] = (3, 10)
    device_anomaly_new_devices: tuple[int, int] = (3, 8)

    # ==========================================================================
    # Feature Engineering
    # ==========================================================================
    entity_column: str = "user_id"
    window_size: str = "1D"
    include_temporal: bool = True

    # ==========================================================================
    # Model Training
    # ==========================================================================
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 2
    target_accept: float = 0.9

    # Prior parameters
    mu_prior_rate: float = 0.1
    alpha_prior_sd: float = 2.0
    overdispersion_prior_sd: float = 2.0

    # ==========================================================================
    # General
    # ==========================================================================
    random_seed: int = 42

    # ==========================================================================
    # Derived Paths (computed properties)
    # ==========================================================================
    @property
    def events_path(self) -> Path:
        return self.data_dir / "events.parquet"

    @property
    def attacks_path(self) -> Path:
        return self.data_dir / "events_attacks.parquet"

    @property
    def modeling_table_path(self) -> Path:
        return self.output_dir / "modeling_table.parquet"

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model.nc"

    @property
    def scores_path(self) -> Path:
        return self.output_dir / "scores.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
