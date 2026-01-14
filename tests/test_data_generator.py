"""Tests for synthetic data generator."""

import pandas as pd
import pytest

from bsad.data_generator import (
    GeneratorConfig,
    generate_baseline_events,
    generate_synthetic_data,
    inject_brute_force_attack,
)
import numpy as np


class TestGeneratorConfig:
    def test_default_config(self):
        config = GeneratorConfig()
        assert config.n_users == 200
        assert config.n_days == 30
        assert config.attack_rate == 0.02

    def test_custom_config(self):
        config = GeneratorConfig(n_users=50, n_days=7, attack_rate=0.05)
        assert config.n_users == 50
        assert config.n_days == 7
        assert config.attack_rate == 0.05


class TestBaselineEvents:
    def test_generates_events(self):
        config = GeneratorConfig(n_users=10, n_days=5, random_seed=42)
        rng = np.random.default_rng(config.random_seed)
        df = generate_baseline_events(config, rng)

        assert len(df) > 0
        assert "user_id" in df.columns
        assert "timestamp" in df.columns
        assert "is_attack" in df.columns

    def test_all_baseline_events_benign(self):
        config = GeneratorConfig(n_users=10, n_days=5, random_seed=42)
        rng = np.random.default_rng(config.random_seed)
        df = generate_baseline_events(config, rng)

        assert df["is_attack"].sum() == 0

    def test_expected_columns(self):
        config = GeneratorConfig(n_users=10, n_days=5, random_seed=42)
        rng = np.random.default_rng(config.random_seed)
        df = generate_baseline_events(config, rng)

        expected_columns = [
            "timestamp",
            "user_id",
            "ip_address",
            "endpoint",
            "status_code",
            "location",
            "device_fingerprint",
            "bytes_transferred",
            "is_attack",
            "attack_type",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"


class TestAttackInjection:
    def test_brute_force_attack(self):
        config = GeneratorConfig(n_users=10, n_days=5, random_seed=42)
        rng = np.random.default_rng(config.random_seed)
        df = generate_baseline_events(config, rng)

        df_with_attack, records = inject_brute_force_attack(df, config, rng)

        assert len(df_with_attack) > len(df)
        assert df_with_attack["is_attack"].sum() > 0
        assert len(records) == 1
        assert records[0]["attack_type"] == "brute_force"


class TestSyntheticData:
    def test_full_generation(self):
        config = GeneratorConfig(n_users=20, n_days=7, attack_rate=0.05, random_seed=42)
        events_df, attacks_df = generate_synthetic_data(config)

        assert len(events_df) > 0
        assert len(attacks_df) > 0
        assert events_df["is_attack"].sum() > 0

    def test_reproducibility(self):
        config = GeneratorConfig(n_users=10, n_days=5, random_seed=123)

        events1, _ = generate_synthetic_data(config)
        events2, _ = generate_synthetic_data(config)

        # Same seed should produce same results
        assert len(events1) == len(events2)
        pd.testing.assert_frame_equal(events1, events2)

    def test_attack_types_present(self):
        config = GeneratorConfig(n_users=50, n_days=14, attack_rate=0.1, random_seed=42)
        events_df, _ = generate_synthetic_data(config)

        attack_types = events_df[events_df["is_attack"]]["attack_type"].unique()
        assert len(attack_types) >= 1  # At least one attack type
