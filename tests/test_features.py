"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from bsad.data_generator import GeneratorConfig, generate_synthetic_data
from bsad.features import (
    FeatureConfig,
    add_entity_features,
    add_temporal_features,
    build_modeling_table,
    create_time_windows,
    encode_entity_ids,
    get_model_arrays,
)


@pytest.fixture
def sample_events():
    """Generate sample events for testing."""
    config = GeneratorConfig(n_users=20, n_days=7, attack_rate=0.05, random_seed=42)
    events_df, _ = generate_synthetic_data(config)
    return events_df


class TestTimeWindows:
    def test_creates_windows(self, sample_events):
        config = FeatureConfig(window_size="1D")
        windowed = create_time_windows(sample_events, config)

        assert "window" in windowed.columns
        assert "event_count" in windowed.columns
        assert len(windowed) > 0

    def test_aggregation_correctness(self, sample_events):
        config = FeatureConfig(window_size="1D")
        windowed = create_time_windows(sample_events, config)

        # Total event count should match
        assert windowed["event_count"].sum() == len(sample_events)

    def test_unique_counts(self, sample_events):
        config = FeatureConfig(window_size="1D")
        windowed = create_time_windows(sample_events, config)

        assert "unique_ips" in windowed.columns
        assert "unique_endpoints" in windowed.columns
        assert "unique_devices" in windowed.columns


class TestTemporalFeatures:
    def test_adds_temporal_columns(self, sample_events):
        config = FeatureConfig()
        windowed = create_time_windows(sample_events, config)
        with_temporal = add_temporal_features(windowed)

        assert "hour" in with_temporal.columns
        assert "day_of_week" in with_temporal.columns
        assert "is_weekend" in with_temporal.columns
        assert "is_business_hours" in with_temporal.columns

    def test_hour_range(self, sample_events):
        config = FeatureConfig()
        windowed = create_time_windows(sample_events, config)
        with_temporal = add_temporal_features(windowed)

        assert with_temporal["hour"].min() >= 0
        assert with_temporal["hour"].max() <= 23

    def test_day_of_week_range(self, sample_events):
        config = FeatureConfig()
        windowed = create_time_windows(sample_events, config)
        with_temporal = add_temporal_features(windowed)

        assert with_temporal["day_of_week"].min() >= 0
        assert with_temporal["day_of_week"].max() <= 6


class TestEntityFeatures:
    def test_adds_entity_stats(self, sample_events):
        config = FeatureConfig()
        windowed = create_time_windows(sample_events, config)
        with_entity = add_entity_features(windowed, config.entity_column)

        assert "entity_mean_count" in with_entity.columns
        assert "entity_std_count" in with_entity.columns
        assert "count_zscore" in with_entity.columns


class TestEntityEncoding:
    def test_creates_integer_encoding(self, sample_events):
        config = FeatureConfig()
        windowed = create_time_windows(sample_events, config)
        encoded, mapping = encode_entity_ids(windowed, config.entity_column)

        assert "entity_idx" in encoded.columns
        assert len(mapping) == windowed[config.entity_column].nunique()

    def test_mapping_is_contiguous(self, sample_events):
        config = FeatureConfig()
        windowed = create_time_windows(sample_events, config)
        encoded, mapping = encode_entity_ids(windowed, config.entity_column)

        indices = list(mapping.values())
        assert min(indices) == 0
        assert max(indices) == len(mapping) - 1


class TestModelingTable:
    def test_full_pipeline(self, sample_events):
        config = FeatureConfig()
        modeling_df, metadata = build_modeling_table(sample_events, config)

        assert len(modeling_df) > 0
        assert "entity_idx" in modeling_df.columns
        assert "n_entities" in metadata
        assert "n_windows" in metadata

    def test_metadata_contents(self, sample_events):
        config = FeatureConfig()
        modeling_df, metadata = build_modeling_table(sample_events, config)

        assert metadata["n_entities"] > 0
        assert metadata["n_windows"] == len(modeling_df)
        assert 0 <= metadata["attack_rate"] <= 1


class TestModelArrays:
    def test_array_extraction(self, sample_events):
        config = FeatureConfig()
        modeling_df, metadata = build_modeling_table(sample_events, config)
        arrays = get_model_arrays(modeling_df)

        assert "y" in arrays
        assert "entity_idx" in arrays
        assert "is_attack" in arrays

        assert len(arrays["y"]) == len(modeling_df)
        assert arrays["y"].dtype == np.int64
        assert arrays["entity_idx"].dtype == np.int64
