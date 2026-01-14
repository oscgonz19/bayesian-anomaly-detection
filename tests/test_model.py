"""Tests for Bayesian model module."""

import numpy as np
import pytest

from bsad.model import ModelConfig, build_hierarchical_negbinom_model


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.n_samples == 2000
        assert config.n_chains == 4
        assert config.target_accept == 0.9

    def test_custom_config(self):
        config = ModelConfig(n_samples=500, n_chains=2)
        assert config.n_samples == 500
        assert config.n_chains == 2


class TestModelBuilding:
    def test_model_creation(self):
        """Test that model builds without errors."""
        np.random.seed(42)
        n_obs = 100
        n_entities = 10

        y = np.random.poisson(5, size=n_obs)
        entity_idx = np.random.randint(0, n_entities, size=n_obs)

        model = build_hierarchical_negbinom_model(
            y=y,
            entity_idx=entity_idx,
            n_entities=n_entities,
        )

        assert model is not None
        assert "mu" in [v.name for v in model.free_RVs]
        assert "alpha" in [v.name for v in model.free_RVs]
        assert "theta" in [v.name for v in model.free_RVs]
        assert "phi" in [v.name for v in model.free_RVs]

    def test_model_with_config(self):
        """Test model building with custom config."""
        np.random.seed(42)
        n_obs = 50
        n_entities = 5

        y = np.random.poisson(3, size=n_obs)
        entity_idx = np.random.randint(0, n_entities, size=n_obs)

        config = ModelConfig(mu_prior_rate=0.2, alpha_prior_sd=1.0)
        model = build_hierarchical_negbinom_model(
            y=y,
            entity_idx=entity_idx,
            n_entities=n_entities,
            config=config,
        )

        assert model is not None


# Note: Full model fitting tests are slow and typically run in integration tests
# The following test is marked as slow and can be skipped in normal test runs

@pytest.mark.slow
class TestModelFitting:
    def test_model_fits(self):
        """Test that model can be fit (slow test)."""
        from bsad.model import fit_model, get_model_diagnostics

        np.random.seed(42)
        n_obs = 50
        n_entities = 5

        y = np.random.poisson(5, size=n_obs)
        entity_idx = np.random.randint(0, n_entities, size=n_obs)

        config = ModelConfig(
            n_samples=100,
            n_tune=50,
            n_chains=2,
        )

        model = build_hierarchical_negbinom_model(
            y=y,
            entity_idx=entity_idx,
            n_entities=n_entities,
            config=config,
        )

        trace = fit_model(model, config)

        assert trace is not None
        assert "posterior" in trace.groups()

        diagnostics = get_model_diagnostics(trace)
        assert "r_hat_max" in diagnostics
        assert "divergences" in diagnostics
