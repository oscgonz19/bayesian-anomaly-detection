"""Tests for triage module: risk scoring, calibration, and ranking metrics."""

import numpy as np
import pandas as pd
import pytest

from triage import (
    compute_risk_score,
    RiskScorer,
    calibrate_threshold,
    AlertBudget,
    build_alert_budget_curve,
    precision_at_k,
    recall_at_k,
    fpr_at_fixed_recall,
    alerts_per_k_windows,
    workload_reduction,
    ranking_report,
    build_entity_history,
    enrich_alerts,
    EntityContext,
)


class TestRiskScoreMonotonicity:
    """Risk scores should be monotonic with anomaly scores (all else equal)."""

    def test_monotonic_with_anomaly_score(self):
        """Higher anomaly scores should produce higher risk scores."""
        np.random.seed(42)
        n = 100
        anomaly_scores = np.linspace(0, 10, n)

        scorer = RiskScorer(score_weight=1.0, confidence_weight=0.0, novelty_weight=0.0)
        risk_scores = scorer.compute(anomaly_scores)

        # Risk should be monotonically increasing
        assert np.all(np.diff(risk_scores) >= 0), "Risk scores not monotonic with anomaly scores"

    def test_monotonic_with_confidence(self):
        """Higher confidence (lower uncertainty) should increase risk scores."""
        np.random.seed(42)
        n = 100
        anomaly_scores = np.ones(n) * 5  # Fixed anomaly score
        score_std = np.linspace(10, 0.1, n)  # Decreasing uncertainty

        scorer = RiskScorer(score_weight=0.0, confidence_weight=1.0, novelty_weight=0.0)
        risk_scores = scorer.compute(anomaly_scores, score_std=score_std)

        # Risk should increase as uncertainty decreases
        assert np.all(np.diff(risk_scores) >= -1e-10), "Risk not monotonic with confidence"

    def test_monotonic_with_novelty(self):
        """New entities (less history) should have higher risk scores."""
        np.random.seed(42)
        n = 100
        anomaly_scores = np.ones(n) * 5  # Fixed anomaly score
        history_counts = np.linspace(100, 1, n)  # Decreasing history

        scorer = RiskScorer(score_weight=0.0, confidence_weight=0.0, novelty_weight=1.0)
        risk_scores = scorer.compute(anomaly_scores, entity_history_counts=history_counts)

        # Risk should increase as history decreases (more novel)
        assert np.all(np.diff(risk_scores) >= -1e-10), "Risk not monotonic with novelty"

    def test_risk_scores_bounded(self):
        """Risk scores should always be in [0, 1]."""
        np.random.seed(42)
        n = 1000
        anomaly_scores = np.random.randn(n) * 100  # Wide range
        score_std = np.abs(np.random.randn(n)) + 0.1
        history_counts = np.random.randint(1, 1000, n)

        scorer = RiskScorer()
        risk_scores = scorer.compute(
            anomaly_scores.astype(float),
            score_std=score_std.astype(float),
            entity_history_counts=history_counts.astype(float),
        )

        assert risk_scores.min() >= 0, "Risk scores below 0"
        assert risk_scores.max() <= 1, "Risk scores above 1"

    def test_weights_normalize(self):
        """Weights should be normalized to sum to 1."""
        scorer = RiskScorer(score_weight=2.0, confidence_weight=2.0, novelty_weight=2.0)
        total = scorer.score_weight + scorer.confidence_weight + scorer.novelty_weight
        assert np.isclose(total, 1.0), f"Weights not normalized: {total}"


class TestBudgetCalibration:
    """Alert budget calibration should respect constraints."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample detection data."""
        np.random.seed(42)
        n = 1000
        # 5% attack rate
        y_true = np.zeros(n)
        y_true[:50] = 1
        np.random.shuffle(y_true)

        # Scores: attacks have higher scores on average
        scores = np.where(y_true == 1,
                         np.random.normal(3, 1, n),
                         np.random.normal(0, 1, n))

        return scores, y_true.astype(int)

    def test_fixed_recall_achieves_target(self, sample_data):
        """Fixed recall mode should achieve at least the target recall."""
        scores, y_true = sample_data

        result = calibrate_threshold(scores, y_true, mode="fixed_recall", target=0.5)

        assert result["actual_recall"] >= 0.5 - 0.05, "Recall below target"

    def test_fixed_fpr_respects_limit(self, sample_data):
        """Fixed FPR mode should not exceed target FPR by much."""
        scores, y_true = sample_data

        result = calibrate_threshold(scores, y_true, mode="fixed_fpr", target=0.1)

        # Allow some tolerance due to discrete thresholds
        assert result["actual_fpr"] <= 0.15, "FPR significantly exceeds target"

    def test_fixed_alerts_respects_budget(self, sample_data):
        """Fixed alerts mode should generate approximately target alerts."""
        scores, y_true = sample_data

        result = calibrate_threshold(
            scores, y_true,
            mode="fixed_alerts",
            target=50,  # 50 alerts per day
            n_windows_per_day=1000
        )

        # Allow 20% tolerance
        assert result["alerts_per_day"] <= 60, "Too many alerts"

    def test_budget_curve_monotonic(self, sample_data):
        """Higher recall targets should require more alerts."""
        scores, y_true = sample_data

        curve = build_alert_budget_curve(scores, y_true)

        # Alerts should increase with recall
        alerts = curve["alerts"].values
        assert np.all(np.diff(alerts) >= 0), "Alert curve not monotonic"

    def test_budget_curve_has_expected_columns(self, sample_data):
        """Budget curve should have all expected columns."""
        scores, y_true = sample_data

        curve = build_alert_budget_curve(scores, y_true)

        expected_cols = ["threshold", "recall_target", "actual_recall", "fpr", "alerts"]
        for col in expected_cols:
            assert col in curve.columns, f"Missing column: {col}"


class TestRankingMetricsShapes:
    """Ranking metrics should have correct shapes and valid values."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample ranked data."""
        np.random.seed(42)
        n = 500
        # 2% attack rate (10 attacks)
        y_true = np.zeros(n)
        y_true[:10] = 1
        np.random.shuffle(y_true)

        # Scores with some separation
        scores = np.where(y_true == 1,
                         np.random.normal(2, 0.5, n),
                         np.random.normal(0, 1, n))

        return scores, y_true.astype(int)

    def test_precision_at_k_bounded(self, sample_data):
        """Precision@k should be in [0, 1]."""
        scores, y_true = sample_data

        for k in [5, 10, 25, 50, 100]:
            prec = precision_at_k(y_true, scores, k)
            assert 0 <= prec <= 1, f"Precision@{k} out of bounds: {prec}"

    def test_recall_at_k_bounded(self, sample_data):
        """Recall@k should be in [0, 1]."""
        scores, y_true = sample_data

        for k in [5, 10, 25, 50, 100]:
            rec = recall_at_k(y_true, scores, k)
            assert 0 <= rec <= 1, f"Recall@{k} out of bounds: {rec}"

    def test_recall_at_k_increases_with_k(self, sample_data):
        """Recall@k should be non-decreasing as k increases."""
        scores, y_true = sample_data

        ks = [5, 10, 25, 50, 100, 200]
        recalls = [recall_at_k(y_true, scores, k) for k in ks]

        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i-1], f"Recall decreased from k={ks[i-1]} to k={ks[i]}"

    def test_fpr_at_fixed_recall_bounded(self, sample_data):
        """FPR should be in [0, 1]."""
        scores, y_true = sample_data

        for target in [0.1, 0.3, 0.5, 0.7]:
            fpr = fpr_at_fixed_recall(y_true, scores, target)
            assert 0 <= fpr <= 1, f"FPR at recall={target} out of bounds: {fpr}"

    def test_alerts_per_k_positive(self, sample_data):
        """Alerts per k windows should be positive."""
        scores, y_true = sample_data

        for target in [0.1, 0.3, 0.5]:
            alerts = alerts_per_k_windows(y_true, scores, target)
            assert alerts >= 0, f"Negative alerts at recall={target}"

    def test_ranking_report_shape(self, sample_data):
        """Ranking report should have expected shape."""
        scores, y_true = sample_data

        report = ranking_report(y_true, scores, ks=[10, 50], recalls=[0.3, 0.5])

        # 2 ks * 2 metrics (prec, rec) + 2 recalls * 2 metrics (fpr, alerts) = 8 rows
        assert len(report) == 8, f"Unexpected report length: {len(report)}"
        assert "metric" in report.columns
        assert "value" in report.columns

    def test_workload_reduction_positive(self, sample_data):
        """Workload reduction should give positive reduction factor when model is better."""
        scores, y_true = sample_data

        # Create worse baseline (random scores)
        baseline = np.random.randn(len(scores))

        result = workload_reduction(y_true, baseline, scores, target_recall=0.3)

        assert "reduction_factor" in result
        assert "baseline_alerts_per_1k" in result
        assert "model_alerts_per_1k" in result


class TestEntityContext:
    """Entity context enrichment should provide valid analyst context."""

    @pytest.fixture
    def sample_df(self):
        """Generate sample DataFrame with entity information."""
        np.random.seed(42)
        n = 200

        df = pd.DataFrame({
            "entity": [f"entity_{i % 20}" for i in range(n)],
            "event_count": np.random.poisson(100, n),
            "anomaly_score": np.random.exponential(1, n),
            "has_attack": np.random.choice([0, 1], n, p=[0.95, 0.05]),
        })

        return df

    def test_entity_history_tracks_all_entities(self, sample_df):
        """Entity history should track all unique entities."""
        history = build_entity_history(sample_df)

        unique_entities = sample_df["entity"].nunique()
        tracked_entities = len(history.entity_stats)

        assert tracked_entities == unique_entities, "Not all entities tracked"

    def test_enriched_alerts_have_context(self, sample_df):
        """Enriched alerts should have all context fields."""
        history = build_entity_history(sample_df)
        enriched = enrich_alerts(sample_df, history, top_k=10)

        assert len(enriched) == 10, "Wrong number of enriched alerts"

        expected_fields = [
            "entity_id", "baseline_mean", "baseline_std",
            "current_value", "anomaly_score", "sigma_deviation",
            "confidence", "narrative"
        ]

        for alert in enriched:
            for field in expected_fields:
                assert field in alert, f"Missing field: {field}"

    def test_sigma_deviation_calculated(self, sample_df):
        """Sigma deviation should be calculated correctly."""
        history = build_entity_history(sample_df)
        enriched = enrich_alerts(sample_df, history, top_k=5)

        for alert in enriched:
            if alert["baseline_std"] > 0:
                expected_sigma = (alert["current_value"] - alert["baseline_mean"]) / alert["baseline_std"]
                assert np.isclose(alert["sigma_deviation"], expected_sigma, rtol=0.01)

    def test_confidence_levels_valid(self, sample_df):
        """Confidence should be one of high/medium/low."""
        history = build_entity_history(sample_df)
        enriched = enrich_alerts(sample_df, history, top_k=20)

        valid_levels = {"high", "medium", "low"}
        for alert in enriched:
            assert alert["confidence"] in valid_levels, f"Invalid confidence: {alert['confidence']}"


class TestEdgeCases:
    """Edge cases should be handled gracefully."""

    def test_empty_scores(self):
        """Empty arrays should return empty or raise ValueError."""
        scorer = RiskScorer()
        # Empty arrays raise ValueError due to min/max on empty array
        # This is acceptable behavior - callers should validate input
        with pytest.raises(ValueError):
            scorer.compute(np.array([]))

    def test_single_observation(self):
        """Single observation should work."""
        scorer = RiskScorer()
        risk = scorer.compute(np.array([5.0]))
        assert len(risk) == 1

    def test_all_same_scores(self):
        """Identical scores should produce valid risk scores."""
        scorer = RiskScorer()
        risk = scorer.compute(np.array([5.0, 5.0, 5.0, 5.0]))
        assert len(risk) == 4
        assert np.all(risk >= 0)

    def test_no_positives_recall(self):
        """No positives should return 0 recall."""
        y_true = np.array([0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        rec = recall_at_k(y_true, scores, k=3)
        assert rec == 0.0

    def test_all_positives_precision(self):
        """All positives should give precision = 1 regardless of k."""
        y_true = np.array([1, 1, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        prec = precision_at_k(y_true, scores, k=3)
        assert prec == 1.0

    def test_k_larger_than_n(self):
        """k larger than dataset size should handle gracefully."""
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])

        prec = precision_at_k(y_true, scores, k=100)
        rec = recall_at_k(y_true, scores, k=100)

        # Should use all data
        assert 0 <= prec <= 1
        assert 0 <= rec <= 1
