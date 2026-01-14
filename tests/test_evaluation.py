"""Tests for evaluation metrics module."""

import numpy as np
import pytest

from bsad.evaluation import (
    compute_all_metrics,
    compute_pr_auc,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_roc_auc,
    format_metrics_report,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])

        recall = compute_recall_at_k(y_true, scores, k=2)
        assert recall == 1.0

    def test_zero_recall(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7])

        recall = compute_recall_at_k(y_true, scores, k=2)
        assert recall == 0.0

    def test_partial_recall(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.2, 0.8, 0.1, 0.3])

        recall = compute_recall_at_k(y_true, scores, k=2)
        assert recall == 0.5

    def test_no_positives(self):
        y_true = np.array([0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        recall = compute_recall_at_k(y_true, scores, k=2)
        assert recall == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])

        precision = compute_precision_at_k(y_true, scores, k=2)
        assert precision == 1.0

    def test_zero_precision(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7])

        precision = compute_precision_at_k(y_true, scores, k=2)
        assert precision == 0.0

    def test_partial_precision(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.2, 0.8, 0.1, 0.3])

        precision = compute_precision_at_k(y_true, scores, k=2)
        assert precision == 0.5


class TestPRAUC:
    def test_perfect_separation(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        pr_auc = compute_pr_auc(y_true, scores)
        assert pr_auc == 1.0

    def test_random_baseline(self):
        np.random.seed(42)
        y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 10% positive
        scores = np.random.rand(10)

        pr_auc = compute_pr_auc(y_true, scores)
        # Random should be around baseline (proportion of positives)
        assert 0 < pr_auc < 1


class TestROCAUC:
    def test_perfect_separation(self):
        y_true = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        roc_auc = compute_roc_auc(y_true, scores)
        assert roc_auc == 1.0


class TestAllMetrics:
    def test_returns_all_keys(self):
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        metrics = compute_all_metrics(y_true, scores, k_values=[5, 10])

        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "n_observations" in metrics
        assert "n_positives" in metrics
        assert "recall_at_5" in metrics
        assert "precision_at_5" in metrics

    def test_observation_counts(self):
        y_true = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])

        metrics = compute_all_metrics(y_true, scores)

        assert metrics["n_observations"] == 5
        assert metrics["n_positives"] == 2
        assert metrics["attack_rate"] == 0.4


class TestMetricsReport:
    def test_format_report(self):
        metrics = {
            "pr_auc": 0.85,
            "roc_auc": 0.92,
            "n_observations": 1000,
            "n_positives": 50,
            "attack_rate": 0.05,
            "recall_at_10": 0.4,
            "recall_at_50": 0.8,
            "precision_at_10": 0.2,
            "precision_at_50": 0.08,
        }

        report = format_metrics_report(metrics)

        assert "PR-AUC" in report
        assert "0.85" in report
        assert "Recall@" in report
