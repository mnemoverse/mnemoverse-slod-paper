"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from slod.utils.metrics import (
    adjusted_rand_index,
    hit_at_l,
    kendall_tau,
    mean_reciprocal_rank,
    variation_of_information,
)


class TestARI:
    """Test Adjusted Rand Index."""

    def test_perfect_agreement(self):
        """Identical labels → ARI = 1.0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)

    def test_random_close_to_zero(self):
        """Random permutation → ARI near 0."""
        rng = np.random.RandomState(42)
        true_labels = np.array([0] * 50 + [1] * 50)
        pred_labels = rng.permutation(true_labels)
        ari = adjusted_rand_index(true_labels, pred_labels)
        assert abs(ari) < 0.3  # Should be near 0


class TestVI:
    """Test Variation of Information."""

    def test_identical_partitions_zero(self):
        """Same partition → VI = 0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        vi = variation_of_information(labels, labels)
        assert vi == pytest.approx(0.0, abs=1e-10)

    def test_different_partitions_positive(self):
        """Different partitions → VI > 0."""
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_labels = np.array([0, 0, 1, 1, 1, 0])
        vi = variation_of_information(true_labels, pred_labels)
        assert vi > 0

    def test_empty_labels(self):
        """Empty input → VI = 0."""
        vi = variation_of_information(np.array([]), np.array([]))
        assert vi == 0.0


class TestKendallTau:
    """Test Kendall rank correlation."""

    def test_perfect_positive(self):
        """Monotonically increasing → τ = 1.0."""
        sigmas = np.array([1.0, 2.0, 3.0, 4.0])
        heights = np.array([1.0, 2.0, 3.0, 4.0])
        assert kendall_tau(sigmas, heights) == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Monotonically decreasing → τ = -1.0."""
        sigmas = np.array([1.0, 2.0, 3.0, 4.0])
        heights = np.array([4.0, 3.0, 2.0, 1.0])
        assert kendall_tau(sigmas, heights) == pytest.approx(-1.0)

    def test_too_few_points(self):
        """< 2 data points → 0.0."""
        assert kendall_tau(np.array([1.0]), np.array([1.0])) == 0.0
        assert kendall_tau(np.array([]), np.array([])) == 0.0


class TestHitAtL:
    """Test Hit@L metric."""

    def test_exact_match(self):
        """All detected depths exactly match true → Hit@0 = 1.0."""
        detected = np.array([3.0, 7.0, 12.0])
        true = np.array([3.0, 7.0, 12.0])
        assert hit_at_l(detected, true, hops=0) == pytest.approx(1.0)

    def test_within_tolerance(self):
        """Detected within 1 hop of true → Hit@1 = 1.0."""
        detected = np.array([3.0, 8.0, 11.0])
        true = np.array([3.0, 7.0, 12.0])
        assert hit_at_l(detected, true, hops=1) == pytest.approx(1.0)

    def test_partial_match(self):
        """Only some detected within tolerance."""
        detected = np.array([3.0, 20.0])
        true = np.array([3.0, 7.0])
        assert hit_at_l(detected, true, hops=1) == pytest.approx(0.5)

    def test_empty(self):
        """Empty detected → 0.0."""
        assert hit_at_l(np.array([]), np.array([3.0]), hops=1) == 0.0


class TestMRR:
    """Test Mean Reciprocal Rank."""

    def test_perfect_rank(self):
        """True ancestor at rank 1 → MRR = 1.0."""
        true = [[5], [10]]
        pred = [[5, 3, 1], [10, 8, 6]]
        assert mean_reciprocal_rank(true, pred) == pytest.approx(1.0)

    def test_rank_two(self):
        """True ancestor at rank 2 → RR = 0.5."""
        true = [[5]]
        pred = [[3, 5, 1]]
        assert mean_reciprocal_rank(true, pred) == pytest.approx(0.5)

    def test_not_found(self):
        """True ancestor not in predicted → RR = 0."""
        true = [[99]]
        pred = [[1, 2, 3]]
        assert mean_reciprocal_rank(true, pred) == pytest.approx(0.0)

    def test_empty(self):
        """Empty inputs → 0.0."""
        assert mean_reciprocal_rank([], []) == 0.0
