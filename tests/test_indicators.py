"""Tests for boundary indicator functions.

Tests validate properties from paper Definitions 4-6:
V(sigma), D_w(sigma), C_k(sigma).
"""

from __future__ import annotations

import numpy as np
import pytest

from slod.boundary.indicators import (
    neighborhood_churn,
    representation_velocity,
    weight_divergence,
)


class TestRepresentationVelocity:
    """Test V(sigma) = d_H(phi_prev, phi_curr) / delta_sigma."""

    def test_identical_points_zero_velocity(self):
        """Same point at both scales → velocity = 0."""
        p = np.array([0.1, 0.2], dtype=np.float64)
        v = representation_velocity(p, p, delta_sigma=0.1)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_distant_points_positive_velocity(self):
        """Different points → positive velocity."""
        p1 = np.array([0.1, 0.0], dtype=np.float64)
        p2 = np.array([0.5, 0.0], dtype=np.float64)
        v = representation_velocity(p1, p2, delta_sigma=0.1)
        assert v > 0

    def test_zero_delta_raises(self):
        """delta_sigma <= 0 raises ValueError."""
        p = np.array([0.1, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="positive"):
            representation_velocity(p, p, delta_sigma=0.0)


class TestWeightDivergence:
    """Test D_w(sigma) = JSD(w_prev || w_curr)."""

    def test_identical_weights_zero(self):
        """Same distribution → JSD = 0."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        d = weight_divergence(w, w)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_different_weights_positive(self):
        """Different distributions → JSD > 0."""
        w1 = np.array([1.0, 0.0, 0.0, 0.0])
        w2 = np.array([0.0, 0.0, 0.0, 1.0])
        d = weight_divergence(w1, w2)
        assert d > 0

    def test_bounded_by_ln2(self):
        """JSD is bounded by ln(2)."""
        w1 = np.array([1.0, 0.0, 0.0, 0.0])
        w2 = np.array([0.0, 0.0, 0.0, 1.0])
        d = weight_divergence(w1, w2)
        assert d <= np.log(2) + 1e-6


class TestNeighborhoodChurn:
    """Test C_k(sigma) = Jaccard distance of kNN sets."""

    def test_same_point_zero_churn(self):
        """Same SLoD point → same neighbors → churn = 0."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.3
        phi = np.array([0.1, 0.0], dtype=np.float64)
        c = neighborhood_churn(phi, phi, pts, k=5)
        assert c == pytest.approx(0.0, abs=1e-10)

    def test_distant_points_positive_churn(self):
        """Very different SLoD points → some churn."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.3
        phi1 = np.array([0.5, 0.0], dtype=np.float64)
        phi2 = np.array([-0.5, 0.0], dtype=np.float64)
        c = neighborhood_churn(phi1, phi2, pts, k=5)
        assert c >= 0.0
        assert c <= 1.0

    def test_churn_in_range(self):
        """Churn is always in [0, 1]."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.5
        phi1 = np.array([0.3, 0.3], dtype=np.float64)
        phi2 = np.array([-0.3, -0.3], dtype=np.float64)
        c = neighborhood_churn(phi1, phi2, pts, k=5)
        assert 0.0 <= c <= 1.0
