"""Smoke tests for Frechet mean computation.

Tests validate properties that must hold for any correct
Frechet mean implementation on the Poincare ball.
"""

from __future__ import annotations

import numpy as np
import pytest

from slod.core.frechet_mean import frechet_mean


class TestFrechetMeanProperties:
    """Test mathematical properties of the Frechet mean."""

    def test_mean_of_single_point_is_itself(self, rng):
        """Frechet mean of a single point is that point."""
        point = rng.randn(5).astype(np.float64) * 0.5
        point = point / (np.linalg.norm(point) + 1e-6) * 0.5
        weights = np.array([1.0])
        mean = frechet_mean(points=point[np.newaxis], weights=weights)
        np.testing.assert_allclose(mean, point, atol=1e-5)

    def test_mean_of_origin_symmetric_points(self):
        """Frechet mean of points symmetric around origin should be near origin."""
        r = 0.5
        points = np.array(
            [
                [r, 0],
                [-r, 0],
                [0, r],
                [0, -r],
            ],
            dtype=np.float64,
        )
        weights = np.ones(4) / 4
        mean = frechet_mean(points, weights)
        np.testing.assert_allclose(mean, np.zeros(2), atol=1e-4)

    def test_mean_inside_ball(self, points_2d_ball):
        """Frechet mean must lie inside the Poincare ball."""
        weights = np.ones(len(points_2d_ball)) / len(points_2d_ball)
        mean = frechet_mean(points_2d_ball, weights)
        assert np.linalg.norm(mean) < 1.0

    def test_mean_is_unique(self, points_2d_ball):
        """Two runs with same data produce the same result (Hadamard uniqueness)."""
        weights = np.ones(len(points_2d_ball)) / len(points_2d_ball)
        mean1 = frechet_mean(points_2d_ball, weights)
        mean2 = frechet_mean(points_2d_ball, weights)
        np.testing.assert_allclose(mean1, mean2, atol=1e-6)

    def test_weights_must_be_normalized(self, points_2d_ball):
        """Frechet mean with equal weights produces valid result."""
        n = len(points_2d_ball)
        weights = np.ones(n) / n
        assert weights.sum() == pytest.approx(1.0)
        mean = frechet_mean(points_2d_ball, weights)
        assert np.all(np.isfinite(mean))
        assert np.linalg.norm(mean) < 1.0

    def test_unnormalized_weights_auto_normalized(self, points_2d_ball):
        """Unnormalized weights produce same result as normalized (auto-normalization)."""
        n = len(points_2d_ball)
        weights_norm = np.ones(n) / n
        weights_unnorm = np.ones(n) * 5.0  # sum = 5*n, not 1

        mean_norm = frechet_mean(points_2d_ball, weights_norm)
        mean_unnorm = frechet_mean(points_2d_ball, weights_unnorm)
        np.testing.assert_allclose(mean_norm, mean_unnorm, atol=1e-6)

    def test_negative_weights_rejected(self):
        """Negative weights raise ValueError."""
        points = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
        weights = np.array([1.0, -0.5])
        with pytest.raises(ValueError, match="non-negative"):
            frechet_mean(points, weights)

    def test_wrong_shape_weights_rejected(self):
        """Weights with wrong length raise ValueError."""
        points = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64)
        weights = np.array([1.0, 0.5, 0.3])  # 3 weights for 2 points
        with pytest.raises(ValueError, match="length"):
            frechet_mean(points, weights)
