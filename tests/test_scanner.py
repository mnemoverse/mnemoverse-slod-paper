"""Tests for BoundaryScan (Algorithm 2).

Smoke tests to verify the pipeline runs end-to-end and returns
correctly shaped results.
"""

from __future__ import annotations

import numpy as np

from slod.boundary.scanner import BoundaryScanResult, boundary_scan


class TestBoundaryScan:
    """Test BoundaryScan pipeline."""

    def test_returns_correct_type(self, ball_points_with_laplacian):
        """boundary_scan returns a BoundaryScanResult."""
        points, laplacian = ball_points_with_laplacian
        result = boundary_scan(
            points, laplacian, focus_idx=0, sigma_grid=np.logspace(-1, 1, 20)
        )
        assert isinstance(result, BoundaryScanResult)

    def test_output_shapes(self, ball_points_with_laplacian):
        """Output arrays have consistent shapes."""
        points, laplacian = ball_points_with_laplacian
        grid = np.logspace(-1, 1, 20)
        result = boundary_scan(points, laplacian, focus_idx=0, sigma_grid=grid)

        assert result.sigma_grid.shape == (20,)
        assert result.scores.shape == (19,)  # T-1 between-scale indicators
        assert result.velocity.shape == (19,)
        assert result.divergence.shape == (19,)
        assert result.churn.shape == (19,)

    def test_scores_are_finite(self, ball_points_with_laplacian):
        """All composite scores must be finite (no NaN/inf)."""
        points, laplacian = ball_points_with_laplacian
        result = boundary_scan(
            points, laplacian, focus_idx=0, sigma_grid=np.logspace(-1, 1, 20)
        )
        assert np.all(np.isfinite(result.scores)), f"Non-finite scores: {result.scores}"

    def test_indicators_nonnegative(self, ball_points_with_laplacian):
        """Raw indicators (V, D_w, C_k) are non-negative."""
        points, laplacian = ball_points_with_laplacian
        result = boundary_scan(
            points, laplacian, focus_idx=0, sigma_grid=np.logspace(-1, 1, 20)
        )
        assert np.all(result.velocity >= 0), "Velocity has negative values"
        assert np.all(result.divergence >= 0), "Divergence has negative values"
        assert np.all(result.churn >= 0), "Churn has negative values"

    def test_peaks_within_bounds(self, ball_points_with_laplacian):
        """All peak indices are valid indices into scores array."""
        points, laplacian = ball_points_with_laplacian
        result = boundary_scan(
            points, laplacian, focus_idx=0, sigma_grid=np.logspace(-1, 1, 20)
        )
        for peak_idx in result.peaks:
            assert 0 <= peak_idx < len(result.scores)

    def test_minimal_sigma_grid(self, ball_points_with_laplacian):
        """Pipeline handles very small sigma grid (3 points) without crash."""
        points, laplacian = ball_points_with_laplacian
        result = boundary_scan(
            points, laplacian, focus_idx=0, sigma_grid=np.array([0.1, 1.0, 10.0])
        )
        assert result.scores.shape == (2,)
        assert np.all(np.isfinite(result.scores))

    def test_fallback_peak_when_none_above_threshold(self, ball_points_with_laplacian):
        """ADR-004: when no peaks exceed threshold, return highest local max."""
        points, laplacian = ball_points_with_laplacian
        # Very high alpha → threshold so high no peak passes
        result = boundary_scan(
            points, laplacian, focus_idx=0,
            sigma_grid=np.logspace(-1, 1, 20),
            peak_alpha=100.0,
        )
        # ADR-004 fallback should still return at most 1 peak
        assert len(result.peaks) <= 1
        for peak_idx in result.peaks:
            assert 0 <= peak_idx < len(result.scores)
