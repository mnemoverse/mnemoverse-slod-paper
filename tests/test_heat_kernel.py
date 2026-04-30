"""Tests for heat kernel weight computation.

Tests validate properties that must hold for heat kernel weights
computed via graph Laplacian eigendecomposition (ADR-001).
"""

from __future__ import annotations

import numpy as np

from slod.core.heat_kernel import heat_kernel_weights, weights_from_eigen


class TestHeatKernelProperties:
    """Test mathematical properties of heat kernel weights."""

    def test_weights_sum_to_one(self, small_graph_laplacian):
        """Heat kernel weights must form a probability distribution."""
        weights = heat_kernel_weights(small_graph_laplacian, sigma=1.0, focus_idx=0)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_weights_nonnegative(self, small_graph_laplacian):
        """Heat kernel weights must be non-negative."""
        weights = heat_kernel_weights(small_graph_laplacian, sigma=1.0, focus_idx=0)
        assert np.all(weights >= 0.0)

    def test_small_sigma_concentrates(self, small_graph_laplacian):
        """Small sigma: weight concentrates on focus node (> 0.5)."""
        weights = heat_kernel_weights(small_graph_laplacian, sigma=0.01, focus_idx=5)
        assert weights[5] > 0.5, f"Focus weight {weights[5]} should be > 0.5 for small sigma"

    def test_large_sigma_spreads(self, small_graph_laplacian):
        """Large sigma: weight spreads uniformly (max weight < 0.1)."""
        weights = heat_kernel_weights(small_graph_laplacian, sigma=100.0, focus_idx=5)
        assert weights.max() < 0.1, f"Max weight {weights.max()} should be < 0.1 for large sigma"

    def test_weights_from_eigen_matches_full(self, small_graph_laplacian):
        """weights_from_eigen produces same result as heat_kernel_weights.

        Direct test for the factored-out eigendecomposition helper.
        """
        sigma, focus_idx = 1.0, 3
        weights_full, eigenvalues, eigenvectors = heat_kernel_weights(
            small_graph_laplacian, sigma=sigma, focus_idx=focus_idx,
            return_eigen=True,
        )
        weights_eigen = weights_from_eigen(eigenvalues, eigenvectors, sigma, focus_idx)
        np.testing.assert_allclose(weights_eigen, weights_full, atol=1e-10)

    def test_return_eigen_shapes(self, small_graph_laplacian):
        """return_eigen=True returns correctly shaped eigenvalues/vectors."""
        weights, eigenvalues, eigenvectors = heat_kernel_weights(
            small_graph_laplacian, sigma=1.0, focus_idx=0, k_eigs=10,
            return_eigen=True,
        )
        n = small_graph_laplacian.shape[0]
        assert weights.shape == (n,)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (n, 10)
        # Eigenvalues should be sorted and non-negative (Laplacian is PSD)
        assert np.all(eigenvalues >= -1e-10)
        assert np.all(np.diff(eigenvalues) >= -1e-10)
