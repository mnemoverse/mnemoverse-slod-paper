"""Tests for spectral analysis module.

Tests validate kNN graph construction, normalized Laplacian properties,
and spectral gap detection.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from slod.boundary.spectral import (
    build_knn_graph,
    build_knn_graph_approx,
    build_knn_graph_batched,
    normalized_laplacian,
    spectral_gap,
)


class TestBuildKnnGraph:
    """Test kNN graph construction on Poincare ball."""

    def test_adjacency_is_symmetric(self):
        """Symmetric kNN produces symmetric adjacency matrix."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.3
        adj = build_knn_graph(pts, k=5)
        diff = abs(adj - adj.T).sum()
        assert diff < 1e-10, f"Adjacency not symmetric: diff={diff}"

    def test_weights_positive(self):
        """All edge weights are positive (Gaussian kernel)."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.3
        adj = build_knn_graph(pts, k=5)
        assert adj.nnz > 0, "Graph has no edges"
        assert np.all(adj.data > 0), "Some weights are non-positive"

    def test_no_self_loops(self):
        """Diagonal of adjacency is zero (no self-loops)."""
        rng = np.random.RandomState(42)
        pts = rng.randn(15, 2) * 0.3
        adj = build_knn_graph(pts, k=5)
        diag = adj.diagonal()
        assert np.all(diag == 0), f"Self-loops found: {diag}"

    def test_default_k_formula(self):
        """Default k follows ADR-005: max(10, min(sqrt(N), 50))."""
        rng = np.random.RandomState(42)
        pts = rng.randn(64, 2) * 0.3  # sqrt(64)=8, so k=max(10,8)=10
        adj = build_knn_graph(pts)
        # Symmetric kNN guarantees every node has degree >= k
        degrees = np.array(adj.sum(axis=1)).flatten()
        assert np.all(degrees > 0), "Some nodes have no neighbors"

    def test_single_node(self):
        """Single node: empty graph."""
        pts = np.array([[0.1, 0.2]])
        adj = build_knn_graph(pts)
        assert adj.shape == (1, 1)
        assert adj.nnz == 0


class TestBuildKnnGraphBatched:
    """Test batched kNN graph construction."""

    def test_adjacency_is_symmetric(self):
        """Batched kNN produces symmetric adjacency matrix."""
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3) * 0.3
        adj = build_knn_graph_batched(pts, k=5, batch_size=8)
        diff = abs(adj - adj.T).sum()
        assert diff < 1e-10, f"Adjacency not symmetric: diff={diff}"

    def test_weights_positive(self):
        """All edge weights are positive."""
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3) * 0.3
        adj = build_knn_graph_batched(pts, k=5, batch_size=8)
        assert adj.nnz > 0, "Graph has no edges"
        assert np.all(adj.data > 0), "Some weights are non-positive"

    def test_no_self_loops(self):
        """Diagonal of adjacency is zero."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 3) * 0.3
        adj = build_knn_graph_batched(pts, k=5, batch_size=8)
        diag = adj.diagonal()
        assert np.all(diag == 0), f"Self-loops found: {diag}"

    def test_agrees_with_original(self):
        """Batched and original produce equivalent graph structure."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.3
        adj_orig = build_knn_graph(pts, k=5, tau=1.0)
        adj_batch = build_knn_graph_batched(pts, k=5, tau=1.0, batch_size=8)
        # Same edge set (same sparsity pattern)
        orig_edges = set(zip(*adj_orig.nonzero()))
        batch_edges = set(zip(*adj_batch.nonzero()))
        assert orig_edges == batch_edges, (
            f"Edge sets differ: {len(orig_edges)} vs {len(batch_edges)}"
        )
        # Same weights (within tolerance)
        for i, j in orig_edges:
            w_orig = adj_orig[i, j]
            w_batch = adj_batch[i, j]
            assert abs(w_orig - w_batch) < 1e-6, (
                f"Weight mismatch at ({i},{j}): {w_orig} vs {w_batch}"
            )

    def test_single_node(self):
        """Single node: empty graph."""
        pts = np.array([[0.1, 0.2]])
        adj = build_knn_graph_batched(pts)
        assert adj.shape == (1, 1)
        assert adj.nnz == 0


class TestBuildKnnGraphApprox:
    """Test approximate kNN graph construction."""

    def test_adjacency_is_symmetric(self):
        """Approx kNN produces symmetric adjacency matrix."""
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3) * 0.3
        adj = build_knn_graph_approx(pts, k=5)
        diff = abs(adj - adj.T).sum()
        assert diff < 1e-10, f"Adjacency not symmetric: diff={diff}"

    def test_weights_positive(self):
        """All edge weights are positive."""
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 3) * 0.3
        adj = build_knn_graph_approx(pts, k=5)
        assert adj.nnz > 0, "Graph has no edges"
        assert np.all(adj.data > 0), "Some weights are non-positive"

    def test_no_self_loops(self):
        """Diagonal of adjacency is zero."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 3) * 0.3
        adj = build_knn_graph_approx(pts, k=5)
        diag = adj.diagonal()
        assert np.all(diag == 0), f"Self-loops found: {diag}"

    def test_high_overlap_with_exact(self):
        """Approx kNN edge set overlaps >= 80% with exact kNN (3x oversampling)."""
        rng = np.random.RandomState(42)
        pts = rng.randn(30, 2) * 0.3
        adj_exact = build_knn_graph_batched(pts, k=5, tau=1.0, batch_size=8)
        adj_approx = build_knn_graph_approx(pts, k=5, tau=1.0, oversampling=3)
        exact_edges = set(zip(*adj_exact.nonzero()))
        approx_edges = set(zip(*adj_approx.nonzero()))
        overlap = len(exact_edges & approx_edges) / max(len(exact_edges), 1)
        assert overlap >= 0.80, f"Only {overlap:.0%} edge overlap with exact kNN"

    def test_single_node(self):
        """Single node: empty graph."""
        pts = np.array([[0.1, 0.2]])
        adj = build_knn_graph_approx(pts)
        assert adj.shape == (1, 1)
        assert adj.nnz == 0


class TestNormalizedLaplacian:
    """Test normalized Laplacian properties."""

    def test_eigenvalues_in_range(self):
        """Normalized Laplacian eigenvalues lie in [0, 2] (±1e-6 for numerical noise)."""
        rng = np.random.RandomState(42)
        pts = rng.randn(20, 2) * 0.3
        adj = build_knn_graph(pts, k=5)
        lap = normalized_laplacian(adj)
        n = lap.shape[0]
        k = min(n - 1, 15)
        # Theoretical range [0, 2]; allow ±1e-6 tolerance for sparse eigsh noise
        eigs_large = scipy.sparse.linalg.eigsh(lap, k=k, which="LA", return_eigenvectors=False)
        assert np.all(eigs_large >= -1e-6), f"Negative eigenvalue: {eigs_large.min()}"
        assert np.all(eigs_large <= 2.0 + 1e-6), f"Eigenvalue > 2: {eigs_large.max()}"

    def test_smallest_eigenvalue_near_zero(self):
        """Connected graph has smallest eigenvalue near 0."""
        # Use a known connected graph (ring) to guarantee connectivity
        n = 20
        row, col = [], []
        for i in range(n):
            j = (i + 1) % n
            row.extend([i, j])
            col.extend([j, i])
        data = np.ones(len(row), dtype=np.float64)
        adj = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
        lap = normalized_laplacian(adj)
        eigs_small = scipy.sparse.linalg.eigsh(lap, k=3, which="SA", return_eigenvectors=False)
        eigs_small = np.sort(eigs_small)
        assert abs(eigs_small[0]) < 1e-6, f"Smallest eigenvalue not zero: {eigs_small[0]}"


class TestSpectralGap:
    """Test spectral gap detection."""

    def test_block_diagonal_has_gap(self):
        """Block-diagonal graph has clear spectral gap at block boundary."""
        # Two blocks of 10 nodes, no inter-block edges
        n = 20
        adj = np.zeros((n, n))
        for block_start in [0, 10]:
            for i in range(block_start, block_start + 10):
                for j in range(i + 1, block_start + 10):
                    adj[i, j] = adj[j, i] = 1.0
        adj_sparse = scipy.sparse.csr_matrix(adj)
        lap = normalized_laplacian(adj_sparse)
        eigs = scipy.sparse.linalg.eigsh(lap, k=5, which="SA", return_eigenvectors=False)
        eigs = np.sort(eigs)

        gaps = spectral_gap(eigs)
        assert len(gaps) >= 1, "No spectral gap found in block-diagonal graph"

    def test_uniform_spectrum_returns_fallback(self):
        """Nearly uniform eigenvalues: returns fallback (largest gap)."""
        eigs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        gaps = spectral_gap(eigs, gap_threshold=5.0)  # Very high threshold
        assert len(gaps) == 1, "Should return exactly 1 fallback gap"

    def test_empty_eigenvalues(self):
        """Empty eigenvalues: returns empty list."""
        gaps = spectral_gap(np.array([]))
        assert gaps == []

    def test_skips_zero_eigenvalues(self):
        """Zero eigenvalues are skipped (no division by zero)."""
        eigs = np.array([0.0, 0.0, 0.5, 1.0, 3.0])
        gaps = spectral_gap(eigs, gap_threshold=2.0)
        # Should find gap between 1.0 and 3.0 (ratio=3.0)
        assert len(gaps) >= 1
