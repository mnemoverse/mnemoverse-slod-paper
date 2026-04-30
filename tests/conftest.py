"""Shared test fixtures for SLoD experiments."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from scipy.spatial.distance import pdist, squareform


@pytest.fixture
def rng():
    """Deterministic random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def points_2d_ball(rng):
    """10 random points inside the 2D Poincare ball (||x|| < 0.9)."""
    pts = rng.randn(10, 2).astype(np.float32)
    # Normalize to lie within the ball (max norm 0.9)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + 1e-6) * 0.9 * rng.uniform(0.1, 1.0, size=(10, 1))
    return pts


@pytest.fixture
def points_10d_ball(rng):
    """50 random points inside the 10D Poincare ball."""
    pts = rng.randn(50, 10).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + 1e-6) * 0.9 * rng.uniform(0.1, 1.0, size=(50, 1))
    return pts


@pytest.fixture
def origin_2d():
    """Origin of the 2D Poincare ball."""
    return np.zeros(2, dtype=np.float32)


@pytest.fixture
def three_level_tree():
    """Simple 3-level tree structure for testing boundary detection.

    Level 0: 1 root
    Level 1: 4 children
    Level 2: 16 leaves (4 per child)
    Total: 21 nodes
    """
    return {
        "n_levels": 3,
        "branching": [4, 4],
        "n_nodes": 21,
    }


@pytest.fixture
def small_graph_laplacian():
    """Graph Laplacian for a 20-node ring graph.

    Ring graph: each node connected to its two neighbors.
    L = D - A where D=2*I (degree 2 for all nodes).
    """
    n = 20
    row = []
    col = []
    for i in range(n):
        j = (i + 1) % n
        row.extend([i, j])
        col.extend([j, i])
    data = np.ones(len(row), dtype=np.float64)
    adjacency = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
    degree = scipy.sparse.diags(np.array(adjacency.sum(axis=1)).flatten())
    laplacian = degree - adjacency
    return laplacian


@pytest.fixture
def ball_points_with_laplacian(rng):
    """20 points in B^2 with kNN graph Laplacian (k=5).

    Returns (points, laplacian) tuple.
    """
    n, d, k = 20, 2, 5
    pts = rng.randn(n, d).astype(np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + 1e-6) * 0.8 * rng.uniform(0.1, 1.0, size=(n, 1))

    # Build kNN adjacency (symmetric)
    dists = squareform(pdist(pts))
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        neighbors = np.argsort(dists[i])[1 : k + 1]
        adjacency[i, neighbors] = 1.0
        adjacency[neighbors, i] = 1.0  # symmetrize

    adj_sparse = scipy.sparse.csr_matrix(adjacency)
    degree = scipy.sparse.diags(np.array(adj_sparse.sum(axis=1)).flatten())
    laplacian = degree - adj_sparse
    return pts, laplacian
