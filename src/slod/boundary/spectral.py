"""Spectral analysis for scale boundary detection.

- Graph Laplacian construction from kNN graph on Poincare ball
- Partial eigendecomposition via scipy.sparse.linalg.eigsh
- Spectral gap detection: ratio r_k = lambda_{k+1} / lambda_k
- Candidate scale identification: sigma* ~ 1/lambda_k at gaps

Key insight (Proposition 3): JSD peaks near sigma* = 1/lambda_k
when there is a spectral gap ratio r_k > R.

See ADR-005: kNN construction strategy.
See ADR-007: spectral gap threshold R=2.0.
"""

from __future__ import annotations

import math

import geoopt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from slod.core.poincare import poincare_distance


def build_knn_graph(
    points: np.ndarray,
    k: int | None = None,
    tau: float | None = None,
    metric: str = "poincare",
    weighted: bool = True,
) -> scipy.sparse.csr_matrix:
    """Build a (weighted or binary) symmetric kNN graph.

    Symmetric (union) kNN: edge (i,j) exists iff i in kNN(j) OR j in kNN(i).
    This guarantees every node has degree >= k (von Luxburg 2007, §2.2).

    Args:
        points: Array of shape (N, d). Geometry of the points must match
            ``metric``; for ``metric="poincare"`` rows must lie in B^d.
        k: Number of neighbors. Default: max(10, min(int(sqrt(N)), 50)) (ADR-005).
        tau: Bandwidth for Gaussian weights. Default: median kNN distance.
            Ignored when ``weighted=False``.
        metric: Pairwise distance used for both neighbor selection and
            Gaussian weighting. ``"poincare"`` (default, matches SLoD main
            path) or ``"euclidean"`` (ablation baseline).
        weighted: If True (default), produce Gaussian-weighted adjacency
            ``W_ij = exp(-d^2/(2*tau^2))``. If False, produce a binary
            adjacency (Tier B ablation: strips the Gaussian weight and
            keeps only the topology).

    Returns:
        Sparse (N, N) symmetric adjacency matrix.
    """
    n = len(points)
    if n <= 1:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    if k is None:
        k = max(10, min(int(math.sqrt(n)), 50))
    k = min(k, n - 1)

    # Pairwise distance matrix under the chosen metric.
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    if metric == "poincare":
        for i in range(n):
            for j in range(i + 1, n):
                d = poincare_distance(points[i], points[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    elif metric == "euclidean":
        # Vectorised Euclidean distance; much faster and no geometry constraint.
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt((diffs ** 2).sum(axis=-1))
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'poincare' or 'euclidean'")

    # Find k nearest neighbors for each point
    knn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neighbors = np.argsort(dist_matrix[i])[1 : k + 1]
        knn_mask[i, neighbors] = True

    symmetric_mask = knn_mask | knn_mask.T

    if not weighted:
        # Binary adjacency: topology only.
        adjacency = symmetric_mask.astype(np.float64)
        return scipy.sparse.csr_matrix(adjacency)

    # Gaussian-weighted path (default, matches Exp 1/2).
    if tau is None:
        edge_dists = dist_matrix[symmetric_mask]
        tau = float(np.median(edge_dists)) if len(edge_dists) > 0 else 1.0
    tau = max(tau, 1e-10)

    adjacency = np.zeros((n, n), dtype=np.float64)
    rows, cols = np.where(symmetric_mask)
    for idx in range(len(rows)):
        i, j = rows[idx], cols[idx]
        adjacency[i, j] = math.exp(-dist_matrix[i, j] ** 2 / (2 * tau**2))
    return scipy.sparse.csr_matrix(adjacency)


def build_knn_graph_batched(
    points: np.ndarray,
    k: int | None = None,
    tau: float | None = None,
    batch_size: int = 512,
    verbose: bool = True,
) -> scipy.sparse.csr_matrix:
    """Build a weighted symmetric kNN graph using vectorized Poincare distance.

    Same semantics as ``build_knn_graph`` but computes pairwise distances in
    batched matrix operations via geoopt tensors.  Memory usage is
    O(batch_size * N) instead of O(N^2), making this suitable for N > 5000.

    Args:
        points: Array of shape (N, d), each row a point in B^d.
        k: Number of neighbors. Default: max(10, min(int(sqrt(N)), 50)).
        tau: Bandwidth for Gaussian weights. Default: median kNN distance.
        batch_size: Number of rows to process at once (controls peak memory).

    Returns:
        Sparse (N, N) symmetric weighted adjacency matrix.
    """
    n = len(points)
    if n <= 1:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    if k is None:
        k = max(10, min(int(math.sqrt(n)), 50))
    k = min(k, n - 1)

    ball = geoopt.PoincareBall(c=1.0)
    pts_t = torch.as_tensor(points, dtype=torch.float64)

    # --- Phase 1: batched kNN indices ---
    knn_indices = np.empty((n, k), dtype=np.int64)
    knn_dists = np.empty((n, k), dtype=np.float64)
    n_batches = (n + batch_size - 1) // batch_size

    for bi, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        # (batch, 1, d) vs (1, N, d) → (batch, N)
        batch_pts = pts_t[start:end].unsqueeze(1)
        all_pts = pts_t.unsqueeze(0)
        dists = ball.dist(batch_pts, all_pts)  # (batch, N)
        # Set self-distance to inf so it's never selected as a neighbor
        for i in range(end - start):
            dists[i, start + i] = float("inf")
        topk_dists, topk_idx = torch.topk(dists, k, dim=1, largest=False)
        knn_indices[start:end] = topk_idx.numpy()
        knn_dists[start:end] = topk_dists.numpy()
        if verbose and n > 5000 and (bi % max(1, n_batches // 5) == 0 or bi == n_batches - 1):
            print(f"    kNN batch {bi+1}/{n_batches} ({end}/{n} nodes)")

    # --- Phase 2: symmetric kNN mask → COO edges ---
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for i in range(n):
        neighbors = knn_indices[i]
        rows_list.append(np.full(k, i, dtype=np.int64))
        cols_list.append(neighbors)
    rows_arr = np.concatenate(rows_list)
    cols_arr = np.concatenate(cols_list)

    # Symmetrize: add reverse edges
    all_rows = np.concatenate([rows_arr, cols_arr])
    all_cols = np.concatenate([cols_arr, rows_arr])

    # Deduplicate using a sparse boolean matrix
    sym_mask = scipy.sparse.coo_matrix(
        (np.ones(len(all_rows), dtype=bool), (all_rows, all_cols)),
        shape=(n, n),
    ).tocsr()

    # --- Phase 3: compute tau from kNN distances if not provided ---
    if tau is None:
        tau = float(np.median(knn_dists))
    tau = max(tau, 1e-10)

    # --- Phase 4: compute weights for all symmetric edges ---
    sym_coo = sym_mask.tocoo()
    edge_rows = sym_coo.row
    edge_cols = sym_coo.col
    n_edges_total = len(edge_rows)
    if verbose and n > 5000:
        print(f"    Symmetrized: {n_edges_total} edges, computing weights...")

    # Compute distances for edges in batches
    weights = np.empty(n_edges_total, dtype=np.float64)
    for start in range(0, n_edges_total, batch_size * k):
        end = min(start + batch_size * k, n_edges_total)
        r_batch = edge_rows[start:end]
        c_batch = edge_cols[start:end]
        d = ball.dist(pts_t[r_batch], pts_t[c_batch]).numpy()
        weights[start:end] = np.exp(-d**2 / (2 * tau**2))

    adjacency = scipy.sparse.csr_matrix(
        (weights, (edge_rows, edge_cols)), shape=(n, n)
    )
    return adjacency


def build_knn_graph_approx(
    points: np.ndarray,
    k: int | None = None,
    tau: float | None = None,
    oversampling: int = 3,
    batch_size: int = 512,
    verbose: bool = True,
) -> scipy.sparse.csr_matrix:
    """Build kNN graph using tangent-space approximate neighbors + exact refinement.

    For N > 10K, the exact O(N^2) Poincare distance computation in
    ``build_knn_graph_batched`` becomes prohibitive.  This function uses the
    Poincare log map at the origin to project points into tangent space, finds
    approximate neighbors via sklearn BallTree in Euclidean space, then refines
    with exact Poincare distances on a k*oversampling candidate set.

    The log map is a diffeomorphism that preserves local neighborhoods well,
    so with oversampling >= 2-3 the true Poincare top-k is recovered with
    high probability (see ADR-005).

    Args:
        points: Array of shape (N, d), each row a point in B^d.
        k: Number of neighbors. Default: max(10, min(int(sqrt(N)), 50)).
        tau: Bandwidth for Gaussian weights. Default: median kNN distance.
        oversampling: Candidate multiplier for tangent-space search.
        batch_size: Batch size for exact distance refinement.

    Returns:
        Sparse (N, N) symmetric weighted adjacency matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(points)
    if n <= 1:
        return scipy.sparse.csr_matrix((n, n), dtype=np.float64)

    if k is None:
        k = max(10, min(int(math.sqrt(n)), 50))
    k = min(k, n - 1)

    k_search = min(k * oversampling, n - 1)

    # --- Phase 1: log map to tangent space at origin ---
    # log_0(x) = atanh(||x||) / ||x|| * x
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms_clipped = np.clip(norms, 1e-15, 1.0 - 1e-7)
    scale = np.arctanh(norms_clipped) / norms_clipped
    tangent_pts = points * scale
    if verbose:
        print(f"    Tangent-space projection done (N={n}, d={points.shape[1]})")

    # --- Phase 2: Euclidean kNN in tangent space ---
    nn = NearestNeighbors(n_neighbors=k_search + 1, algorithm="ball_tree", metric="euclidean")
    nn.fit(tangent_pts)
    # +1 because query includes self
    euclid_dists, euclid_idx = nn.kneighbors(tangent_pts)
    # Remove self (first column)
    candidate_idx = euclid_idx[:, 1:]  # (N, k_search)
    if verbose:
        print(f"    BallTree kNN done (k_search={k_search})")

    # --- Phase 3: exact Poincare distance on candidates, keep top-k ---
    ball = geoopt.PoincareBall(c=1.0)
    pts_t = torch.as_tensor(points, dtype=torch.float64)

    knn_indices = np.empty((n, k), dtype=np.int64)
    knn_dists = np.empty((n, k), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_size_actual = end - start
        # Gather candidate points for this batch
        batch_cand_idx = candidate_idx[start:end]  # (batch, k_search)
        # Compute exact Poincare distances to candidates
        for i in range(batch_size_actual):
            node_idx = start + i
            cands = batch_cand_idx[i]  # (k_search,)
            d = ball.dist(pts_t[node_idx].unsqueeze(0), pts_t[cands]).detach().numpy()
            # Keep top-k by exact distance
            top_k_local = np.argsort(d)[:k]
            knn_indices[node_idx] = cands[top_k_local]
            knn_dists[node_idx] = d[top_k_local]

    if verbose and n > 5000:
        print(f"    Exact refinement done ({n} nodes × {k_search} candidates → top-{k})")

    # --- Phase 4: symmetrize ---
    rows_arr = np.repeat(np.arange(n, dtype=np.int64), k)
    cols_arr = knn_indices.ravel()

    all_rows = np.concatenate([rows_arr, cols_arr])
    all_cols = np.concatenate([cols_arr, rows_arr])

    sym_mask = scipy.sparse.coo_matrix(
        (np.ones(len(all_rows), dtype=bool), (all_rows, all_cols)),
        shape=(n, n),
    ).tocsr()

    # --- Phase 5: tau + weights ---
    if tau is None:
        tau = float(np.median(knn_dists))
    tau = max(tau, 1e-10)

    sym_coo = sym_mask.tocoo()
    edge_rows = sym_coo.row
    edge_cols = sym_coo.col
    n_edges_total = len(edge_rows)
    if verbose and n > 5000:
        print(f"    Symmetrized: {n_edges_total} edges, computing weights...")

    weights = np.empty(n_edges_total, dtype=np.float64)
    edge_batch = batch_size * k
    for start in range(0, n_edges_total, edge_batch):
        end = min(start + edge_batch, n_edges_total)
        r_batch = edge_rows[start:end]
        c_batch = edge_cols[start:end]
        d = ball.dist(pts_t[r_batch], pts_t[c_batch]).detach().numpy()
        weights[start:end] = np.exp(-d**2 / (2 * tau**2))

    adjacency = scipy.sparse.csr_matrix(
        (weights, (edge_rows, edge_cols)), shape=(n, n)
    )
    return adjacency


def normalized_laplacian(
    adjacency: scipy.sparse.spmatrix,
) -> scipy.sparse.csr_matrix:
    """Compute normalized graph Laplacian L = I - D^{-1/2} W D^{-1/2}.

    Args:
        adjacency: Sparse (N, N) symmetric weighted adjacency matrix.

    Returns:
        Sparse (N, N) normalized Laplacian. Eigenvalues in [0, 2].
    """
    n = adjacency.shape[0]
    if n == 0:
        return scipy.sparse.csr_matrix((0, 0), dtype=np.float64)

    degree = np.array(adjacency.sum(axis=1)).flatten()

    # Guard against isolated nodes (degree=0)
    degree_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    degree_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    d_inv_sqrt = scipy.sparse.diags(degree_inv_sqrt)
    laplacian = scipy.sparse.eye(n, dtype=np.float64) - d_inv_sqrt @ adjacency @ d_inv_sqrt
    return laplacian.tocsr()


def spectral_gap(
    eigenvalues: np.ndarray,
    gap_threshold: float = 2.0,
) -> list[tuple[int, float]]:
    """Find spectral gaps indicating natural scale boundaries.

    A spectral gap at index k means lambda_{k+1}/lambda_k > gap_threshold,
    suggesting a natural boundary at scale sigma* ~ 1/lambda_k (Proposition 3).

    Args:
        eigenvalues: Sorted ascending eigenvalues from eigsh.
        gap_threshold: Minimum ratio for a significant gap (ADR-007, default 2.0).

    Returns:
        List of (k, candidate_sigma) tuples, sorted by gap ratio descending.
        If no gap exceeds threshold, returns the single largest gap as fallback.
    """
    if len(eigenvalues) < 2:
        return []

    # Skip zero/near-zero eigenvalues (graph connectivity eigenvalue)
    eps = 1e-10
    results: list[tuple[int, float]] = []  # (k, candidate_sigma)
    ratios: list[float] = []  # parallel array for sorting
    best_ratio = 0.0
    best_gap: tuple[int, float] | None = None

    for k in range(len(eigenvalues) - 1):
        lam_k = eigenvalues[k]
        lam_k1 = eigenvalues[k + 1]

        if lam_k < eps:
            continue

        ratio = lam_k1 / lam_k
        candidate_sigma = 1.0 / lam_k

        if ratio > best_ratio:
            best_ratio = ratio
            best_gap = (k, candidate_sigma)

        if ratio > gap_threshold:
            results.append((k, candidate_sigma))
            ratios.append(ratio)

    if results:
        # Sort by gap ratio descending (most significant gap first)
        order = np.argsort(ratios)[::-1]
        return [results[i] for i in order]

    # Fallback: return largest gap even if below threshold (ADR-007)
    if best_gap is not None:
        return [best_gap]

    return []
