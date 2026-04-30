"""Heat kernel weights via graph Laplacian eigendecomposition.

Computes localized heat kernel weights for a focus node:
    w_i(sigma) = K_sigma(focus, i) / sum_j K_sigma(focus, j)

where K_sigma = exp(-sigma * L) is the heat kernel matrix and L is
the graph Laplacian.

Spectral decomposition: K_sigma(i, j) = sum_k exp(-sigma * lam_k) * phi_k(i) * phi_k(j)

Uses scipy.sparse.linalg.eigsh for partial eigendecomposition (ADR-001).
Eigenvalues can be returned for reuse by BoundaryScan (spectral gap detection).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def weights_from_eigen(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sigma: float,
    focus_idx: int,
) -> np.ndarray:
    """Compute normalized heat kernel weights from pre-computed eigendecomposition.

    K_sigma(focus, j) = sum_k exp(-sigma * lam_k) * phi_k(focus) * phi_k(j)

    Shared by heat_kernel_weights() and boundary scanner (which caches eigen).

    Args:
        eigenvalues: (k,) smallest eigenvalues of graph Laplacian.
        eigenvectors: (N, k) corresponding eigenvectors.
        sigma: Diffusion scale parameter.
        focus_idx: Index of the focus node.

    Returns:
        Weights array of shape (N,), non-negative, sums to 1.0.
    """
    phi_focus = eigenvectors[focus_idx, :]
    heat_coeffs = np.exp(-sigma * eigenvalues)
    weights = (eigenvectors * heat_coeffs[np.newaxis, :] * phi_focus[np.newaxis, :]).sum(axis=1)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        n = eigenvectors.shape[0]
        weights = np.ones(n) / n
    return weights


def heat_kernel_weights(
    graph_laplacian: scipy.sparse.spmatrix,
    sigma: float,
    focus_idx: int,
    k_eigs: int = 50,
    return_eigen: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute normalized heat kernel weights centered at focus_idx.

    Args:
        graph_laplacian: Sparse (N, N) symmetric graph Laplacian.
        sigma: Diffusion scale parameter (small = local, large = global).
        focus_idx: Index of the focus node.
        k_eigs: Number of smallest eigenvalues to compute (default 50).
        return_eigen: If True, return (weights, eigenvalues, eigenvectors).

    Returns:
        Weights array of shape (N,), non-negative, sums to 1.0.
        If return_eigen=True: tuple of (weights, eigenvalues, eigenvectors).
    """
    n = graph_laplacian.shape[0]

    # Edge case: single node
    if n <= 1:
        w = np.ones(max(n, 1))
        if return_eigen:
            return w, np.zeros(0), np.zeros((max(n, 1), 0))
        return w

    # Validate focus_idx
    if not (0 <= focus_idx < n):
        raise ValueError(f"`focus_idx` must be in [0, {n}); got {focus_idx}")

    # sigma=0: exact delta at focus (no diffusion)
    if sigma == 0:
        w = np.zeros(n)
        w[focus_idx] = 1.0
        if return_eigen:
            k = min(k_eigs, n - 1)
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                graph_laplacian.astype(np.float64), k=k, which="SA"
            )
            eigenvalues = np.maximum(eigenvalues, 0.0)
            return w, eigenvalues, eigenvectors
        return w

    k = min(k_eigs, n - 1)

    # Partial eigendecomposition: smallest algebraic eigenvalues of L
    # "SA" is more robust than "SM" for PSD matrices (avoids shift-invert issues)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        graph_laplacian.astype(np.float64), k=k, which="SA"
    )

    # Clamp tiny negative eigenvalues (numerical noise) to zero
    eigenvalues = np.maximum(eigenvalues, 0.0)

    weights = weights_from_eigen(eigenvalues, eigenvectors, sigma, focus_idx)

    if return_eigen:
        return weights, eigenvalues, eigenvectors
    return weights
