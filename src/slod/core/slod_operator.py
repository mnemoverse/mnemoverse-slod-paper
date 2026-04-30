"""The SLoD operator: Phi_sigma(V, x0).

Semantic Level of Detail at scale sigma with focus x0:
1. Compute heat kernel weights: w_i(sigma, x0) = K_sigma(x0, v_i) / sum_j K_sigma(x0, v_j)
2. Compute weighted Frechet mean: Phi_sigma = argmin sum_i w_i * d_H^2(y, v_i)

Properties (proven in paper):
- Hierarchical coherence: d_H(Phi_s1, Phi_s2) <= C * |s2 - s1| * log(n)
- Approximation error: O(sigma)
- sigma -> 0: preserves local detail
- sigma -> inf: global semantic summary
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from slod.core.frechet_mean import frechet_mean
from slod.core.heat_kernel import heat_kernel_weights


def slod(
    points: np.ndarray,
    graph_laplacian: scipy.sparse.spmatrix,
    sigma: float,
    focus_idx: int,
    k_eigs: int = 50,
) -> np.ndarray:
    """Compute SLoD operator Phi_sigma at a focus node.

    Args:
        points: Array of shape (N, d), each row a point in B^d.
        graph_laplacian: Sparse (N, N) graph Laplacian.
        sigma: Diffusion scale (small = local, large = global).
        focus_idx: Index of the focus node.
        k_eigs: Number of eigenpairs for heat kernel approximation.

    Returns:
        Weighted Frechet mean in B^d, shape (d,).
    """
    weights = heat_kernel_weights(graph_laplacian, sigma, focus_idx, k_eigs=k_eigs)
    return frechet_mean(points, weights)
