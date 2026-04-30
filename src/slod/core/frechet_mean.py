"""Algorithm 1: Frechet mean via tangent-space aggregation.

Computes the weighted Frechet mean on the Poincare ball B^d:
    mu = argmin_{y in B^d} sum_i w_i * d_H(y, v_i)^2

Iterative algorithm:
1. Map points to tangent space at current estimate: u_i = Log_mu(v_i)
2. Weighted average in tangent space: u_bar = sum_i w_i * u_i
3. Map back to manifold: mu_new = Exp_mu(eta * u_bar)
4. Project to ball: ||mu_new|| <= 1 - epsilon

Convergence guaranteed on Hadamard manifolds (Afsari et al. 2013).
Uniqueness guaranteed by strict convexity of Frechet functional (Sturm 2003).

See ADR-002: Manual Algorithm 1 over geoopt built-in.
"""

from __future__ import annotations

import geoopt
import numpy as np
import torch

from slod.core.poincare import project_to_ball

_ball = geoopt.PoincareBall(c=1.0)


def frechet_mean(
    points: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 15,
    lr: float = 1.0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Weighted Frechet mean on the Poincare ball (Algorithm 1).

    Args:
        points: Array of shape (N, d), each row a point in B^d.
        weights: Array of shape (N,), non-negative, sums to 1.
        n_iter: Maximum number of iterations (default 15, ADR-002).
        lr: Learning rate / step size (default 1.0).
        tol: Convergence tolerance on d_H(mu_new, mu_old) (default 1e-6).

    Returns:
        Frechet mean point in B^d, shape (d,).
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if points.ndim != 2:
        raise ValueError(f"`points` must have shape (N, d); got {points.shape!r}")
    if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
        raise ValueError(
            f"`weights` must be 1D with length N={points.shape[0]}; got {weights.shape!r}"
        )
    if np.any(weights < 0):
        raise ValueError("`weights` must be non-negative.")

    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("Sum of `weights` must be positive.")
    weights = weights / weight_sum

    if len(points) == 1:
        return project_to_ball(points[0].copy())

    # Vectorized via geoopt: batch logmap + expmap on tensors
    pts_t = torch.as_tensor(points, dtype=torch.float64)
    w_t = torch.as_tensor(weights, dtype=torch.float64)

    # Initialize at the point with highest weight
    mu_np = project_to_ball(points[np.argmax(weights)].copy())
    mu_t = torch.as_tensor(mu_np, dtype=torch.float64)

    for _ in range(n_iter):
        # Step 1: Batch logmap — all points to tangent space at mu
        # logmap(mu, pts) with mu broadcast: (1, d) vs (N, d) → (N, d)
        tangent_vecs = _ball.logmap(mu_t.unsqueeze(0), pts_t)  # (N, d)

        # Step 2: Weighted average in tangent space
        tangent_avg = (w_t.unsqueeze(1) * tangent_vecs).sum(dim=0)  # (d,)

        # Step 3: Move along weighted tangent direction
        mu_new_t = _ball.expmap(mu_t, lr * tangent_avg)
        mu_new_t = _ball.projx(mu_new_t)

        # Step 4: Check convergence
        dist = float(_ball.dist(mu_t, mu_new_t))
        if dist < tol:
            result = mu_new_t.numpy()
            return project_to_ball(result)

        mu_t = mu_new_t

    result = mu_t.numpy()
    return project_to_ball(result)
