"""Boundary indicators for scale transition detection.

Three complementary signals (Definitions 4-6 in paper):
- V(sigma): representation velocity = d_H(Phi_{sigma+delta}, Phi_sigma) / delta
- D_w(sigma): weight divergence = JSD(w(sigma) || w(sigma + delta))
- C_k(sigma): neighborhood churn = 1 - |NN_k(sigma) & NN_k(sigma+delta)| / |union|

Jensen-Shannon divergence is ideal: bounded [0, ln 2], symmetric, sqrt is a metric.
"""

from __future__ import annotations

import geoopt
import numpy as np
import scipy.spatial.distance
import torch

from slod.core.poincare import poincare_distance

_ball = geoopt.PoincareBall(c=1.0)


def representation_velocity(
    phi_prev: np.ndarray,
    phi_curr: np.ndarray,
    delta_sigma: float,
) -> float:
    """Representation velocity V(sigma) = d_H(Phi_curr, Phi_prev) / delta_sigma.

    Measures how fast the SLoD representation moves in hyperbolic space
    as sigma changes. High velocity → rapid change → potential boundary.

    Paper: Definition 4.

    Args:
        phi_prev: SLoD point at sigma - delta, shape (d,).
        phi_curr: SLoD point at sigma, shape (d,).
        delta_sigma: Step size in sigma space.

    Returns:
        Non-negative scalar velocity.
    """
    if delta_sigma <= 0:
        raise ValueError(f"`delta_sigma` must be positive; got {delta_sigma}")
    return poincare_distance(phi_prev, phi_curr) / delta_sigma


def weight_divergence(
    w_prev: np.ndarray,
    w_curr: np.ndarray,
) -> float:
    """Weight divergence D_w(sigma) = JSD(w_prev || w_curr).

    Jensen-Shannon divergence between consecutive weight distributions.
    Bounded [0, ln(2)]. Large JSD → weight structure changes significantly.

    Paper: Definition 5.

    Note: scipy.spatial.distance.jensenshannon returns sqrt(JSD).
    We square it to get JSD proper.

    Args:
        w_prev: Weight distribution at sigma - delta, shape (N,).
        w_curr: Weight distribution at sigma, shape (N,).

    Returns:
        JSD value in [0, ln(2)].
    """
    # Ensure proper probability distributions
    w_prev = np.asarray(w_prev, dtype=np.float64)
    w_curr = np.asarray(w_curr, dtype=np.float64)

    # Guard against zero distributions.
    # If either distribution is all-zero, the heat kernel failed to produce
    # meaningful weights (degenerate case). Return 0.0 ("no information")
    # rather than ln(2) ("maximum divergence") — safer for composite scoring.
    if w_prev.sum() == 0 or w_curr.sum() == 0:
        return 0.0

    # scipy returns sqrt(JSD), square to get JSD
    sqrt_jsd = scipy.spatial.distance.jensenshannon(w_prev, w_curr)
    if np.isnan(sqrt_jsd):
        return 0.0
    return float(sqrt_jsd**2)


def neighborhood_churn(
    phi_prev: np.ndarray,
    phi_curr: np.ndarray,
    all_points: np.ndarray,
    k: int = 10,
) -> float:
    """Neighborhood churn C_k(sigma) = Jaccard distance of kNN index sets.

    C_k = 1 - |NN_k(phi_prev) ∩ NN_k(phi_curr)| / |NN_k(phi_prev) ∪ NN_k(phi_curr)|

    Measures how much the local neighborhood changes between scales.
    Range [0, 1]. High churn → neighborhood restructuring → potential boundary.

    Paper: Definition 6.

    Args:
        phi_prev: SLoD point at sigma - delta, shape (d,).
        phi_curr: SLoD point at sigma, shape (d,).
        all_points: All embedding points, shape (N, d).
        k: Number of nearest neighbors.

    Returns:
        Jaccard distance in [0, 1].
    """
    n = len(all_points)
    k = min(k, n)

    if k == 0:
        return 0.0

    # Vectorized Poincare distances via geoopt
    pts_t = torch.as_tensor(all_points, dtype=torch.float64)
    prev_t = torch.as_tensor(phi_prev, dtype=torch.float64).unsqueeze(0)
    curr_t = torch.as_tensor(phi_curr, dtype=torch.float64).unsqueeze(0)
    dists_prev = _ball.dist(prev_t, pts_t).numpy()
    dists_curr = _ball.dist(curr_t, pts_t).numpy()

    nn_prev = set(np.argsort(dists_prev)[:k])
    nn_curr = set(np.argsort(dists_curr)[:k])

    intersection = len(nn_prev & nn_curr)
    union = len(nn_prev | nn_curr)

    if union == 0:
        return 0.0

    return 1.0 - intersection / union
