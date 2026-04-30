"""Poincare ball B^d operations.

Implements:
- Geodesic distance d_H(x, y)
- Exponential map Exp_x(v)
- Logarithmic map Log_x(y)
- Mobius addition x (+) y
- Conformal factor lambda_x
- Projection to ball interior

All operations use the Poincare ball model with curvature kappa = -1.
Wraps geoopt.PoincareBall for correctness; accepts/returns numpy arrays.
"""

from __future__ import annotations

import geoopt
import numpy as np
import torch

_ball = geoopt.PoincareBall(c=1.0)


def _to_torch(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor (float64 for numerical precision)."""
    return torch.as_tensor(x, dtype=torch.float64)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return t.detach().cpu().numpy()


def poincare_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Geodesic distance d_H(x, y) on the Poincare ball.

    Args:
        x: Point in B^d, shape (d,).
        y: Point in B^d, shape (d,).

    Returns:
        Non-negative scalar distance.
    """
    return float(_ball.dist(_to_torch(x), _to_torch(y)))


def exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exponential map Exp_x(v): tangent vector v at x -> point on ball.

    Args:
        x: Base point in B^d, shape (d,).
        v: Tangent vector at x, shape (d,).

    Returns:
        Point in B^d, shape (d,).
    """
    result = _ball.expmap(_to_torch(x), _to_torch(v))
    return _to_numpy(_ball.projx(result))


def log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logarithmic map Log_x(y): point y -> tangent vector at x.

    Args:
        x: Base point in B^d, shape (d,).
        y: Target point in B^d, shape (d,).

    Returns:
        Tangent vector at x, shape (d,).
    """
    return _to_numpy(_ball.logmap(_to_torch(x), _to_torch(y)))


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mobius addition x (+) y on the Poincare ball.

    Args:
        x: Point in B^d, shape (d,).
        y: Point in B^d, shape (d,).

    Returns:
        Point in B^d, shape (d,).
    """
    result = _ball.mobius_add(_to_torch(x), _to_torch(y))
    return _to_numpy(_ball.projx(result))


def project_to_ball(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Project point to Poincare ball interior: ||x|| < 1 - eps.

    Args:
        x: Point, shape (d,).
        eps: Safety margin from boundary.

    Returns:
        Point in B^d with ||x|| < 1 - eps, shape (d,).
    """
    norm = np.linalg.norm(x)
    max_norm = 1.0 - eps
    if norm >= max_norm:
        return x * (max_norm / norm)
    return x.copy()


def conformal_factor(x: np.ndarray) -> float:
    """Conformal factor lambda_x = 2 / (1 - ||x||^2).

    Args:
        x: Point in B^d, shape (d,).

    Returns:
        Positive scalar. Explodes as ||x|| -> 1.
    """
    norm_sq = float(np.dot(x, x))
    return 2.0 / max(1.0 - norm_sq, 1e-10)
