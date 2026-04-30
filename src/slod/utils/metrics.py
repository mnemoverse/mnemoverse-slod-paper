"""Evaluation metrics for SLoD experiments.

- Adjusted Rand Index (ARI): partition quality vs planted structure
- Variation of Information (VI): true metric on partitions
- Kendall tau: rank correlation (boundary order vs ancestor depth)
- Hit@L: fraction of boundaries within L hops of true depth boundary
- Mean Reciprocal Rank (MRR): ancestor retrieval across scales
"""

from __future__ import annotations

import numpy as np
import scipy.stats
import sklearn.metrics
import sklearn.metrics.cluster


def adjusted_rand_index(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    """Adjusted Rand Index between two clusterings.

    ARI = 1.0 for perfect agreement, 0.0 for random, negative for worse-than-random.

    Args:
        labels_true: Ground truth labels, shape (N,).
        labels_pred: Predicted labels, shape (N,).

    Returns:
        ARI score in [-1, 1].
    """
    return float(sklearn.metrics.adjusted_rand_score(labels_true, labels_pred))


def variation_of_information(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> float:
    """Variation of Information: VI(X, Y) = H(X|Y) + H(Y|X).

    True metric on partitions. VI = 0 iff partitions are identical.
    Uses natural logarithm (nats). For empty inputs, returns 0.0.

    Args:
        labels_true: Ground truth labels, shape (N,).
        labels_pred: Predicted labels, shape (N,).

    Returns:
        Non-negative VI score in nats. 0.0 for identical partitions.
    """
    n = len(labels_true)
    if n == 0:
        return 0.0

    contingency = sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred)
    contingency_norm = contingency / n  # Normalize to joint probability

    # Marginals
    p_true = contingency_norm.sum(axis=1)
    p_pred = contingency_norm.sum(axis=0)

    def _entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_true = _entropy(p_true)
    h_pred = _entropy(p_pred)
    h_joint = _entropy(contingency_norm.flatten())

    # VI = H(X|Y) + H(Y|X) = 2*H(X,Y) - H(X) - H(Y)
    return max(0.0, 2 * h_joint - h_true - h_pred)


def kendall_tau(
    boundary_sigmas: np.ndarray,
    boundary_heights: np.ndarray,
) -> float:
    """Kendall rank correlation between detected boundary scales and hierarchy heights.

    Exp 2 metric: correlate σ* (detected boundary scales) with height
    (max_depth − depth). Expect positive τ: larger σ = coarser = shallower = higher.

    Args:
        boundary_sigmas: Detected boundary scales, shape (m,).
        boundary_heights: Corresponding hierarchy heights, shape (m,).

    Returns:
        Kendall τ correlation coefficient in [-1, 1].
        Returns 0.0 if fewer than 2 boundaries.
    """
    if len(boundary_sigmas) < 2 or len(boundary_heights) < 2:
        return 0.0
    tau, _pvalue = scipy.stats.kendalltau(boundary_sigmas, boundary_heights)
    return float(tau) if np.isfinite(tau) else 0.0


def hit_at_l(
    detected_depths: np.ndarray,
    true_depths: np.ndarray,
    hops: int = 1,
) -> float:
    """Fraction of detected depths within L hops of a true depth boundary.

    For each detected depth, check if any true depth is within ``hops`` hops.

    Args:
        detected_depths: Detected boundary depths, shape (m,).
        true_depths: True hierarchy depth boundaries, shape (k,).
        hops: Hop tolerance (default 1).

    Returns:
        Fraction in [0, 1]. Returns 0.0 if no detected depths.
    """
    if len(detected_depths) == 0:
        return 0.0
    if len(true_depths) == 0:
        return 0.0
    detected = np.asarray(detected_depths, dtype=np.float64)
    true = np.asarray(true_depths, dtype=np.float64)
    # For each detected, minimum distance to any true boundary
    min_dists = np.abs(detected[:, None] - true[None, :]).min(axis=1)
    return float(np.mean(min_dists <= hops))


def mean_reciprocal_rank(
    true_ancestors: list[list[int]],
    pred_ancestors: list[list[int]],
) -> float:
    """Mean Reciprocal Rank for ancestor retrieval across scales.

    For each query, the predicted ancestors are ranked. MRR measures how early
    the first true ancestor appears in the predicted ranking.

    Args:
        true_ancestors: List of true ancestor sets, one per query.
            Each element is a list of true ancestor indices.
        pred_ancestors: List of predicted ancestor rankings, one per query.
            Each element is a ranked list of predicted ancestor indices.

    Returns:
        MRR score in (0, 1]. Returns 0.0 if no queries or no true ancestors found.
    """
    if not true_ancestors or not pred_ancestors:
        return 0.0
    rr_sum = 0.0
    n_evaluated = 0
    n = min(len(true_ancestors), len(pred_ancestors))
    for i in range(n):
        true_set = set(true_ancestors[i])
        if not true_set:
            continue
        n_evaluated += 1
        for rank, pred in enumerate(pred_ancestors[i], start=1):
            if pred in true_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / n_evaluated if n_evaluated > 0 else 0.0
