"""Embedding quality tests for Poincare embeddings.

Quality gates that must pass before running BoundaryScan on real-world data.
Based on slod_notes.md point 1 (Eduard, 2026-02-26).

Three tests:
1. Hierarchy preservation: d(parent, child) < d(parent, random) for >90% of pairs
2. Depth correlation: Pearson(||embed||, depth) > 0.6
3. Sibling proximity: d(sib1, sib2) < d(sib1, random) for >85% of pairs
"""

from __future__ import annotations

import numpy as np
import scipy.stats

from slod.core.poincare import poincare_distance


def hierarchy_preservation_test(
    points: np.ndarray,
    parent_child_pairs: list[tuple[int, int]],
    n: int = 1000,
    seed: int = 42,
) -> float:
    """Test that parent-child distances are shorter than parent-random distances.

    For each (parent, child) pair, sample a random node and check:
        d_H(parent, child) < d_H(parent, random_node)

    Args:
        points: Embedding array, shape (N, d).
        parent_child_pairs: List of (parent_idx, child_idx) tuples.
        n: Number of pairs to test (samples from parent_child_pairs).
        seed: Random seed.

    Returns:
        Fraction of pairs where hierarchy is preserved, in [0, 1].
        Quality gate: >0.90.
    """
    rng = np.random.RandomState(seed)
    n_nodes = points.shape[0]
    n_pairs = len(parent_child_pairs)

    if n_pairs == 0 or n_nodes < 3:
        return 0.0

    # Sample n pairs (with replacement if n > n_pairs)
    indices = rng.choice(n_pairs, size=min(n, n_pairs), replace=False)

    preserved = 0
    for idx in indices:
        parent_idx, child_idx = parent_child_pairs[idx]
        random_idx = rng.randint(0, n_nodes)
        # Avoid self-comparison
        while random_idx == parent_idx or random_idx == child_idx:
            random_idx = rng.randint(0, n_nodes)

        d_pc = poincare_distance(points[parent_idx], points[child_idx])
        d_pr = poincare_distance(points[parent_idx], points[random_idx])

        if d_pc < d_pr:
            preserved += 1

    return preserved / len(indices)


def depth_correlation_test(
    points: np.ndarray,
    depths: np.ndarray,
) -> float:
    """Test correlation between embedding norm and hierarchy depth.

    In hyperbolic space, deeper nodes should be farther from the origin
    (higher ||x||). Pearson correlation measures this relationship.

    Args:
        points: Embedding array, shape (N, d).
        depths: Depth array, shape (N,).

    Returns:
        Pearson correlation coefficient.
        Quality gate: >0.60.
    """
    norms = np.linalg.norm(points, axis=1)
    if len(norms) < 3 or np.std(norms) < 1e-10 or np.std(depths) < 1e-10:
        return 0.0
    r, _pvalue = scipy.stats.pearsonr(norms, depths)
    return float(r) if np.isfinite(r) else 0.0


def sibling_proximity_test(
    points: np.ndarray,
    sibling_groups: list[list[int]],
    n: int = 500,
    seed: int = 42,
) -> float:
    """Test that siblings are closer to each other than to random nodes.

    For each sibling pair, check: d_H(sib1, sib2) < d_H(sib1, random).

    Args:
        points: Embedding array, shape (N, d).
        sibling_groups: List of groups, each a list of sibling node indices.
            Siblings share the same parent in the hierarchy.
        n: Number of pairs to test.
        seed: Random seed.

    Returns:
        Fraction of pairs where siblings are closer, in [0, 1].
        Quality gate: >0.85.
    """
    rng = np.random.RandomState(seed)
    n_nodes = points.shape[0]

    # Collect all sibling pairs
    all_pairs = []
    for group in sibling_groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                all_pairs.append((group[i], group[j]))

    if not all_pairs or n_nodes < 3:
        return 0.0

    # Sample n pairs
    indices = rng.choice(len(all_pairs), size=min(n, len(all_pairs)), replace=False)

    preserved = 0
    for idx in indices:
        sib1, sib2 = all_pairs[idx]
        random_idx = rng.randint(0, n_nodes)
        while random_idx == sib1 or random_idx == sib2:
            random_idx = rng.randint(0, n_nodes)

        d_siblings = poincare_distance(points[sib1], points[sib2])
        d_random = poincare_distance(points[sib1], points[random_idx])

        if d_siblings < d_random:
            preserved += 1

    return preserved / len(indices)
