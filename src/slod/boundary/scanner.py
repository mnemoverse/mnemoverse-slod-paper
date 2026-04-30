"""Algorithm 2: SLoD-BoundaryScan.

Detects natural abstraction boundaries in the scale space:
1. Compute graph Laplacian eigenvalues (spectral prior)
2. Identify candidate scales from spectral gaps
3. Sweep log-spaced sigma grid, compute SLoD at each scale
4. Compute boundary indicators (V, D_w, C_k) between consecutive scales
5. Composite score S = alpha1*V_hat + alpha2*D_hat + alpha3*C_hat
6. Peak picking with robust threshold (median + alpha * MAD)
7. Compute effective dimensionality K*(sigma*) at each boundary

Complexity: O(TNd + NK^2), linear in N for fixed T, K, d.

See ADR-003: composite score weights (equal 1/3).
See ADR-004: peak detection threshold (median + 2*MAD).
See ADR-006: sigma grid (logspace -2 to 2, 100 points).
See ADR-008: effective dimensionality (95% energy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg

from slod.boundary.indicators import (
    neighborhood_churn,
    representation_velocity,
    weight_divergence,
)
from slod.boundary.spectral import spectral_gap
from slod.core.frechet_mean import frechet_mean
from slod.core.heat_kernel import weights_from_eigen


@dataclass
class BoundaryScanResult:
    """Result of Algorithm 2: BoundaryScan.

    Attributes:
        sigma_grid: (T,) sigma values swept.
        scores: (T-1,) composite boundary score S(sigma).
        velocity: (T-1,) representation velocity V(sigma).
        divergence: (T-1,) weight divergence D_w(sigma).
        churn: (T-1,) neighborhood churn C_k(sigma).
        peaks: Indices into ``scores`` where boundaries detected.
        spectral_candidates: Candidate sigmas from spectral gap analysis.
        effective_dims: Mapping peak_idx → K*(sigma*).

    Convention (single source of truth for σ* downstream):
        The indicator arrays have length T-1 because each entry is
        computed between ``sigma_grid[i]`` and ``sigma_grid[i+1]``. Peak
        indices live in the ``scores`` domain (range 0..T-2). Throughout
        the code base — this module's ``_effective_dimensionality``
        call, every ``experiments/ablation_runner*.py`` script, and the
        paper's Table 3–4 rows — a peak index ``i`` maps to
        ``sigma_grid[i]`` (the *start* of the interval). Plot scripts
        must use the same mapping (``sigma_grid[:-1]`` for the x-axis)
        so that marker positions match the tabulated σ*.
    """

    sigma_grid: np.ndarray
    scores: np.ndarray
    velocity: np.ndarray
    divergence: np.ndarray
    churn: np.ndarray
    peaks: list[int] = field(default_factory=list)
    spectral_candidates: list[float] = field(default_factory=list)
    effective_dims: dict[int, int] = field(default_factory=dict)
    slod_points: list[np.ndarray] = field(default_factory=list)


def _zscore_safe(values: np.ndarray) -> np.ndarray:
    """Z-score normalization with zero-std guard (ADR-003)."""
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - mean) / std


def _effective_dimensionality(
    eigenvalues: np.ndarray,
    sigma: float,
    energy_threshold: float = 0.95,
) -> int:
    """Compute K*(sigma) = smallest k capturing energy_threshold of heat energy (ADR-008).

    K*(sigma) = min k such that sum_{i=1}^{k} exp(-sigma*lambda_i) / total >= threshold.
    """
    heat_energies = np.exp(-sigma * eigenvalues)
    total = heat_energies.sum()
    if total <= 0:
        return len(eigenvalues)

    cumsum = np.cumsum(heat_energies) / total
    indices = np.where(cumsum >= energy_threshold)[0]
    if len(indices) == 0:
        return len(eigenvalues)
    return int(indices[0]) + 1


def boundary_scan(
    points: np.ndarray,
    graph_laplacian: scipy.sparse.spmatrix,
    focus_idx: int = 0,
    sigma_grid: np.ndarray | None = None,
    k_eigs: int = 50,
    peak_alpha: float = 2.0,
    peak_distance: int = 5,
    energy_threshold: float = 0.95,
    churn_k: int = 10,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    alpha_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> BoundaryScanResult:
    """Algorithm 2: BoundaryScan — detect natural scale boundaries.

    Args:
        points: Array of shape (N, d), each row a point in B^d.
        graph_laplacian: Sparse (N, N) graph Laplacian.
        focus_idx: Index of focus node for SLoD computation.
        sigma_grid: Array of sigma values to sweep. Default: logspace(-2, 2, 100) (ADR-006).
        k_eigs: Number of eigenvalues for spectral approximation.
        peak_alpha: Threshold multiplier for peak detection: median + alpha*MAD (ADR-004).
        peak_distance: Minimum distance between peaks in grid points (ADR-004).
        energy_threshold: Energy fraction for K*(sigma) computation (ADR-008).
        churn_k: Number of neighbors for neighborhood churn C_k.
        eigenvalues: Pre-computed eigenvalues (k,). If provided with eigenvectors,
            skips eigendecomposition (useful when calling multiple times with same Laplacian).
        eigenvectors: Pre-computed eigenvectors (N, k).
        alpha_weights: Weights for composite score: (V_weight, D_w_weight, C_k_weight).
            Default (1/3, 1/3, 1/3) = equal weights (ADR-003). Must sum to ~1.0.

    Returns:
        BoundaryScanResult with all indicators, composite scores, detected peaks,
        and SLoD points at each sigma.
    """
    # Validate alpha_weights (ADR-003)
    if len(alpha_weights) != 3:
        raise ValueError(f"alpha_weights must have 3 elements, got {len(alpha_weights)}")
    weight_sum = sum(alpha_weights)
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"alpha_weights must sum to ~1.0, got {weight_sum:.4f}")

    n = graph_laplacian.shape[0]

    if sigma_grid is None:
        sigma_grid = np.logspace(-2, 2, 100)

    t = len(sigma_grid)

    # Step 1: Eigendecomposition (use pre-computed if available)
    if eigenvalues is not None and eigenvectors is not None:
        eigenvalues = np.maximum(eigenvalues, 0.0)
    else:
        k = min(k_eigs, n - 1)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            graph_laplacian.astype(np.float64), k=k, which="SA"
        )
        eigenvalues = np.maximum(eigenvalues, 0.0)

    # Step 2: Spectral gap candidates
    spectral_candidates_list = spectral_gap(eigenvalues)
    spectral_candidates = [sigma for _k, sigma in spectral_candidates_list]

    # Step 3: Sweep sigma grid — compute weights and SLoD at each scale
    all_weights = []
    all_slod_points = []

    for sigma in sigma_grid:
        w = weights_from_eigen(eigenvalues, eigenvectors, sigma, focus_idx)
        all_weights.append(w)
        phi = frechet_mean(points, w)
        all_slod_points.append(phi)

    # Step 4: Compute indicators between consecutive scales
    velocity_arr = np.zeros(t - 1)
    divergence_arr = np.zeros(t - 1)
    churn_arr = np.zeros(t - 1)

    for i in range(t - 1):
        delta_sigma = sigma_grid[i + 1] - sigma_grid[i]
        velocity_arr[i] = representation_velocity(
            all_slod_points[i], all_slod_points[i + 1], delta_sigma
        )
        divergence_arr[i] = weight_divergence(all_weights[i], all_weights[i + 1])
        churn_arr[i] = neighborhood_churn(
            all_slod_points[i], all_slod_points[i + 1], points, k=churn_k
        )

    # Step 5: Z-score normalization + composite score (ADR-003)
    v_hat = _zscore_safe(velocity_arr)
    d_hat = _zscore_safe(divergence_arr)
    c_hat = _zscore_safe(churn_arr)
    scores = alpha_weights[0] * v_hat + alpha_weights[1] * d_hat + alpha_weights[2] * c_hat

    # Step 6: Peak detection (ADR-004)
    peaks_list: list[int] = []
    if len(scores) > 0:
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        # When MAD ≈ 0 (all scores similar), threshold ≈ median,
        # so any score above median becomes a peak. This is intentional.
        mad = max(mad, 1e-10)
        threshold = median + peak_alpha * mad

        peak_indices, _ = scipy.signal.find_peaks(
            scores, height=threshold, distance=peak_distance
        )
        peaks_list = peak_indices.tolist()

        # ADR-004 fallback: if no peaks above threshold, return highest local max
        if len(peaks_list) == 0:
            all_peaks, _ = scipy.signal.find_peaks(scores, distance=peak_distance)
            if len(all_peaks) > 0:
                best = all_peaks[np.argmax(scores[all_peaks])]
                peaks_list = [int(best)]

    # Step 7: Effective dimensionality at each peak (ADR-008)
    effective_dims = {}
    for peak_idx in peaks_list:
        sigma_star = sigma_grid[peak_idx]
        effective_dims[peak_idx] = _effective_dimensionality(
            eigenvalues, sigma_star, energy_threshold
        )

    return BoundaryScanResult(
        sigma_grid=sigma_grid,
        scores=scores,
        velocity=velocity_arr,
        divergence=divergence_arr,
        churn=churn_arr,
        peaks=peaks_list,
        spectral_candidates=spectral_candidates,
        effective_dims=effective_dims,
        slod_points=all_slod_points,
    )
