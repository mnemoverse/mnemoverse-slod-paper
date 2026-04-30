"""D_w(σ) overlay with 1/λ_k vertical lines — Proposition 1(i) empirical anchor.

Generates Figure (panel for ablation_profiles or new figure) showing the
weight-divergence indicator D_w(σ) = JSD(w(σ) ∥ w(σ+Δσ)) plotted on a log-σ
axis with vertical dashed lines at 1/λ_k for the top spectral gaps. The
peaks of D_w(σ) should align with the 1/λ_k positions, providing the
empirical anchor for Proposition 1(i) that Figure 1's right panel does not
(Figure 1 right panel shows |dK*/d log σ|, which probes Prop 1(ii) instead).

Uses cached HSBM r=200, seed=42 ablation data — same Laplacian as Tables 3-4.

Output: paper/figures/dw_vs_inv_lambda.pdf (and .png)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # avoid Type 3 fonts
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon


REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "results" / "exp1" / "ablation_cache_r200_s42.pkl"
OUT_PDF = REPO / "paper" / "figures" / "dw_vs_inv_lambda.pdf"
OUT_PNG = REPO / "paper" / "figures" / "dw_vs_inv_lambda.png"


def heat_weights(eigenvalues: np.ndarray, eigenvectors: np.ndarray, sigma: float, focus: int) -> np.ndarray:
    """K_sigma(focus, j) = sum_k exp(-sigma lam_k) phi_k(focus) phi_k(j); normalized to simplex."""
    coeffs = np.exp(-sigma * eigenvalues) * eigenvectors[focus, :]
    raw = eigenvectors @ coeffs
    raw = np.maximum(raw, 0.0)
    s = raw.sum()
    if s <= 0:
        # focus at scale where all mass leaks out (numerical underflow); fall back to uniform
        return np.full(eigenvectors.shape[0], 1.0 / eigenvectors.shape[0])
    return raw / s


def main() -> int:
    if not CACHE.exists():
        print(f"ERROR: cache not found at {CACHE}", file=sys.stderr)
        return 1

    with open(CACHE, "rb") as f:
        data = pickle.load(f)

    eigvals = data["eigenvalues_knn"]
    eigvecs = data["eigenvectors_knn"]
    n = data["n_nodes"]
    print(f"Loaded r=200 seed=42 cache: N={n}, K_eigs={len(eigvals)}")
    print(f"  λ_1..λ_5 = {eigvals[:5]}")
    print(f"  λ_2/λ_1 = {eigvals[1]/max(eigvals[0],1e-12):.3g}")
    print(f"  λ_3/λ_2 = {eigvals[2]/max(eigvals[1],1e-12):.3g}")
    print(f"  λ_9/λ_8 = {eigvals[8]/max(eigvals[7],1e-12):.3g}")

    # Focus = node 0 (matches paper's BoundaryScan setup, §6.3).
    focus = 0

    # Sigma grid: extended to logspace(-2, 3, 150) to cover 1/λ_2 ≈ 128 (macro
    # candidate scale at r=200). The default ADR-006 grid stops at σ=100, which
    # truncates the macro Fiedler mode's expected D_w peak.
    sigma_grid = np.logspace(-2, 3, 150)

    # Compute heat-kernel weights on grid.
    weights = np.zeros((len(sigma_grid), n), dtype=np.float64)
    for i, sigma in enumerate(sigma_grid):
        weights[i] = heat_weights(eigvals, eigvecs, float(sigma), focus)

    # D_w(σ) = JSD(w(σ_t), w(σ_{t+1})). scipy.spatial.distance.jensenshannon returns
    # sqrt(JSD); square it to get JSD itself (matches indicators.py convention).
    d_w = np.zeros(len(sigma_grid) - 1, dtype=np.float64)
    for t in range(len(sigma_grid) - 1):
        js = jensenshannon(weights[t], weights[t + 1], base=np.e)
        d_w[t] = 0.0 if not np.isfinite(js) else js * js

    # Plot.
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    sigma_plot = sigma_grid[:-1]
    ax.plot(sigma_plot, d_w, color="C0", lw=1.8, label=r"$D_w(\sigma)$ (weight-divergence indicator)")

    # Vertical lines at 1/λ_k for the macro and meso candidate scales — these are
    # the ones predicted by Proposition 1(i) given large gap above. Skip k=1
    # because λ_1 = 0 (trivial constant mode). Show 1/λ_2 (macro) and 1/λ_8 (meso)
    # plus a few mid-spectrum scales where the dominant D_w peak lies.
    inv_lambdas = []
    for k in range(1, len(eigvals)):
        if eigvals[k] > 1e-10:
            inv_lambdas.append((k, 1.0 / eigvals[k]))

    # Collect specific scales to mark: macro (k=1 → 1/λ_2), meso (k=7 → 1/λ_8),
    # and the mid-spectrum k that hosts the dominant empirical peak.
    marked = {}
    for k, inv_lam in inv_lambdas:
        if k in (1, 7, 19):
            marked[k] = inv_lam

    legend_added = False
    for k in (1, 7, 19):
        if k not in marked:
            continue
        inv_lam = marked[k]
        if k == 1:
            label_str = r"$1/\lambda_2$ (macro, Prop 1(i) anchor)"
            color = "C3"
        elif k == 7:
            label_str = r"$1/\lambda_8$ (meso; small gap $\lambda_9/\lambda_8{\approx}1.5$)"
            color = "C2"
        elif k == 19:
            label_str = r"$1/\lambda_{20}$ (mid-spectrum, dominant empirical peak)"
            color = "C1"
        ax.axvline(inv_lam, color=color, lw=1.2, ls="--", alpha=0.8,
                   label=label_str if not legend_added or k != 1 else label_str)
        legend_added = True

    ax.set_xscale("log")
    ax.set_xlabel(r"diffusion scale $\sigma$ (log)")
    ax.set_ylabel(r"$D_w(\sigma) = \mathrm{JSD}(w(\sigma)\,\|\,w(\sigma{+}\Delta\sigma))$")
    ax.set_title(
        "Empirical anchor for Proposition 1(i): "
        "HSBM $r{=}200$ (seed 42), kNN-of-Poincaré-embedding Laplacian, focus = node 0"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, bbox_inches="tight", dpi=180)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")

    # Also print numerical alignment summary.
    peaks = []
    for t in range(1, len(d_w) - 1):
        if d_w[t] > d_w[t - 1] and d_w[t] > d_w[t + 1] and d_w[t] > d_w.mean():
            peaks.append((sigma_plot[t], d_w[t]))
    print("\nLocal D_w peaks (above-mean, σ_peak):")
    for sigma_pk, val in peaks[:8]:
        # Match to nearest 1/λ_k.
        diffs = [(abs(np.log(sigma_pk) - np.log(inv_lam)), k, inv_lam) for k, inv_lam in inv_lambdas if inv_lam > 1e-6]
        diffs.sort()
        d_log, k_match, inv_lam_match = diffs[0]
        print(f"  σ_peak = {sigma_pk:.3g}, D_w = {val:.4g} → nearest 1/λ_{k_match+1} = {inv_lam_match:.3g} (log-distance {d_log:.2f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
