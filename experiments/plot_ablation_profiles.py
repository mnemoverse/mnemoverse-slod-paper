#!/usr/bin/env python
"""ARI vs σ (per-config composite score profile) figure for §6.3.

Shows why different indicator mixes peak at different σ: plots the full
composite score S(σ) for {full, V-only, D-only, C-only} at a selected r,
with the picked top-peak marked. Optionally overlays the Euclidean-full
profile for a geometry comparison.

Inputs: results/exp1/ablation_cache_r{r}_s{seed}.pkl
        results/exp1/ablation_cache_euclidean_r{r}_s{seed}.pkl (optional)

Output: paper/figures/ablation_profiles.png
"""
from __future__ import annotations

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType (Type 42) instead of Type 3 (preflight-friendly)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slod.boundary.scanner import boundary_scan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "..", "paper", "figures", "ablation_profiles.png")
CACHE_DIR = os.path.join(HERE, "..", "results", "exp1")


def profile(cache: dict, alpha_weights: tuple[float, float, float],
            peak_alpha: float = 2.0, source: str = "knn"):
    """Run boundary_scan once, return (sigma_x, scores, peaks, full_result).

    Convention (matches scanner.py::boundary_scan step 7 and every
    ablation runner): score index ``i`` maps to ``sigma_grid[i]`` (the
    *start* of the (sigma_i, sigma_{i+1}) interval between which the
    indicator was computed). Peaks are indices into ``scores`` and are
    displayed at the same ``sigma_grid[i]`` locations, so that the σ*
    marked on this plot matches the σ* stored by the runners and the
    σ* used internally by ``_effective_dimensionality``.
    """
    if source == "knn":
        lap = cache["laplacian_knn"]
        evs = cache["eigenvalues_knn"]
        evecs = cache["eigenvectors_knn"]
    else:
        lap = cache["laplacian_direct"]
        evs = cache["eigenvalues_direct"]
        evecs = cache["eigenvectors_direct"]
    res = boundary_scan(
        points=cache["points"],
        graph_laplacian=lap,
        focus_idx=0,
        alpha_weights=alpha_weights,
        peak_alpha=peak_alpha,
        eigenvalues=evs,
        eigenvectors=evecs,
    )
    # scores has length T-1; scanner's convention is sigma_grid[peak_idx]
    # (start of the (sigma_i, sigma_{i+1}) interval). We plot on the same
    # grid so peak markers line up with the σ* that runners store and
    # _effective_dimensionality uses. Using sigma_grid[1:] here would
    # shift the markers by one log-step and no longer match the table.
    return res.sigma_grid[:-1], res.scores, res.peaks, res


def load_cache(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_poincare_cache(r: int, seed: int) -> str | None:
    """Try per-(r, seed) pickle, then legacy r-only pickle, then global legacy."""
    candidates = [
        os.path.join(CACHE_DIR, f"ablation_cache_r{r}_s{seed}.pkl"),
        os.path.join(CACHE_DIR, f"ablation_cache_r{r}.pkl"),
    ]
    if seed == 42 and r == 200:
        candidates.append(os.path.join(CACHE_DIR, "ablation_cache.pkl"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main(r: int = 200, seed: int = 42) -> None:
    p_path = _find_poincare_cache(r, seed)
    e_path = os.path.join(CACHE_DIR, f"ablation_cache_euclidean_r{r}_s{seed}.pkl")
    if p_path is None:
        raise SystemExit(f"No Poincaré cache found for r={r} seed={seed}")
    print(f"[load] Poincaré: {p_path}")
    p_cache = load_cache(p_path)
    e_cache = load_cache(e_path)

    default = (1 / 3, 1 / 3, 1 / 3)
    configs = [
        ("full (Poincaré)",   default,           "knn",    "firebrick",  "-"),
        ("V-only",            (1.0, 0.0, 0.0),   "knn",    "#1f77b4",     "--"),
        ("D-only",            (0.0, 1.0, 0.0),   "knn",    "#2ca02c",     "--"),
        ("C-only",            (0.0, 0.0, 1.0),   "knn",    "#ff7f0e",     "--"),
        ("direct Laplacian",  default,           "direct", "gray",        ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # Panel (a): Poincaré composite profiles
    ax = axes[0]
    for name, weights, src, color, ls in configs:
        sx, scores, peaks, _ = profile(p_cache, weights, source=src)
        ax.plot(sx, scores, label=name, color=color, linestyle=ls, linewidth=1.4, alpha=0.85)
        if peaks:
            best = int(max(peaks, key=lambda i: scores[i]))
            ax.plot(sx[best], scores[best], "v", color=color, markersize=9,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"Scale $\sigma$ (log)", fontsize=11)
    ax.set_ylabel(r"Composite score $S(\sigma)$ (z-scored)", fontsize=11)
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.set_title(f"(a) Poincaré, $r{{=}}{r}$, seed={seed}", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25, which="both")

    # Panel (b): Euclidean vs Poincaré full profile (geometry comparison)
    ax = axes[1]
    if e_cache is not None:
        sx, scores, peaks, _ = profile(p_cache, default, source="knn")
        ax.plot(sx, scores, label="Poincaré full", color="firebrick",
                linewidth=1.6, alpha=0.9)
        if peaks:
            best = int(max(peaks, key=lambda i: scores[i]))
            ax.plot(sx[best], scores[best], "v", color="firebrick", markersize=10,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)

        sx_e, sc_e, peaks_e, _ = profile(e_cache, default, source="knn")
        ax.plot(sx_e, sc_e, label="Euclidean full", color="#1f77b4",
                linewidth=1.6, linestyle="--", alpha=0.9)
        if peaks_e:
            best_e = int(max(peaks_e, key=lambda i: sc_e[i]))
            ax.plot(sx_e[best_e], sc_e[best_e], "s", color="#1f77b4", markersize=9,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)
        ax.set_title(f"(b) Geometry comparison, full composite, $r{{=}}{r}$", fontsize=12)
        ax.legend(fontsize=9, loc="best")
    else:
        ax.text(0.5, 0.5, "No Euclidean cache", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel(r"Scale $\sigma$ (log)", fontsize=11)
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(r=args.r, seed=args.seed)
