#!/usr/bin/env python
"""Visualize Exp 1 results: BoundaryScan on 3-level HSBM.

Generates 4 plots:
  A: Score(σ) profiles per focus node (one subplot per r)
  B: Histogram of peak σ values across all foci → cluster structure
  C: K*(σ) — effective dimensionality drop across scale
  D: Summary table: rho, peak counts, peak bands
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType (Type 42) instead of default Type 3 — preflight-friendly
# (CEUR-WS, arXiv) and PDF text remains searchable.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1", "figures")


def load_results() -> dict[int, dict]:
    """Load per-r result JSONs."""
    results = {}
    for r in [20, 10, 5, 2]:
        path = os.path.join(RESULTS_DIR, f"full_r{r}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[r] = json.load(f)
    return results


def plot_a_score_profiles(results: dict[int, dict]) -> None:
    """Plot A: Score(σ) profiles for each focus node, one subplot per r."""
    r_values = sorted(results.keys(), reverse=True)
    fig, axes = plt.subplots(len(r_values), 1, figsize=(12, 3.5 * len(r_values)), sharex=True)
    if len(r_values) == 1:
        axes = [axes]

    for ax, r in zip(axes, r_values):
        data = results[r]
        sigma_grid = np.array(data["sigma_grid"])

        for fi, focus in enumerate(data["focus_results"]):
            scores = np.array(focus["score_profile"])
            # Scores are derivatives (length N-1), use sigma_grid[1:]
            sigma_x = sigma_grid[1:] if len(scores) == len(sigma_grid) - 1 else sigma_grid
            ax.plot(sigma_x, scores, alpha=0.7, label=f"Focus {fi} (node {focus['focus_node']})")

            # Mark peaks (raw, unfiltered)
            for peak in focus["peaks"]:
                ax.plot(peak["sigma"], peak["score"], "v", color="red", markersize=6, alpha=0.7)

        ax.set_xscale("log")
        ax.set_ylabel("Composite score S(σ)")
        ax.set_title(f"r = {r}  (ρ = {data['spearman_rho']:.3f}, "
                     f"avg peaks = {data['avg_peaks']:.1f})")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Scale σ (log)")
    fig.suptitle("Exp 1: BoundaryScan Score Profiles on 3-level HSBM", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "A_score_profiles.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "A_score_profiles.png"), dpi=150)
    plt.close(fig)
    print("  Plot A saved")


def plot_b_peak_histogram(results: dict[int, dict]) -> None:
    """Plot B: Histogram of peak σ values — are there 2 clear clusters?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, r in zip(axes.flat, sorted(results.keys(), reverse=True)):
        data = results[r]
        all_sigmas = []
        all_scores = []
        for focus in data["focus_results"]:
            for peak in focus["peaks"]:
                all_sigmas.append(peak["sigma"])
                all_scores.append(peak["score"])

        all_sigmas = np.array(all_sigmas)
        all_scores = np.array(all_scores)

        # Color by score magnitude
        ax.scatter(all_sigmas, all_scores, c=all_scores, cmap="RdYlGn", vmin=-1, vmax=5,
                   s=40, edgecolors="black", linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Peak σ (log)")
        ax.set_ylabel("Score S at peak")
        ax.set_title(f"r = {r}  ({len(all_sigmas)} peaks total)")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 1: Peak Distribution — σ vs Score", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "B_peak_distribution.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "B_peak_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Plot B saved")


def plot_c_effective_dims(results: dict[int, dict]) -> None:
    """Plot C: K*(σ) — effective dimensionality across scale."""
    r_values = sorted(results.keys(), reverse=True)
    fig, axes = plt.subplots(len(r_values), 1, figsize=(12, 3.5 * len(r_values)), sharex=True)
    if len(r_values) == 1:
        axes = [axes]

    for ax, r in zip(axes, r_values):
        data = results[r]
        sigma_grid = np.array(data["sigma_grid"])

        for fi, focus in enumerate(data["focus_results"]):
            eff_dims = focus["effective_dims"]
            # effective_dims is a dict of grid_idx -> K*
            indices = sorted(int(k) for k in eff_dims.keys())
            sigmas = [sigma_grid[i] for i in indices]
            k_stars = [eff_dims[str(i)] for i in indices]

            ax.plot(sigmas, k_stars, "o-", markersize=3, alpha=0.6,
                    label=f"Focus {fi}")

        ax.set_xscale("log")
        ax.set_ylabel("K* (effective dim)")
        ax.set_title(f"r = {r}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Scale σ (log)")
    fig.suptitle("Exp 1: Effective Dimensionality K*(σ)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "C_effective_dims.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "C_effective_dims.png"), dpi=150)
    plt.close(fig)
    print("  Plot C saved")


def plot_d_summary(results: dict[int, dict]) -> None:
    """Plot D: Summary — rho vs r, peak counts, peak band visualization."""
    r_values = sorted(results.keys(), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # D1: Spearman rho vs r
    ax = axes[0]
    rhos = [results[r]["spearman_rho"] for r in r_values]
    ax.plot(r_values, rhos, "o-", color="steelblue", markersize=8)
    ax.axhline(y=0.7, color="red", linewidth=1, linestyle="--", label="Quality gate (0.7)")
    ax.set_xlabel("r (hierarchy strength)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Embedding Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # D2: Average peak count vs r
    ax = axes[1]
    avg_peaks = [results[r]["avg_peaks"] for r in r_values]
    ax.bar(range(len(r_values)), avg_peaks, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(r_values)))
    ax.set_xticklabels([f"r={r}" for r in r_values])
    ax.axhline(y=2, color="green", linewidth=1, linestyle="--", label="Expected (2)")
    ax.set_ylabel("Avg peaks per focus node")
    ax.set_title("Peak Counts (raw, unfiltered)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # D3: All peak sigmas as violin/strip plot
    ax = axes[2]
    for i, r in enumerate(r_values):
        data = results[r]
        for focus in data["focus_results"]:
            for peak in focus["peaks"]:
                ax.scatter(i + np.random.uniform(-0.15, 0.15), peak["sigma"],
                           c="steelblue" if peak["score"] > 1.0 else "lightgray",
                           s=max(5, peak["score"] * 8), alpha=0.6, edgecolors="none")
    ax.set_yscale("log")
    ax.set_xticks(range(len(r_values)))
    ax.set_xticklabels([f"r={r}" for r in r_values])
    ax.set_ylabel("Peak σ (log)")
    ax.set_title("Peak Locations (blue=S>1, gray=S≤1)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 1: HSBM Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "D_summary.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "D_summary.png"), dpi=150)
    plt.close(fig)
    print("  Plot D saved")


def plot_e_ari_vs_sigma() -> None:
    """Plot E: ARI vs sigma — how well do spectral clusters match planted labels?"""
    group_path = os.path.join(RESULTS_DIR, "group_analysis_r20.json")
    if not os.path.exists(group_path):
        print("  Plot E skipped (no group_analysis_r20.json — run analyze_groups.py first)")
        return

    with open(group_path) as f:
        gdata = json.load(f)

    sweep = gdata["sweep_results"]
    sigmas = [r["sigma"] for r in sweep]
    ari_macro = [r["ari_macro"] for r in sweep]
    ari_meso = [r["ari_meso"] for r in sweep]
    nmi_macro = [r["nmi_macro"] for r in sweep]
    nmi_meso = [r["nmi_meso"] for r in sweep]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ARI
    ax = axes[0]
    ax.plot(sigmas, ari_macro, "o-", color="firebrick", markersize=5, label="ARI vs macro (k=2)")
    ax.plot(sigmas, ari_meso, "s-", color="steelblue", markersize=5, label="ARI vs meso (k=8)")
    ax.set_xscale("log")
    ax.set_xlabel("Scale σ (log)")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Spectral Clustering vs Planted Labels (ARI)")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark peak sigmas
    for peak in gdata.get("unique_peaks", []):
        ax.axvline(x=peak["sigma"], color="red", alpha=0.15, linewidth=1)

    # NMI
    ax = axes[1]
    ax.plot(sigmas, nmi_macro, "o-", color="firebrick", markersize=5, label="NMI vs macro (k=2)")
    ax.plot(sigmas, nmi_meso, "s-", color="steelblue", markersize=5, label="NMI vs meso (k=8)")
    ax.set_xscale("log")
    ax.set_xlabel("Scale σ (log)")
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("Spectral Clustering vs Planted Labels (NMI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for peak in gdata.get("unique_peaks", []):
        ax.axvline(x=peak["sigma"], color="red", alpha=0.15, linewidth=1)

    fig.suptitle("Exp 1 (r=20): Do Spectral Clusters Match Planted Labels?", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "E_ari_vs_sigma.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "E_ari_vs_sigma.png"), dpi=150)
    plt.close(fig)
    print("  Plot E saved")


def plot_f_ablation() -> None:
    """Plot F: Direct graph vs kNN-of-embedding — ARI comparison + eigenvalue profiles."""
    knn_path = os.path.join(RESULTS_DIR, "group_analysis_r20.json")
    direct_path = os.path.join(RESULTS_DIR, "group_analysis_direct_r20.json")
    if not os.path.exists(knn_path) or not os.path.exists(direct_path):
        print("  Plot F skipped (need both group_analysis_r20.json and "
              "group_analysis_direct_r20.json)")
        return

    with open(knn_path) as f:
        knn_data = json.load(f)
    with open(direct_path) as f:
        direct_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # F1: ARI macro comparison
    ax = axes[0]
    knn_sweep = knn_data["sweep_results"]
    dir_sweep = direct_data["sweep_results"]

    ax.plot([r["sigma"] for r in knn_sweep], [r["ari_macro"] for r in knn_sweep],
            "o-", color="steelblue", markersize=4, label="kNN-of-embedding")
    ax.plot([r["sigma"] for r in dir_sweep], [r["ari_macro"] for r in dir_sweep],
            "s-", color="firebrick", markersize=4, label="Direct graph")
    ax.set_xscale("log")
    ax.set_xlabel("Scale σ (log)")
    ax.set_ylabel("ARI vs macro (k=2)")
    ax.set_title("Macro Recovery: Direct vs Embedding")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F2: ARI meso comparison
    ax = axes[1]
    ax.plot([r["sigma"] for r in knn_sweep], [r["ari_meso"] for r in knn_sweep],
            "o-", color="steelblue", markersize=4, label="kNN-of-embedding")
    ax.plot([r["sigma"] for r in dir_sweep], [r["ari_meso"] for r in dir_sweep],
            "s-", color="firebrick", markersize=4, label="Direct graph")
    ax.set_xscale("log")
    ax.set_xlabel("Scale σ (log)")
    ax.set_ylabel("ARI vs meso (k=8)")
    ax.set_title("Meso Recovery: Direct vs Embedding")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F3: Eigenvalue profiles
    ax = axes[2]
    dir_evals = np.array(direct_data["eigenvalues"][:20])
    ax.plot(range(len(dir_evals)), dir_evals, "o-", color="firebrick",
            markersize=5, label="Direct graph")

    # Mark planted levels
    for k, lbl in [(2, "k=2 (macro)"), (8, "k=8 (meso)")]:
        if k < len(dir_evals):
            ax.axvline(x=k, color="gray", linewidth=1, linestyle="--", alpha=0.5)
            ax.annotate(lbl, xy=(k, dir_evals[min(k, len(dir_evals) - 1)]),
                        fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title("Eigenvalue Profile (Direct Graph)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 1 Ablation (r=20): Direct Graph vs kNN-of-Embedding", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "F_ablation.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "F_ablation.png"), dpi=150)
    plt.close(fig)
    print("  Plot F saved")


def plot_g_multir_phase() -> None:
    """Plot G: Multi-r phase diagram — ARI and spectral gap vs r.

    Shows the phase transition at the Kesten-Stigum threshold:
    - Left: ARI macro + meso vs r (with KS threshold marked)
    - Center: Spectral gap ratio vs r
    - Right: SNR vs r (with threshold line at SNR=1)
    """
    multir_path = os.path.join(RESULTS_DIR, "multir_direct.json")
    if not os.path.exists(multir_path):
        print("  Plot G skipped (no multir_direct.json — run run_exp1_multir.py first)")
        return

    with open(multir_path) as f:
        data = json.load(f)

    r_vals = [d["r"] for d in data]
    ari_mac = [d["fixed_ari_macro"] for d in data]
    ari_mes = [d["fixed_ari_meso"] for d in data]
    gap_2 = [d["gap_ratio_2"] for d in data]
    snr_mac = [d["snr_macro"] for d in data]
    snr_mes = [d["snr_meso"] for d in data]

    # ARI at crossings (where K* reaches planted level)
    cross_ari_mac = []
    cross_ari_mes = []
    cross_r_mac = []
    cross_r_mes = []
    for d in data:
        if "2" in d["crossings"]:
            cross_r_mac.append(d["r"])
            cross_ari_mac.append(d["crossings"]["2"]["ari"])
        if "8" in d["crossings"]:
            cross_r_mes.append(d["r"])
            cross_ari_mes.append(d["crossings"]["8"]["ari"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # G1: ARI vs r
    ax = axes[0]
    ax.plot(r_vals, ari_mac, "o-", color="firebrick", markersize=7,
            linewidth=2, label="ARI macro (fixed k=2)")
    ax.plot(r_vals, ari_mes, "s-", color="steelblue", markersize=7,
            linewidth=2, label="ARI meso (fixed k=8)")
    if cross_r_mac:
        ax.plot(cross_r_mac, cross_ari_mac, "^", color="firebrick",
                markersize=10, markeredgecolor="black", label="ARI at K*→2")
    if cross_r_mes:
        ax.plot(cross_r_mes, cross_ari_mes, "D", color="steelblue",
                markersize=8, markeredgecolor="black", label="ARI at K*→8")
    # KS threshold zone
    ax.axvspan(35, 45, alpha=0.15, color="orange", label="KS threshold (macro)")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("r (hierarchy strength)", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index", fontsize=12)
    ax.set_title("Community Recovery vs r", fontsize=13)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # G2: Spectral gap ratio vs r — multi-seed median + IQR band if available,
    # else fall back to single-seed line.
    ax = axes[1]
    seeds_path = os.path.join(RESULTS_DIR, "multir_seeds.json")
    if os.path.exists(seeds_path):
        with open(seeds_path) as f:
            sdata = json.load(f)
        agg = sdata["aggregated"]
        sr = [a["r"] for a in agg]
        sm = [a["gap_ratio_2_median"] for a in agg]
        sq25 = [a["gap_ratio_2_q25"] for a in agg]
        sq75 = [a["gap_ratio_2_q75"] for a in agg]
        n_seeds = sdata["per_seed"][0].get("seed", 0)  # informational
        n_used = len(sdata["seeds"])
        ax.fill_between(sr, sq25, sq75, color="darkgreen", alpha=0.20,
                        label=f"IQR ({n_used} seeds)")
        ax.plot(sr, sm, "o-", color="darkgreen", markersize=7, linewidth=2,
                label=r"$\lambda_3/\lambda_2$ median")
    else:
        ax.plot(r_vals, gap_2, "o-", color="darkgreen", markersize=7,
                linewidth=2, label=r"$\lambda_3/\lambda_2$ (single seed=42)")
    ax.axhline(y=2.0, color="red", linewidth=1.5, linestyle="--",
               label="Gap threshold (2.0)")
    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("r (hierarchy strength)", fontsize=12)
    ax.set_ylabel("Spectral gap ratio", fontsize=12)
    ax.set_title("Spectral Gap at k=2", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # G3: SNR vs r
    ax = axes[2]
    ax.plot(r_vals, snr_mac, "o-", color="firebrick", markersize=7,
            linewidth=2, label="SNR macro")
    ax.plot(r_vals, snr_mes, "s-", color="steelblue", markersize=7,
            linewidth=2, label="SNR meso")
    ax.axhline(y=1.0, color="red", linewidth=1.5, linestyle="--",
               label="KS threshold (SNR=1)")
    ax.set_xlabel("r (hierarchy strength)", fontsize=12)
    ax.set_ylabel("Signal-to-Noise Ratio", fontsize=12)
    ax.set_title("Kesten-Stigum SNR", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "G_multir_phase.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "G_multir_phase.png"), dpi=150)
    plt.close(fig)
    print("  Plot G saved")


def plot_h_kstar_trajectory() -> None:
    """Plot H: K*(σ) trajectories for all r values overlaid.

    Shows how hierarchy "emerges" in K*(σ) as r increases:
    - At low r: K* barely decays (flat, no hierarchy visible)
    - At high r: K* shows clear steps at planted levels (2, 8)
    """
    multir_path = os.path.join(RESULTS_DIR, "multir_direct.json")
    if not os.path.exists(multir_path):
        print("  Plot H skipped (no multir_direct.json)")
        return

    with open(multir_path) as f:
        data = json.load(f)

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        savgol_filter = None

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(data)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # H1: K*(σ) trajectories on log–log axes (planted levels equispaced)
    ax = axes[0]
    for i, d in enumerate(data):
        sigmas = np.array([p["sigma"] for p in d["kstar_curve"]])
        kstars = np.array([p["k_star"] for p in d["kstar_curve"]], dtype=float)
        is_high_r = d["r"] >= 150
        ax.plot(sigmas, kstars, "-", color=colors[i],
                linewidth=2.6 if is_high_r else 1.1,
                alpha=0.95 if is_high_r else 0.45,
                label=f"r={d['r']}")

    # Mark planted levels (equispaced on log y) — labels on the right edge
    for k_lev, lab_color, lab in [
        (64, "gray",       r"K=64 (micro)"),
        (8,  "steelblue",  r"K=8 (meso)"),
        (2,  "firebrick",  r"K=2 (macro)"),
    ]:
        ax.axhline(y=k_lev, color=lab_color, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(95, k_lev, f" {lab}", color=lab_color, fontsize=8,
                va="center", ha="right", alpha=0.95,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.7, 90)
    ax.set_xlabel(r"Scale $\sigma$ (log)", fontsize=12)
    ax.set_ylabel(r"$K^*(\sigma)$ (log)", fontsize=12)
    ax.set_title(r"$K^*(\sigma)$ Trajectory: Sweeps Through Planted Levels", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)

    # H2: smoothed |dK*/d log σ| for select r
    ax = axes[1]
    select_r = [20, 60, 100, 200]
    select_colors = {20: "gray", 60: "orange", 100: "steelblue", 200: "firebrick"}

    for d in data:
        if d["r"] not in select_r:
            continue
        sigmas = np.array([p["sigma"] for p in d["kstar_curve"]])
        kstars = np.array([p["k_star"] for p in d["kstar_curve"]], dtype=float)
        log_sigma = np.log10(sigmas)
        deriv = np.abs(np.gradient(kstars, log_sigma))
        if savgol_filter is not None and len(deriv) >= 11:
            deriv = savgol_filter(deriv, window_length=11, polyorder=3, mode="nearest")
        deriv = np.clip(deriv, 0, None)
        ax.plot(sigmas, deriv, "-", color=select_colors[d["r"]],
                linewidth=2.0, alpha=0.9, label=f"r={d['r']}")

    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xlabel(r"Scale $\sigma$ (log)", fontsize=12)
    ax.set_ylabel(r"$|dK^*/d\log\sigma|$  (smoothed)", fontsize=12)
    ax.set_title("Spectral Boundary Indicator (Savitzky--Golay smoothed)", fontsize=12)

    # Mark σ-locations where K*(σ)=64, 8, 2 for r=200 — boundary scales the indicator should highlight
    r200 = next((d for d in data if d["r"] == 200), None)
    if r200 is not None:
        sigmas_arr = np.array([p["sigma"] for p in r200["kstar_curve"]])
        kstars_arr = np.array([p["k_star"] for p in r200["kstar_curve"]])
        ymax = ax.get_ylim()[1]
        # Get individual σ for each level
        sig_locs = {}
        for k_lev in (64, 8, 2):
            idx = int(np.argmin(np.abs(kstars_arr - k_lev)))
            sig_locs[k_lev] = float(sigmas_arr[idx])
        # K=64 stand-alone (well separated)
        ax.axvline(x=sig_locs[64], color="gray", linewidth=1.4,
                   linestyle="--", alpha=0.85)
        ax.text(sig_locs[64], ymax * 0.97, " K=64", color="gray",
                fontsize=9, va="top", ha="left", alpha=0.95)
        # K=8 and K=2 are 1.2× apart in σ at r=200 (saturation regime) —
        # render both lines but label as a band
        ax.axvline(x=sig_locs[8], color="steelblue", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.axvline(x=sig_locs[2], color="firebrick", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.axvspan(sig_locs[8], sig_locs[2], color="orange", alpha=0.12)
        ax.text(np.sqrt(sig_locs[8] * sig_locs[2]), ymax * 0.97,
                "K=8 → K=2", color="black", fontsize=9, va="top", ha="center",
                alpha=0.9, bbox=dict(facecolor="white", edgecolor="none",
                                     alpha=0.75, pad=1))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(r"Exp 1: $K^*(\sigma)$ as Continuous Level of Detail", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "H_kstar_trajectory.pdf"), dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, "H_kstar_trajectory.png"), dpi=150)
    plt.close(fig)
    print("  Plot H saved")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading results...")
    results = load_results()
    if not results:
        print("No results found. Run run_exp1.py first.")
        return

    # Check if new format (with focus_results)
    sample = next(iter(results.values()))
    if "focus_results" not in sample:
        print("Results in old format (no focus_results). Re-run run_exp1.py to get full data.")
        return

    print(f"Found results for r = {sorted(results.keys(), reverse=True)}")
    print("\nGenerating plots...")
    plot_a_score_profiles(results)
    plot_b_peak_histogram(results)
    plot_c_effective_dims(results)
    plot_d_summary(results)
    plot_e_ari_vs_sigma()
    plot_f_ablation()
    plot_g_multir_phase()
    plot_h_kstar_trajectory()
    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
