#!/usr/bin/env python
"""Visualizations for Exp 2: WordNet Hierarchical Consistency.

Reads results/exp2*/full_results.json and generates:
1. Sigma-depth scatter with Kendall tau annotation
2. Precision & Recall bar chart
3. MRR comparison (focus vs SLoD)
4. Peak sigma distribution histogram
5. Peaks per focus node histogram
6. Embedding quality gates
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Embed TrueType (Type 42) instead of default Type 3 — preflight-friendly
# (CEUR-WS, arXiv) and PDF text remains searchable.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def load_results(results_dir: str) -> dict:
    """Load full_results.json."""
    path = os.path.join(results_dir, "full_results.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run run_exp2.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_sigma_depth_scatter(data: dict, out_dir: str) -> None:
    """Scatter: detected sigma vs nearest TRUE ancestor depth."""
    all_sigmas = []
    all_true_depths = []
    has_true = "nearest_true_depths" in data["scan_results"][0]

    for sr in data["scan_results"]:
        sigmas = sr["peak_sigmas"]
        if has_true:
            true_d = sr["nearest_true_depths"]
            for s, d in zip(sigmas, true_d):
                if d is not None:
                    all_sigmas.append(s)
                    all_true_depths.append(d)
        else:
            # Fallback for old data without nearest_true_depths
            all_sigmas.extend(sigmas)
            all_true_depths.extend(sr["peak_depths"])

    if not all_sigmas:
        print("  No peaks detected — skipping sigma-depth scatter.")
        return

    tau = data["metrics"]["kendall_tau"]
    max_depth = data["max_depth"]

    fig, ax = plt.subplots(figsize=(8, 5))
    # Jitter integer depths slightly for visibility
    jittered = np.array(all_true_depths) + np.random.default_rng(42).uniform(
        -0.15, 0.15, len(all_true_depths)
    )
    ax.scatter(all_sigmas, jittered, alpha=0.4, s=12, c="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel(r"Detected $\sigma^*$")
    ylabel = "Nearest ancestor depth" if has_true else "Mapped depth level"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Exp 2: Boundary Scale vs Ancestor Depth "
        rf"($\tau$ = {tau:.2f})"
    )
    ax.invert_yaxis()  # Depth 0 (root) at top
    # Horizontal lines at integer depths
    for d in range(max_depth + 1):
        ax.axhline(d, color="gray", linewidth=0.3, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "sigma_depth_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_precision_recall(data: dict, out_dir: str) -> None:
    """Bar chart: Precision@0.5 and Recall@1, Recall@2."""
    metrics = data["metrics"]
    labels = ["Prec@0.5", "Recall@1", "Recall@2"]
    values = [
        metrics.get("precision_at_05", 0.0),
        metrics.get("recall_at_1", 0.0),
        metrics.get("recall_at_2", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#e74c3c", "#3498db", "#2ecc71"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction")
    ax.set_title("Exp 2: Precision & Recall — Boundary Detection")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 baseline")
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "precision_recall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_mrr_comparison(data: dict, out_dir: str) -> None:
    """Bar chart: MRR_focus vs MRR_slod."""
    metrics = data["metrics"]
    mrr_focus = metrics.get("mrr_focus", 0.0)
    mrr_slod = metrics.get("mrr_slod", 0.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(
        ["MRR (focus)", "MRR (SLoD)"],
        [mrr_focus, mrr_slod],
        color=["#3498db", "#e67e22"],
    )
    ax.set_ylim(0, max(mrr_focus, mrr_slod) * 1.4 + 0.05)
    ax.set_ylabel("MRR")
    ax.set_title("Exp 2: Ancestor Retrieval (MRR)")

    for bar, val in zip(bars, [mrr_focus, mrr_slod]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "mrr_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_peak_distribution(data: dict, out_dir: str) -> None:
    """Histogram: distribution of peak sigmas across all focus nodes."""
    all_sigmas = []
    for sr in data["scan_results"]:
        all_sigmas.extend(sr["peak_sigmas"])

    if not all_sigmas:
        print("  No peaks — skipping distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        np.log10(all_sigmas), bins=40,
        color="steelblue", edgecolor="white", alpha=0.8,
    )
    ax.set_xlabel(r"$\log_{10}(\sigma^*)$")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Exp 2: Distribution of Detected Boundary Scales "
        f"(n={len(all_sigmas)})"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "peak_sigma_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_peaks_per_focus(data: dict, out_dir: str) -> None:
    """Histogram: number of peaks per focus node."""
    peak_counts = data["peak_counts"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        peak_counts, bins=range(0, max(peak_counts) + 2),
        color="coral", edgecolor="white", alpha=0.8,
        align="left",
    )
    ax.set_xlabel("Number of detected peaks")
    ax.set_ylabel("Number of focus nodes")
    ax.set_title(
        f"Exp 2: Peaks per Focus Node "
        f"(avg={np.mean(peak_counts):.1f})"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "peaks_per_focus.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_quality_gates(data: dict, out_dir: str) -> None:
    """Bar chart: embedding quality gate scores vs thresholds."""
    quality = data["quality"]
    labels = ["Hierarchy\nPreservation", "Depth\nCorrelation", "Sibling\nProximity"]
    scores = [
        quality["hierarchy_preservation"],
        quality["depth_correlation"],
        quality["sibling_proximity"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color="steelblue", alpha=0.8, label="Score")
    # Only plot thresholds for HP and SP (DC relaxed for DAGs)
    threshold_x = [0, 2]
    threshold_vals = [0.90, 0.85]
    ax.scatter(threshold_x, threshold_vals, color="red", marker="_", s=200,
               linewidths=3, zorder=5, label="Gate threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Exp 2: Embedding Quality Gates")
    ax.legend()

    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "quality_gates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp 2 results")
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to results directory (default: results/exp2)",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "results", "exp2"
        )
    out_dir = os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading Exp 2 results from {results_dir}...")
    data = load_results(results_dir)

    print("\nGenerating plots:")
    plot_sigma_depth_scatter(data, out_dir)
    plot_precision_recall(data, out_dir)
    plot_mrr_comparison(data, out_dir)
    plot_peak_distribution(data, out_dir)
    plot_peaks_per_focus(data, out_dir)
    plot_quality_gates(data, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
