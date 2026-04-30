#!/usr/bin/env python
"""ARI vs r curves across all ablation configs, with multi-seed CI bands.

Summarises the full ablation sweep in two panels (macro / meso ARI vs r).
Reads `results/exp1/ablation_agg.json` produced by aggregate_ablation.py.
Output: paper/figures/ablation_curves.png.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN_JSON = os.path.join(HERE, "..", "results", "exp1", "ablation_agg.json")
OUT_PNG = os.path.join(HERE, "..", "paper", "figures", "ablation_curves.png")


def bootstrap_bounds(values: list[float], n: int = 2000) -> tuple[float, float, float]:
    if len(values) < 2:
        v = float(values[0]) if values else float("nan")
        return v, v, v
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.median(values)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def extract_series(entry: dict, r_values: list[int], metric: str
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med, lo, hi, ns = [], [], [], []
    for r in r_values:
        key = str(r)
        vals = entry["by_r"].get(key, {}).get(f"ari_{metric}_values", [])
        m, l, h = bootstrap_bounds(vals) if vals else (float("nan"),) * 3
        med.append(m); lo.append(l); hi.append(h); ns.append(len(vals))
    return np.array(med), np.array(lo), np.array(hi), np.array(ns)


def main() -> None:
    with open(IN_JSON) as f:
        agg = json.load(f)

    # pick key configs to display (skip MAD variants — redundant with full)
    wanted = [
        ("full",                    "Poincaré full",       "firebrick",  "-",  "o"),
        ("V-only",                  "V-only",              "#1f77b4",    "--", "s"),
        ("D-only",                  "D-only",              "#2ca02c",    "--", "D"),
        ("C-only",                  "C-only",              "#ff7f0e",    "--", "^"),
        ("direct Laplacian",        "direct Laplacian",    "gray",       ":",  "x"),
        ("full (euclidean)",        "Euclidean full",      "#1f77b4",    "-.", "*"),
        ("full (binary kNN)",       "Poincaré + binary kNN","firebrick", ":",  "v"),
    ]
    by_name = {e["name"]: e for e in agg}
    r_values = [20, 40, 60, 80, 100, 150, 200]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for panel_idx, (ax, metric, title, klevel) in enumerate(
        [(axes[0], "macro", r"ARI$_{\mathrm{mac}}$  ($K^*{=}2$)", 2),
         (axes[1], "meso",  r"ARI$_{\mathrm{mes}}$  ($K^*{=}8$)", 8)]):
        for name, label, color, ls, marker in wanted:
            if name not in by_name:
                continue
            med, lo, hi, ns = extract_series(by_name[name], r_values, metric)
            # only plot r values where data exists
            mask = ~np.isnan(med)
            ax.plot(np.array(r_values)[mask], med[mask], linestyle=ls, marker=marker,
                    color=color, label=label, linewidth=1.4, markersize=5)
            if mask.any() and any(ns[mask] >= 3):
                ax.fill_between(
                    np.array(r_values)[mask], lo[mask], hi[mask],
                    color=color, alpha=0.12,
                )
        ax.set_xlabel(r"Inter-level ratio $r$ (hierarchy strength)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xticks(r_values)
        ax.axhline(0, color="lightgray", linewidth=0.5)
        # Kesten–Stigum zone (approximate 30-45 for this HSBM shape)
        ax.axvspan(30, 45, alpha=0.08, color="orange",
                   label="KS threshold zone" if panel_idx == 0 else None)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"({chr(ord('a')+panel_idx)}) {title} vs $r$  (n=5 seeds, shaded 95% CI)",
                     fontsize=12)
        if panel_idx == 0:
            ax.legend(fontsize=7.5, loc="lower right", ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
