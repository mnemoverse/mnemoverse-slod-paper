#!/usr/bin/env python
"""Aggregate ablation data across four sidecars and emit CI-aware tables.

Inputs (merged):
  results/exp1/ablation.json            — main Poincaré sweep (multi-seed)
  results/exp1/ablation_euclidean.json  — Euclidean MDS baseline (seed=42)
  results/exp1/ablation_binary_knn.json — Binary (unweighted) kNN variant
  results/exp1/ablation_random_points.json — Random-points control

Outputs:
  results/exp1/ablation_agg.json  — per-(config, r) aggregated records
                                    with bootstrap 95% CIs.
  paper/tables/ablation.tex — two stacked LaTeX tables
                                    (macro + meso ARI).

Design choices addressing Copilot review comments on PR #60:
- Row labels use LaTeX math mode (`lax ($\\alpha{=}1$)` etc.).
- No duplicate `(direct)` suffix — config names already disambiguate.
- ARI cells clamp `-0.00` → `0.00` (no cosmetic negative zero in camera-ready).
- Table scope is a curated, publication-ready subset
  (KEEP_CONFIGS below); the JSON remains complete for the appendix /
  arXiv-v2 long form.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN_MAIN = os.path.join(HERE, "..", "results", "exp1", "ablation.json")
IN_EUCL = os.path.join(HERE, "..", "results", "exp1", "ablation_euclidean.json")
IN_BIN = os.path.join(HERE, "..", "results", "exp1", "ablation_binary_knn.json")
IN_RND = os.path.join(HERE, "..", "results", "exp1", "ablation_random_points.json")
OUT_JSON = os.path.join(HERE, "..", "results", "exp1", "ablation_agg.json")
OUT_TEX = os.path.join(HERE, "..", "paper", "tables", "ablation.tex")

# Curated row order for the paper table. The JSON keeps every config
# for the appendix / arXiv-v2 long form.
KEEP_CONFIGS: list[tuple[str, str]] = [
    # (internal name, LaTeX row label)
    ("full",                  r"full (Poincaré)"),
    ("V-only",                r"$V$-only"),
    ("D-only",                r"$D_w$-only"),
    ("C-only",                r"$C_k$-only"),
    ("direct Laplacian",      r"direct Laplacian"),
    ("full (euclidean)",      r"full (Euclidean MDS)"),
    ("full (binary kNN)",     r"full (binary kNN)"),
    ("full (random points)",  r"full (random points)"),
]


def bootstrap_ci(
    values: list[float],
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "BCa",
) -> tuple[float, float, float]:
    """Median + (1−α) bootstrap CI on the median.

    Two methodology fixes vs. the pre-2026-04-25 implementation:
    (i) point estimate AND CI are both on the median (was: median point
        estimate + mean-of-resamples CI — inconsistent on skewed ARI
        distributions);
    (ii) BCa (bias-corrected and accelerated) method by default — better
        coverage than the percentile method for small n_seeds and skewed
        statistics. Falls back to percentile if scipy is unavailable or the
        statistic has zero variance (BCa formula divides by jackknife
        variance).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        nan = float("nan")
        return nan, nan, nan
    if len(arr) == 1:
        v = float(arr[0])
        return v, v, v
    point = float(np.median(arr))

    if method == "BCa":
        try:
            from scipy.stats import bootstrap as _scipy_bootstrap
            rng = np.random.default_rng(seed)
            res = _scipy_bootstrap(
                (arr,),
                statistic=lambda x: np.median(x, axis=-1),
                n_resamples=n,
                confidence_level=1.0 - alpha,
                method="BCa",
                random_state=rng,
                vectorized=True,
            )
            return point, float(res.confidence_interval.low), float(res.confidence_interval.high)
        except Exception:
            # BCa requires non-zero jackknife variance; fall through to percentile.
            pass

    # Percentile fallback (or method='percentile' explicitly).
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(arr, size=(n, len(arr)), replace=True), axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, lo, hi


def load_records() -> list[dict]:
    recs: list[dict] = []
    for path in (IN_MAIN, IN_EUCL, IN_BIN, IN_RND):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            recs.extend(json.load(f))
    return recs


def _clamp_neg_zero(x: float) -> float:
    """Convert tiny negative floats to +0.0 so cells read "0.00" not "-0.00"."""
    if -5e-3 < x < 0:
        return 0.0
    return x


def fmt_cell(values: list[float]) -> str:
    """median [lo, hi] when ≥3 seeds; single value otherwise; '--' if missing."""
    if len(values) == 0:
        return "--"
    if len(values) >= 3:
        med, lo, hi = bootstrap_ci(values)
        return f"{_clamp_neg_zero(med):.2f} [{_clamp_neg_zero(lo):.2f}, {_clamp_neg_zero(hi):.2f}]"
    return f"{_clamp_neg_zero(values[0]):.2f}"


def fmt_weights(w: list[float]) -> str:
    a, b, c = w
    if abs(a - 1 / 3) < 1e-3 and abs(b - 1 / 3) < 1e-3 and abs(c - 1 / 3) < 1e-3:
        return r"$(\tfrac{1}{3},\tfrac{1}{3},\tfrac{1}{3})$"
    return f"$({a:g},{b:g},{c:g})$"


def aggregate(recs: list[dict]) -> tuple[list[dict], list[int], dict[str, dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in recs:
        grouped[(r["name"], int(r.get("r", 200)))].append(r)

    config_meta: dict[str, dict] = {}
    for r in recs:
        name = r["name"]
        if name not in config_meta:
            config_meta[name] = {
                "alpha_weights": r["alpha_weights"],
                "peak_alpha": r["peak_alpha"],
                "laplacian_source": r["laplacian_source"],
                "embedding_type": r.get("embedding_type", "poincare"),
            }

    r_values = sorted({int(r.get("r", 200)) for r in recs})

    agg: list[dict] = []
    for name, meta in config_meta.items():
        entry = {"name": name, **meta, "by_r": {}}
        for r in r_values:
            macros = [rec["ari_macro"] for rec in grouped.get((name, r), [])]
            mesos = [rec["ari_meso"] for rec in grouped.get((name, r), [])]
            sigmas = [rec["sigma_best"] for rec in grouped.get((name, r), [])]
            entry["by_r"][str(r)] = {
                "n_seeds": len(macros),
                "ari_macro_median": float(np.median(macros)) if macros else None,
                "ari_meso_median": float(np.median(mesos)) if mesos else None,
                "sigma_best_median": float(np.median(sigmas)) if sigmas else None,
                "ari_macro_values": macros,
                "ari_meso_values": mesos,
            }
        agg.append(entry)
    return agg, r_values, config_meta


def _emit_table(metric: str, caption: str, label: str,
                config_meta: dict[str, dict], grouped: dict,
                r_values: list[int]) -> list[str]:
    n_r = len(r_values)
    col_spec = "@{}l" + "c" * n_r + "@{}"
    r_hdr = " & ".join([r"$r{=}" + str(r) + "$" for r in r_values])

    rows: list[str] = []
    for internal, display in KEEP_CONFIGS:
        if internal not in config_meta:
            continue
        cells = [
            fmt_cell([rec[f"ari_{metric}"] for rec in grouped.get((internal, r), [])])
            for r in r_values
        ]
        rows.append(" & ".join([display] + cells) + r" \\")

    return [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\footnotesize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Config & " + r_hdr + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]


def main() -> None:
    recs = load_records()
    print(f"Loaded {len(recs)} records")

    agg, r_values, config_meta = aggregate(recs)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregated: {OUT_JSON}")

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in recs:
        grouped[(r["name"], int(r.get("r", 200)))].append(r)

    r_list_txt = ", ".join(str(r) for r in r_values)
    # Estimate seed count from the data — pick the modal count across cells
    # and fall back to a sensible default if the data isn't there yet.
    cell_counts = [len(v) for v in grouped.values()]
    n_seeds = max(set(cell_counts), key=cell_counts.count) if cell_counts else 5
    shared_caption_suffix = (
        f" on HSBM across $r \\in \\{{{r_list_txt}\\}}$. $N{{=}}1024$, "
        r"planted 2{$\times$}4{$\times$}8 hierarchy. Cells are medians "
        f"over {n_seeds} Poincar\\'e-MDS seeds with bootstrap (BCa, "
        r"$n{=}10{,}000$) 95\% CIs; CIs are uncorrected for multiple "
        r"comparisons across the $8\times7$ ablation grid. "
        r"\emph{full (Euclidean MDS)} swaps the embedding; \emph{full "
        r"(binary kNN)} strips Gaussian edge weights; \emph{full "
        r"(random points)} replaces the Poincar\'e coordinates with "
        r"i.i.d.\ Gaussian noise while keeping the same Poincar\'e-kNN "
        r"Laplacian (isolates the role of the points for the Fr\'echet "
        r"mean step). \emph{direct Laplacian} uses the combinatorial "
        r"Laplacian of the HSBM graph instead of a kNN-of-embedding "
        r"Laplacian (isolates the Laplacian construction)."
    )

    macro_tbl = _emit_table(
        "macro",
        r"Ablation: macro-scale ARI ($K^*{=}2$)" + shared_caption_suffix,
        "tab:ablation_macro",
        config_meta, grouped, r_values,
    )
    meso_tbl = _emit_table(
        "meso",
        r"Ablation: meso-scale ARI ($K^*{=}8$)" + shared_caption_suffix,
        "tab:ablation_meso",
        config_meta, grouped, r_values,
    )

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(macro_tbl) + "\n\n" + "\n".join(meso_tbl) + "\n")
    print(f"Wrote {OUT_TEX}")
    print(f"Kept {sum(1 for k,_ in KEEP_CONFIGS if k in config_meta)}/{len(KEEP_CONFIGS)} "
          f"curated configs; r values: {r_values}")


if __name__ == "__main__":
    main()
