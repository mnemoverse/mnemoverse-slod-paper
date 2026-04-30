#!/usr/bin/env python
"""Ablation study for SLoD BoundaryScan (closes #39).

Sweeps composite-score weights (α₁, α₂, α₃), peak-picker threshold
(peak_alpha), and Laplacian source (kNN-of-embedding vs direct graph)
across the full range r ∈ {20, 40, 60, 80, 100, 150, 200} with 5 seeds
per r (42..46). Produces results/exp1/ablation.json (resumable).

Design:
- Reconstruct HSBM + Poincaré MDS + kNN Laplacian ONCE per (r, seed)
  (slow — ~6–8 min), cached per-pair to
  results/exp1/ablation_cache_r{r}_s{seed}.pkl for subsequent runs.
- Reconstruct direct Laplacian in the same pickle (fast, <5s).
- For each ablation config, reuse precomputed eigenpairs with
  boundary_scan (alpha_weights, peak_alpha vary; graph/eigendecomp fixed).
- Score each config via fixed-K spectral clustering ARI at macro/meso.

Total time budget: ~4.5 h of MDS for a fresh full sweep (35 pairs). With
caches populated, the inner loop runs in a few minutes.

Sidecar scripts (separate JSON outputs, aggregated by
aggregate_ablation.py) add further ablation axes without racing this
file's writes:
- ablation_runner_euclidean.py       — swap Poincaré MDS for Euclidean
- ablation_runner_binary_knn.py      — strip Gaussian edge weights
- ablation_runner_random_points.py   — replace points with Gaussian noise
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import networkx as nx
import numpy as np
import scipy.sparse.linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exp1_hsbm"))

from slod.boundary.scanner import boundary_scan
from slod.boundary.spectral import build_knn_graph, normalized_laplacian
from slod.utils.data import generate_hsbm
from slod.utils.data import hyperbolic_mds
from slod.utils.metrics import adjusted_rand_index
from sklearn.metrics import normalized_mutual_info_score

from analyze_groups import analyze_at_fixed_k


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp1")
OUT_PATH = os.environ.get("SLOD_OUT_PATH") or os.path.join(RESULTS_DIR, "ablation.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cache schema version. Bump whenever the data dict layout, MDS hyperparams,
# kNN construction, or Laplacian computation changes. Caches with a different
# version are silently regenerated rather than reused (prevents seed=42 from
# using a stale pickle while seeds 43+ get fresh ones — the root cause of the
# 5→10 seed claim-flip pre-2026-04-25).
CACHE_SCHEMA_VERSION = 2


def cache_path(r: float, seed: int = 42) -> str:
    """Per-(r, seed) pickle path."""
    return os.path.join(RESULTS_DIR, f"ablation_cache_r{int(r)}_s{int(seed)}.pkl")


def _atomic_pickle_dump(obj, path: str) -> None:
    """Pickle to path via temp-file + atomic rename to avoid partial writes."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _atomic_json_dump(obj, path: str) -> None:
    """JSON-serialize via temp-file + atomic rename. Survives SIGINT mid-write."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def reconstruct_cached(r: float = 200.0, seed: int = 42, use_cache: bool = True) -> dict:
    """Build HSBM + Poincaré embedding + kNN Laplacian. Cache per-(r, seed) to pickle.

    Cache invalidation: caches with a different ``schema_version`` than the
    current ``CACHE_SCHEMA_VERSION`` are regenerated. There is no legacy
    (r-only or global) fallback — for the camera-ready pristine rerun every
    (r, seed) pair must come from the same code path.
    """
    path = cache_path(r, seed)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            ver = cached.get("schema_version") if isinstance(cached, dict) else None
            if ver == CACHE_SCHEMA_VERSION:
                print(f"[cache] loading {path} (v{ver})")
                return cached
            print(f"[cache] {path} schema_version={ver!r} != {CACHE_SCHEMA_VERSION}; regenerating")
        except Exception as e:
            print(f"[cache] {path} unreadable ({e}); regenerating")

    print(f"[reconstruct] r={r}, seed={seed}")
    t0 = time.time()
    graph, labels = generate_hsbm(
        n_nodes=1024, n_macro=2, n_meso_per_macro=4, n_micro_per_meso=8,
        r=r, seed=seed,
    )
    if not nx.is_connected(graph):
        lcc = max(nx.connected_components(graph), key=len)
        mapping = {old: new for new, old in enumerate(sorted(lcc))}
        graph = nx.relabel_nodes(graph.subgraph(lcc).copy(), mapping)
        cc_indices = sorted(lcc)
        labels = {k: v[cc_indices] for k, v in labels.items()}
    print(f"  LCC: N={graph.number_of_nodes()}, M={graph.number_of_edges()}, t={time.time()-t0:.1f}s")

    t0 = time.time()
    # Opt-in GPU MDS via SLOD_DEVICE=cuda (default cpu = pristine CPU baseline).
    _device = os.environ.get("SLOD_DEVICE", "cpu")
    points = hyperbolic_mds(graph, dim=10, max_iter=5000, lr=0.2, seed=seed, device=_device)
    print(f"  MDS: t={time.time()-t0:.1f}s [device={_device}]")

    t0 = time.time()
    adj_knn = build_knn_graph(points)
    lap_knn = normalized_laplacian(adj_knn)
    k_eigs = min(50, graph.number_of_nodes() - 1)
    eigenvalues_knn, eigenvectors_knn = scipy.sparse.linalg.eigsh(
        lap_knn.astype(np.float64), k=k_eigs, which="SA"
    )
    eigenvalues_knn = np.maximum(eigenvalues_knn, 0.0)
    print(f"  kNN Laplacian + eigs: t={time.time()-t0:.1f}s")

    # Direct Laplacian (no embedding)
    t0 = time.time()
    adj_direct = nx.adjacency_matrix(graph).astype(np.float64)
    lap_direct = normalized_laplacian(adj_direct)
    eigenvalues_direct, eigenvectors_direct = scipy.sparse.linalg.eigsh(
        lap_direct.astype(np.float64), k=k_eigs, which="SA"
    )
    eigenvalues_direct = np.maximum(eigenvalues_direct, 0.0)
    print(f"  Direct Laplacian + eigs: t={time.time()-t0:.1f}s")

    data = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "r": r,
        "seed": seed,
        "n_nodes": graph.number_of_nodes(),
        "labels": labels,
        "points": points,
        "laplacian_knn": lap_knn,
        "eigenvalues_knn": eigenvalues_knn,
        "eigenvectors_knn": eigenvectors_knn,
        "laplacian_direct": lap_direct,
        "eigenvalues_direct": eigenvalues_direct,
        "eigenvectors_direct": eigenvectors_direct,
    }
    _atomic_pickle_dump(data, path)
    print(f"[cache] saved {path} (v{CACHE_SCHEMA_VERSION})")
    return data


def score_config(
    name: str,
    alpha_weights: tuple[float, float, float],
    peak_alpha: float,
    laplacian_source: str,  # "knn" or "direct"
    cached: dict,
) -> dict:
    """Run boundary_scan with given config and compute ARI@macro/meso.

    SCOPE OF THE `laplacian_source` AXIS (CLAUDE.md Rule 5):
    This function ablates the LAPLACIAN ONLY. The Poincaré `points` are
    reused across both branches, so `laplacian_source="direct"` does NOT
    remove the embedding — it only swaps the Laplacian construction
    (kNN-of-embedding vs combinatorial HSBM graph). The paper's §6.3
    takeaway (a) must therefore frame this row as a Laplacian-source
    ablation, not an embedding ablation. A true embedding ablation
    (replacing `points`) lives in `experiments/ablation_runner_random_points.py`.
    """
    if laplacian_source == "knn":
        lap = cached["laplacian_knn"]
        evs = cached["eigenvalues_knn"]
        evecs = cached["eigenvectors_knn"]
    elif laplacian_source == "direct":
        lap = cached["laplacian_direct"]
        evs = cached["eigenvalues_direct"]
        evecs = cached["eigenvectors_direct"]
    else:
        raise ValueError(f"unknown laplacian_source: {laplacian_source}")

    points = cached["points"]  # NOTE: Poincaré points in both branches (see docstring).
    labels = cached["labels"]

    t0 = time.time()
    res = boundary_scan(
        points=points,
        graph_laplacian=lap,
        focus_idx=0,
        alpha_weights=alpha_weights,
        peak_alpha=peak_alpha,
        eigenvalues=evs,
        eigenvectors=evecs,
    )
    t_scan = time.time() - t0

    # Pick best peak by composite score; if no peaks, use the sigma where score is max.
    if res.peaks:
        best_idx = int(max(res.peaks, key=lambda i: res.scores[i]))
    else:
        best_idx = int(np.argmax(res.scores))
    sigma_best = float(res.sigma_grid[best_idx])
    k_star_best = int(res.effective_dims.get(best_idx, 0))

    # Fixed-K spectral clustering at sigma_best for macro (k=2) and meso (k=8)
    pred_macro = analyze_at_fixed_k(evs, evecs, sigma_best, n_clusters=2)
    pred_meso = analyze_at_fixed_k(evs, evecs, sigma_best, n_clusters=8)

    ari_macro = float(adjusted_rand_index(labels["macro"], pred_macro))
    ari_meso = float(adjusted_rand_index(labels["meso"], pred_meso))
    nmi_macro = float(normalized_mutual_info_score(labels["macro"], pred_macro))
    nmi_meso = float(normalized_mutual_info_score(labels["meso"], pred_meso))

    print(
        f"  [{name:18s}] alpha={alpha_weights} peak_alpha={peak_alpha} src={laplacian_source} "
        f"-> peaks={len(res.peaks)} sigma*={sigma_best:.3f} K*={k_star_best} "
        f"ARI_mac={ari_macro:.3f} ARI_mes={ari_meso:.3f} t={t_scan:.1f}s"
    )

    return {
        "name": name,
        "alpha_weights": list(alpha_weights),
        "peak_alpha": peak_alpha,
        "laplacian_source": laplacian_source,
        "n_peaks": len(res.peaks),
        "sigma_best": sigma_best,
        "k_star_best": k_star_best,
        "ari_macro": ari_macro,
        "ari_meso": ari_meso,
        "nmi_macro": nmi_macro,
        "nmi_meso": nmi_meso,
        "t_scan_s": t_scan,
    }


def build_configs() -> list[dict]:
    """Tier A: 4 alpha_weights + 2 peak_alpha variants + 1 direct-Laplacian = 7 configs.

    The default peak_alpha=2.0 is shared between the four weight configs
    and the direct-Laplacian row; the two explicit peak_alpha variants
    (1.0 "lax" and 3.0 "strict") reuse the default composite weights.
    """
    default_alpha = (1 / 3, 1 / 3, 1 / 3)
    return [
        # Composite weight ablation (all at peak_alpha=2.0, knn source)
        dict(name="full",        alpha_weights=default_alpha, peak_alpha=2.0, src="knn"),
        dict(name="V-only",      alpha_weights=(1.0, 0.0, 0.0), peak_alpha=2.0, src="knn"),
        dict(name="D-only",      alpha_weights=(0.0, 1.0, 0.0), peak_alpha=2.0, src="knn"),
        dict(name="C-only",      alpha_weights=(0.0, 0.0, 1.0), peak_alpha=2.0, src="knn"),
        # Peak-picker sensitivity (at default composite)
        dict(name="lax (alpha=1)",   alpha_weights=default_alpha, peak_alpha=1.0, src="knn"),
        dict(name="strict (alpha=3)", alpha_weights=default_alpha, peak_alpha=3.0, src="knn"),
        # Laplacian source ablation
        dict(name="direct Laplacian", alpha_weights=default_alpha, peak_alpha=2.0, src="direct"),
    ]


def main() -> None:
    """Full preprint-grade sweep: 7 r values x 5 seeds x 7 configs = 245 runs.

    r grid matches Exp 1 Table 1 so ablation columns align with the main
    table. Seeds give bootstrap CIs on ARI. The slow part is Poincaré MDS
    (~8 min per (r, seed)), which caches to per-pair pickles so subsequent
    invocations of this script are near-instant.

    Total MDS budget: up to 7*5 = 35 embeddings ≈ 4.7 hours (laptop).
    Partial runs are safe — results are flushed after each (r, seed, cfg).
    """
    r_values = [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    # Default: 5 seeds matching original plan. Override via SLOD_SEEDS="47,48,49,50,51"
    # to extend sweep on a workstation without touching the JSON's earlier records.
    _raw = os.environ.get("SLOD_SEEDS")
    seeds = [int(x) for x in _raw.split(",") if x.strip()] if _raw else [42, 43, 44, 45, 46]

    configs = build_configs()

    # Resume-aware: load existing JSON and skip (r, seed, name) tuples we
    # already have. This lets us run full sweeps in the background and
    # top them up incrementally.
    done_keys: set[tuple[int, int, str]] = set()
    results: list[dict] = []
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH) as f:
                results = json.load(f)
            for rec in results:
                # Older records may lack "seed"; they used seed=42.
                k = (int(rec.get("r", 200)), int(rec.get("seed", 42)), rec["name"])
                done_keys.add(k)
            print(f"[resume] {len(done_keys)} entries already in {OUT_PATH}")
        except Exception as e:
            print(f"[resume] could not parse existing JSON ({e}); starting fresh")
            results = []

    for r in r_values:
        for seed in seeds:
            cached = None  # lazy — don't reconstruct if every config for (r, seed) is done
            for cfg in configs:
                key = (int(r), int(seed), cfg["name"])
                if key in done_keys:
                    continue
                if cached is None:
                    cached = reconstruct_cached(r=r, seed=seed)
                    print(f"\n=== Ablation on HSBM r={int(r)} seed={seed} (N={cached['n_nodes']}) ===")
                result = score_config(
                    name=cfg["name"],
                    alpha_weights=cfg["alpha_weights"],
                    peak_alpha=cfg["peak_alpha"],
                    laplacian_source=cfg["src"],
                    cached=cached,
                )
                result["r"] = int(r)
                result["seed"] = int(seed)
                results.append(result)
                done_keys.add(key)
                _atomic_json_dump(results, OUT_PATH)

    print(f"\nSaved {len(results)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
