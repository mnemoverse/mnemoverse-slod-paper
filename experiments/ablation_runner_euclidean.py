#!/usr/bin/env python
"""Euclidean-embedding ablation sidecar to experiments/ablation_runner.py.

Runs the same composite-weight + peak-picker ablation axes, but with a
Euclidean MDS embedding in place of the Poincaré one, to isolate the
contribution of hyperbolic geometry. Results are appended to
results/exp1/ablation.json with ``embedding_type="euclidean"`` so a
downstream aggregator can group by embedding type.

Design:
- Separate per-(r, seed) pickle cache for Euclidean MDS (so we don't
  contend with the main Poincaré sweep).
- Re-generates HSBM graphs deterministically via generate_hsbm(seed); the
  graph itself matches the Poincaré path (only MDS differs).
- Scope for preprint: r ∈ {20, 40, 60, 80, 100, 150, 200}, seeds ∈ {42}
  (single seed — Euclidean is a control baseline; multi-seed matters
  mainly for the main Poincaré path in ablation_runner.py).
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
from slod.utils.data import euclidean_mds, generate_hsbm
from slod.utils.metrics import adjusted_rand_index
from sklearn.metrics import normalized_mutual_info_score

from analyze_groups import analyze_at_fixed_k

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp1")
# Per-runner JSON; aggregator merges all four. SLOD_OUT_PATH overrides for
# parallel_runner.py (per-seed paths to eliminate write contention).
OUT_PATH = os.environ.get("SLOD_OUT_PATH") or os.path.join(RESULTS_DIR, "ablation_euclidean.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_SCHEMA_VERSION = 2


def cache_path(r: float, seed: int = 42) -> str:
    return os.path.join(RESULTS_DIR, f"ablation_cache_euclidean_r{int(r)}_s{int(seed)}.pkl")


def _atomic_pickle_dump(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _atomic_json_dump(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def reconstruct_euclidean_cached(r: float, seed: int = 42) -> dict:
    """Build HSBM + Euclidean MDS + Euclidean-kNN Laplacian. Cache per-(r, seed)."""
    path = cache_path(r, seed)
    if os.path.exists(path):
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

    print(f"[reconstruct-euclidean] r={r}, seed={seed}")
    t0 = time.time()
    graph, labels = generate_hsbm(
        n_nodes=1024, n_macro=2, n_meso_per_macro=4, n_micro_per_meso=8,
        r=r, seed=seed,
    )
    if not nx.is_connected(graph):
        lcc = max(nx.connected_components(graph), key=len)
        graph = nx.relabel_nodes(
            graph.subgraph(lcc).copy(),
            {o: i for i, o in enumerate(sorted(lcc))},
        )
        cc_indices = sorted(lcc)
        labels = {k: v[cc_indices] for k, v in labels.items()}
    print(f"  LCC: N={graph.number_of_nodes()}, M={graph.number_of_edges()}, t={time.time()-t0:.1f}s")

    t0 = time.time()
    # Opt-in GPU MDS via SLOD_DEVICE=cuda (default cpu = pristine CPU baseline).
    _device = os.environ.get("SLOD_DEVICE", "cpu")
    points = euclidean_mds(graph, dim=10, max_iter=5000, lr=0.2, seed=seed, device=_device)
    print(f"  Euclidean MDS: t={time.time()-t0:.1f}s")

    t0 = time.time()
    adj_knn = build_knn_graph(points, metric="euclidean")
    lap_knn = normalized_laplacian(adj_knn)
    k_eigs = min(50, graph.number_of_nodes() - 1)
    evs, evecs = scipy.sparse.linalg.eigsh(lap_knn.astype(np.float64), k=k_eigs, which="SA")
    evs = np.maximum(evs, 0.0)
    print(f"  kNN (euclidean) Laplacian + eigs: t={time.time()-t0:.1f}s")

    data = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "r": r,
        "seed": seed,
        "n_nodes": graph.number_of_nodes(),
        "labels": labels,
        "points": points,
        "laplacian_knn": lap_knn,
        "eigenvalues_knn": evs,
        "eigenvectors_knn": evecs,
        "embedding_type": "euclidean",
    }
    _atomic_pickle_dump(data, path)
    print(f"[cache] saved {path} (v{CACHE_SCHEMA_VERSION})")
    return data


def score_euclidean(
    name: str,
    alpha_weights: tuple[float, float, float],
    peak_alpha: float,
    cached: dict,
) -> dict:
    """Score a single Euclidean-ablation config."""
    lap = cached["laplacian_knn"]
    evs = cached["eigenvalues_knn"]
    evecs = cached["eigenvectors_knn"]
    points = cached["points"]
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

    if res.peaks:
        best_idx = int(max(res.peaks, key=lambda i: res.scores[i]))
    else:
        best_idx = int(np.argmax(res.scores))
    sigma_best = float(res.sigma_grid[best_idx])
    k_star_best = int(res.effective_dims.get(best_idx, 0))

    pred_macro = analyze_at_fixed_k(evs, evecs, sigma_best, n_clusters=2)
    pred_meso = analyze_at_fixed_k(evs, evecs, sigma_best, n_clusters=8)
    ari_macro = float(adjusted_rand_index(labels["macro"], pred_macro))
    ari_meso = float(adjusted_rand_index(labels["meso"], pred_meso))
    nmi_macro = float(normalized_mutual_info_score(labels["macro"], pred_macro))
    nmi_meso = float(normalized_mutual_info_score(labels["meso"], pred_meso))

    print(
        f"  [{name:22s}] alpha={alpha_weights} peak={peak_alpha} "
        f"-> peaks={len(res.peaks)} sigma*={sigma_best:.3f} K*={k_star_best} "
        f"ARI_mac={ari_macro:.3f} ARI_mes={ari_meso:.3f} t={t_scan:.1f}s"
    )

    return {
        "name": f"{name} (euclidean)",
        "alpha_weights": list(alpha_weights),
        "peak_alpha": peak_alpha,
        "laplacian_source": "knn",
        "embedding_type": "euclidean",
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
    """Same composite-weight axis as the Poincaré sweep (no direct-Laplacian
    or peak_alpha variants — those are independent of embedding type)."""
    default_alpha = (1 / 3, 1 / 3, 1 / 3)
    return [
        dict(name="full",   alpha_weights=default_alpha,         peak_alpha=2.0),
        dict(name="V-only", alpha_weights=(1.0, 0.0, 0.0),        peak_alpha=2.0),
        dict(name="D-only", alpha_weights=(0.0, 1.0, 0.0),        peak_alpha=2.0),
        dict(name="C-only", alpha_weights=(0.0, 0.0, 1.0),        peak_alpha=2.0),
    ]


def _env_seeds(default: list[int]) -> list[int]:
    raw = os.environ.get("SLOD_SEEDS")
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    r_values = [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    # Single-seed by default (control baseline). Override via SLOD_SEEDS="43,44,45,46"
    # when multi-seed tightening is wanted on the workstation.
    seeds = _env_seeds([42])
    configs = build_configs()

    # Resume from existing JSON.
    done_keys: set[tuple[int, int, str]] = set()
    results: list[dict] = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        for rec in results:
            k = (int(rec.get("r", 200)), int(rec.get("seed", 42)), rec["name"])
            done_keys.add(k)
        print(f"[resume] {len(done_keys)} entries total in {OUT_PATH}")

    for r in r_values:
        for seed in seeds:
            cached = None
            for cfg in configs:
                name_final = f"{cfg['name']} (euclidean)"
                key = (int(r), int(seed), name_final)
                if key in done_keys:
                    continue
                if cached is None:
                    cached = reconstruct_euclidean_cached(r=r, seed=seed)
                    print(f"\n=== Euclidean ablation on HSBM r={int(r)} seed={seed} (N={cached['n_nodes']}) ===")
                result = score_euclidean(
                    name=cfg["name"],
                    alpha_weights=cfg["alpha_weights"],
                    peak_alpha=cfg["peak_alpha"],
                    cached=cached,
                )
                result["r"] = int(r)
                result["seed"] = int(seed)
                results.append(result)
                done_keys.add(key)
                _atomic_json_dump(results, OUT_PATH)

    print(f"\nSaved {len(results)} total entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
