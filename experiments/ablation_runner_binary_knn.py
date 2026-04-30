#!/usr/bin/env python
"""Binary-kNN sidecar ablation: strip Gaussian edge weights, keep topology.

Reads existing Poincaré pickles from the main sweep (no new MDS) and
rebuilds the kNN Laplacian with ``weighted=False`` before re-running the
4 composite-weight configs. Tests whether the Gaussian edge-weight tune
actually contributes or whether the kNN topology alone is sufficient.

Output: results/exp1/ablation_binary_knn.json (merged by aggregate_ablation.py).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import scipy.sparse.linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exp1_hsbm"))

from slod.boundary.scanner import boundary_scan
from slod.boundary.spectral import build_knn_graph, normalized_laplacian
from slod.utils.metrics import adjusted_rand_index
from sklearn.metrics import normalized_mutual_info_score

from analyze_groups import analyze_at_fixed_k

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp1")
OUT_PATH = os.environ.get("SLOD_OUT_PATH") or os.path.join(RESULTS_DIR, "ablation_binary_knn.json")
os.makedirs(RESULTS_DIR, exist_ok=True)


def poincare_cache_path(r: float, seed: int) -> str:
    """Match experiments/ablation_runner.py naming."""
    return os.path.join(RESULTS_DIR, f"ablation_cache_r{int(r)}_s{int(seed)}.pkl")


def _atomic_json_dump(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def build_binary_variant(poincare_cache: dict) -> dict:
    """Rebuild kNN Laplacian from the same Poincaré points but binary weights."""
    points = poincare_cache["points"]
    labels = poincare_cache["labels"]
    n = points.shape[0]
    k_eigs = min(50, n - 1)

    adj_bin = build_knn_graph(points, metric="poincare", weighted=False)
    lap_bin = normalized_laplacian(adj_bin)
    evs, evecs = scipy.sparse.linalg.eigsh(
        lap_bin.astype(np.float64), k=k_eigs, which="SA"
    )
    evs = np.maximum(evs, 0.0)

    return {
        "points": points,
        "labels": labels,
        "laplacian_knn": lap_bin,
        "eigenvalues_knn": evs,
        "eigenvectors_knn": evecs,
    }


def score_binary(name: str, alpha_weights: tuple[float, float, float],
                 peak_alpha: float, binary_data: dict) -> dict:
    lap = binary_data["laplacian_knn"]
    evs = binary_data["eigenvalues_knn"]
    evecs = binary_data["eigenvectors_knn"]
    points = binary_data["points"]
    labels = binary_data["labels"]

    t0 = time.time()
    res = boundary_scan(
        points=points, graph_laplacian=lap, focus_idx=0,
        alpha_weights=alpha_weights, peak_alpha=peak_alpha,
        eigenvalues=evs, eigenvectors=evecs,
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

    return {
        "name": f"{name} (binary kNN)",
        "alpha_weights": list(alpha_weights),
        "peak_alpha": peak_alpha,
        "laplacian_source": "knn",
        "knn_weighting": "binary",
        "embedding_type": "poincare",
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
    default_alpha = (1 / 3, 1 / 3, 1 / 3)
    return [
        dict(name="full",   alpha_weights=default_alpha,       peak_alpha=2.0),
        dict(name="V-only", alpha_weights=(1.0, 0.0, 0.0),     peak_alpha=2.0),
        dict(name="D-only", alpha_weights=(0.0, 1.0, 0.0),     peak_alpha=2.0),
        dict(name="C-only", alpha_weights=(0.0, 0.0, 1.0),     peak_alpha=2.0),
    ]


def _env_seeds(default: list[int]) -> list[int]:
    raw = os.environ.get("SLOD_SEEDS")
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    r_values = [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    seeds = _env_seeds([42, 43, 44, 45, 46])
    configs = build_configs()

    done_keys: set[tuple[int, int, str]] = set()
    results: list[dict] = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        for rec in results:
            k = (int(rec.get("r", 200)), int(rec.get("seed", 42)), rec["name"])
            done_keys.add(k)
        print(f"[resume] {len(done_keys)} entries in {OUT_PATH}")

    for r in r_values:
        for seed in seeds:
            p_cache = poincare_cache_path(r, seed)
            if not os.path.exists(p_cache):
                # Main Poincaré sweep hasn't reached this (r, seed) yet.
                continue
            binary_data = None
            for cfg in configs:
                name_final = f"{cfg['name']} (binary kNN)"
                key = (int(r), int(seed), name_final)
                if key in done_keys:
                    continue
                if binary_data is None:
                    print(f"[build-binary] r={r}, seed={seed}")
                    with open(p_cache, "rb") as f:
                        pc = pickle.load(f)
                    binary_data = build_binary_variant(pc)
                    print(f"  binary kNN Laplacian + eigs built")
                result = score_binary(
                    name=cfg["name"],
                    alpha_weights=cfg["alpha_weights"],
                    peak_alpha=cfg["peak_alpha"],
                    binary_data=binary_data,
                )
                result["r"] = int(r)
                result["seed"] = int(seed)
                results.append(result)
                done_keys.add(key)
                print(f"  [{cfg['name']:15s}]  sigma*={result['sigma_best']:.3f}  ARI_mac={result['ari_macro']:+.3f}  ARI_mes={result['ari_meso']:+.3f}")
                _atomic_json_dump(results, OUT_PATH)

    print(f"\nSaved {len(results)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
