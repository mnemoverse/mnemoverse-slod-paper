#!/usr/bin/env python
"""Random-points control sidecar: ablates the Poincaré embedding itself.

Uses the same Poincaré-derived kNN Laplacian as the main sweep but replaces
`points` with i.i.d. Gaussian noise in R^10 before running boundary_scan.
This is a true "no embedding" control — whatever geometric information the
Poincaré embedding contributes should show up as a gap between this baseline
and the main Poincaré sweep. Unlike the "direct Laplacian" ablation, which
only swaps the Laplacian construction (and still uses Poincaré points for
Fréchet mean / V / C), this actually strips the embedding.

Reuses pickles from the main sweep (no new MDS). Fast — few seconds per
(r, seed). Output: results/exp1/ablation_random_points.json.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exp1_hsbm"))

from slod.boundary.scanner import boundary_scan
from slod.utils.metrics import adjusted_rand_index
from sklearn.metrics import normalized_mutual_info_score

from analyze_groups import analyze_at_fixed_k

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "exp1")
OUT_PATH = os.environ.get("SLOD_OUT_PATH") or os.path.join(RESULTS_DIR, "ablation_random_points.json")


def poincare_cache_path(r: float, seed: int) -> str:
    return os.path.join(RESULTS_DIR, f"ablation_cache_r{int(r)}_s{int(seed)}.pkl")


def _atomic_json_dump(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def score_random(name: str, alpha_weights: tuple[float, float, float],
                 peak_alpha: float, cached: dict, rng_seed: int) -> dict:
    n, d = cached["points"].shape
    rng = np.random.default_rng(rng_seed)
    noise_scale = _noise_scale()
    random_points = rng.standard_normal(size=(n, d)) * noise_scale

    lap = cached["laplacian_knn"]
    evs = cached["eigenvalues_knn"]
    evecs = cached["eigenvectors_knn"]
    labels = cached["labels"]

    t0 = time.time()
    res = boundary_scan(
        points=random_points, graph_laplacian=lap, focus_idx=0,
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
        "name": f"{name} (random points)",
        "alpha_weights": list(alpha_weights),
        "peak_alpha": peak_alpha,
        "laplacian_source": "knn",
        "embedding_type": "random",
        "points_rng_seed": rng_seed,
        "n_peaks": len(res.peaks),
        "sigma_best": sigma_best,
        "k_star_best": k_star_best,
        "ari_macro": ari_macro,
        "ari_meso": ari_meso,
        "nmi_macro": nmi_macro,
        "nmi_meso": nmi_meso,
        "t_scan_s": t_scan,
    }


def _env_seeds(default: list[int]) -> list[int]:
    raw = os.environ.get("SLOD_SEEDS")
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def _noise_scale() -> float:
    """Override random-points noise σ via SLOD_NOISE_SCALE (default 0.1)."""
    return float(os.environ.get("SLOD_NOISE_SCALE", "0.1"))


def main() -> None:
    r_values = [20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
    seeds = _env_seeds([42, 43, 44, 45, 46])
    configs = [
        dict(name="full",   alpha_weights=(1 / 3, 1 / 3, 1 / 3), peak_alpha=2.0),
        dict(name="D-only", alpha_weights=(0.0, 1.0, 0.0),        peak_alpha=2.0),
    ]

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
            pcache = poincare_cache_path(r, seed)
            if not os.path.exists(pcache):
                continue
            cached = None
            for cfg in configs:
                name_final = f"{cfg['name']} (random points)"
                key = (int(r), int(seed), name_final)
                if key in done_keys:
                    continue
                if cached is None:
                    with open(pcache, "rb") as f:
                        cached = pickle.load(f)
                # Use same "points rng seed" as graph seed — reproducible but
                # independent from the Poincaré embedding seed.
                result = score_random(
                    name=cfg["name"],
                    alpha_weights=cfg["alpha_weights"],
                    peak_alpha=cfg["peak_alpha"],
                    cached=cached,
                    rng_seed=seed,
                )
                result["r"] = int(r)
                result["seed"] = int(seed)
                results.append(result)
                done_keys.add(key)
                print(f"  r={int(r):3d} seed={seed} [{cfg['name']:6s}] sigma*={result['sigma_best']:6.2f} "
                      f"macro={result['ari_macro']:+.3f} meso={result['ari_meso']:+.3f}")
                _atomic_json_dump(results, OUT_PATH)

    print(f"\nSaved {len(results)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
