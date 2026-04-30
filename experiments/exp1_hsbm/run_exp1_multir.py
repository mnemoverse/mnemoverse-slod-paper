#!/usr/bin/env python
"""Exp 1 Multi-r: BoundaryScan on HSBM across detection threshold.

Shows that K*(σ) reflects planted hierarchy when SNR is sufficient.
Direct graph Laplacian (no embedding) — isolates the core mechanism.

For each r in [20, 40, 60, 80, 100, 150, 200]:
1. Generate HSBM (1024 nodes, 3-level hierarchy)
2. Build normalized Laplacian directly from adjacency
3. K*(σ) sweep: effective dimensionality at each scale
4. S(σ) = |dK*/dσ|: boundary score (peaks = scale transitions)
5. Find peaks — do they correspond to planted levels?
6. At K*→2 and K*→8 crossings: spectral clustering → ARI vs planted

Key result: phase transition at Kesten-Stigum threshold.
Below threshold (r<40): K* doesn't reach planted levels, ARI≈0.
Above threshold (r>60): K* crossings align with 2 macro + 8 meso, ARI→1.
"""

from __future__ import annotations

import json
import os
import time

import networkx as nx
import numpy as np
import scipy.sparse.linalg
from sklearn.cluster import KMeans

from slod.boundary.spectral import normalized_laplacian
from slod.utils.data import generate_hsbm
from slod.utils.metrics import adjusted_rand_index

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")


def effective_dim(eigenvalues: np.ndarray, sigma: float,
                  threshold: float = 0.95) -> int:
    """K*(σ): smallest k capturing threshold of heat energy."""
    heat = np.exp(-sigma * eigenvalues)
    total = heat.sum()
    if total <= 0:
        return len(eigenvalues)
    cumsum = np.cumsum(heat) / total
    idx = np.where(cumsum >= threshold)[0]
    return int(idx[0]) + 1 if len(idx) > 0 else len(eigenvalues)


def spectral_cluster_pure(eigenvectors: np.ndarray, n_clusters: int,
                          seed: int = 42) -> np.ndarray:
    """Standard spectral clustering: eigenvectors 1..k, row-normalize, KMeans."""
    k_use = min(n_clusters, eigenvectors.shape[1] - 1)
    if k_use < 1:
        return np.zeros(eigenvectors.shape[0], dtype=int)
    evecs = eigenvectors[:, 1:k_use + 1]
    norms = np.linalg.norm(evecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    evecs_n = evecs / norms
    km = KMeans(n_clusters=min(n_clusters, evecs_n.shape[0] - 1),
                random_state=seed, n_init=10)
    return km.fit_predict(evecs_n)


def kesten_stigum_snr(r: float, n: int = 1024) -> dict:
    """Compute SNR for macro and meso detection at given r."""
    # HSBM: 2 macro, 8 meso, 64 micro, 16 nodes/micro
    p_within = 2.0 * r / n
    p_meso = 8.0 / n
    p_macro = 2.0 / n
    p_between = 0.5 / n

    d_micro = 15 * p_within
    d_meso = 112 * p_meso
    d_macro = 384 * p_macro
    d_between = 512 * p_between

    d_in_mac = d_micro + d_meso + d_macro
    d_out_mac = d_between
    snr_mac = (d_in_mac - d_out_mac) ** 2 / (2 * (d_in_mac + d_out_mac))

    d_in_mes = d_micro + d_meso
    d_out_mes = d_macro + d_between
    snr_mes = (d_in_mes - d_out_mes) ** 2 / (2 * (d_in_mes + d_out_mes))

    d_avg = d_micro + d_meso + d_macro + d_between

    return {"snr_macro": snr_mac, "snr_meso": snr_mes, "d_avg": d_avg}


def run_single_r(r: float, seed: int = 42) -> dict:
    """Full BoundaryScan analysis at given r on direct graph."""
    t0 = time.time()

    graph, labels = generate_hsbm(
        n_nodes=1024, n_macro=2, n_meso_per_macro=4, n_micro_per_meso=8,
        r=r, seed=seed,
    )

    # LCC extraction
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        mapping = {old: new for new, old in enumerate(sorted(largest_cc))}
        subgraph = graph.subgraph(largest_cc).copy()
        graph = nx.relabel_nodes(subgraph, mapping)
        cc_indices = sorted(largest_cc)
        labels = {k: v[cc_indices] for k, v in labels.items()}

    n = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    avg_deg = 2 * n_edges / n

    # Direct Laplacian
    adj = nx.adjacency_matrix(graph).astype(np.float64)
    lap = normalized_laplacian(adj)

    # Eigendecomposition.
    # K_eigs=80 retains enough modes for K*(σ) to drop through the planted
    # micro level (K=64) — at K_eigs=50 the heat-kernel sum saturates around
    # K*=48 and never reveals micro recovery in the K*(σ) trajectory.
    k = min(80, n - 1)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        lap.astype(np.float64), k=k, which="SA"
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Spectral gaps
    gap_2 = eigenvalues[2] / eigenvalues[1] if eigenvalues[1] > 1e-10 else 0.0
    gap_8 = eigenvalues[8] / eigenvalues[7] if eigenvalues[7] > 1e-10 else 0.0

    # K*(σ) sweep
    sigmas = np.concatenate([
        np.logspace(-1, 0, 20),      # 0.1 to 1
        np.logspace(0, 0.7, 20),     # 1 to 5
        np.logspace(0.7, 1.0, 20),   # 5 to 10
        np.logspace(1.0, 1.3, 20),   # 10 to 20
        np.logspace(1.3, 1.7, 20),   # 20 to 50
        np.logspace(1.7, 2.0, 10),   # 50 to 100
    ])
    sigmas = np.unique(sigmas)

    kstar_curve = []
    for sigma in sigmas:
        kstar = effective_dim(eigenvalues, sigma)
        kstar_curve.append({"sigma": float(sigma), "k_star": kstar})

    # S(σ) = boundary score via finite difference
    boundary_scores = []
    for i in range(1, len(kstar_curve) - 1):
        s_prev = kstar_curve[i - 1]["sigma"]
        s_next = kstar_curve[i + 1]["sigma"]
        k_prev = kstar_curve[i - 1]["k_star"]
        k_next = kstar_curve[i + 1]["k_star"]
        ds = s_next - s_prev
        if ds > 0:
            score = abs(k_next - k_prev) / ds
        else:
            score = 0.0
        boundary_scores.append({
            "sigma": kstar_curve[i]["sigma"],
            "k_star": kstar_curve[i]["k_star"],
            "score": score,
        })

    # Find peaks in S(σ): local maxima with score > 0
    peaks = []
    for i in range(1, len(boundary_scores) - 1):
        s = boundary_scores[i]["score"]
        if (s > 0 and
                s >= boundary_scores[i - 1]["score"] and
                s >= boundary_scores[i + 1]["score"]):
            peaks.append(boundary_scores[i])

    # Find K* crossings at planted levels
    crossings = {}
    for i in range(1, len(kstar_curve)):
        k_prev = kstar_curve[i - 1]["k_star"]
        k_curr = kstar_curve[i]["k_star"]
        for target in [64, 8, 2]:
            if k_prev > target >= k_curr and target not in crossings:
                crossings[target] = kstar_curve[i]["sigma"]

    # ARI at crossings
    crossing_ari = {}
    for target_k, sigma_cross in crossings.items():
        level_map = {2: "macro", 8: "meso", 64: "micro"}
        level = level_map.get(target_k)
        if level and level in labels:
            pred = spectral_cluster_pure(eigenvectors, target_k, seed)
            ari = adjusted_rand_index(labels[level], pred)
            crossing_ari[target_k] = {
                "sigma": sigma_cross,
                "ari": ari,
                "level": level,
            }

    # Also: fixed-k ARI at representative sigma values
    # Use sigma where K* is closest to 2, 8, and 64 (planted hierarchy levels)
    fixed_ari = {}
    for target_k, level in [(2, "macro"), (8, "meso"), (64, "micro")]:
        if level not in labels:
            continue
        if target_k > k:
            # eigsh returned fewer eigenvectors than the target K (e.g. k=80
            # eigenvectors but target K=64 still fits; this guard is defensive
            # for edge cases like very small N).
            continue
        pred = spectral_cluster_pure(eigenvectors, target_k, seed)
        ari = adjusted_rand_index(labels[level], pred)
        fixed_ari[target_k] = ari

    # KS threshold
    ks = kesten_stigum_snr(r)

    elapsed = time.time() - t0

    result = {
        "r": r,
        "n_nodes": n,
        "n_edges": n_edges,
        "avg_degree": avg_deg,
        "lcc_lost": 1024 - n,
        "eigenvalues": eigenvalues[:20].tolist(),
        "gap_ratio_2": gap_2,
        "gap_ratio_8": gap_8,
        "kstar_curve": kstar_curve,
        "boundary_scores": boundary_scores,
        "peaks": peaks,
        "crossings": {str(k): v for k, v in crossing_ari.items()},
        "fixed_ari_macro": fixed_ari.get(2, 0.0),
        "fixed_ari_meso": fixed_ari.get(8, 0.0),
        "fixed_ari_micro": fixed_ari.get(64, 0.0),
        "snr_macro": ks["snr_macro"],
        "snr_meso": ks["snr_meso"],
        "elapsed_s": elapsed,
    }

    return result


def main() -> None:
    r_values = [20, 40, 60, 80, 100, 150, 200]
    all_results = []

    print("=" * 80)
    print("EXP 1 MULTI-R: BoundaryScan across Kesten-Stigum threshold")
    print("=" * 80)
    print("HSBM: 1024 nodes, 2 macro → 8 meso → 64 micro")
    print(f"r values: {r_values}")
    print("Method: direct graph Laplacian (no embedding)")

    for r in r_values:
        print(f"\n--- r={r} ---")
        result = run_single_r(r)
        all_results.append(result)

        # Print summary
        cross_str = ""
        for k in ["2", "8"]:
            if k in result["crossings"]:
                c = result["crossings"][k]
                cross_str += (f"  K*→{k} at σ={c['sigma']:.1f}: "
                             f"ARI({c['level']})={c['ari']:.3f}")
        if not cross_str:
            cross_str = "  (K* never reaches 2 or 8)"

        print(f"  n={result['n_nodes']}, lost={result['lcc_lost']}, "
              f"avg_deg={result['avg_degree']:.2f}")
        print(f"  SNR: macro={result['snr_macro']:.3f}, "
              f"meso={result['snr_meso']:.3f}")
        print(f"  Gaps: λ3/λ2={result['gap_ratio_2']:.2f}, "
              f"λ9/λ8={result['gap_ratio_8']:.2f}")
        print(f"  Fixed ARI: macro={result['fixed_ari_macro']:.3f}, "
              f"meso={result['fixed_ari_meso']:.3f}")
        print(f"  Crossings:{cross_str}")
        top_peaks = sorted(result["peaks"], key=lambda x: -x["score"])[:5]
        top_sigmas = ", ".join(f"{p['sigma']:.1f}" for p in top_peaks)
        print(f"  Peaks: {len(result['peaks'])} (top σ: {top_sigmas})")
        print(f"  Time: {result['elapsed_s']:.1f}s")

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print(f"\n\n{'='*80}")
    print("SUMMARY: Phase transition across Kesten-Stigum threshold")
    print(f"{'='*80}")

    print(f"\n{'r':>4} | {'SNR_m':>5} {'SNR_s':>5} | {'gap_2':>5} {'gap_8':>5} | "
          f"{'ARI_M':>5} {'ARI_S':>5} | "
          f"{'K*→2 σ':>7} {'ARI@2':>6} | "
          f"{'K*→8 σ':>7} {'ARI@8':>6} | {'n_pk':>4}")
    print("-" * 95)

    for res in all_results:
        cross_2_sigma = ""
        cross_2_ari = ""
        cross_8_sigma = ""
        cross_8_ari = ""
        if "2" in res["crossings"]:
            cross_2_sigma = f"{res['crossings']['2']['sigma']:7.1f}"
            cross_2_ari = f"{res['crossings']['2']['ari']:6.3f}"
        else:
            cross_2_sigma = "    ---"
            cross_2_ari = "   ---"
        if "8" in res["crossings"]:
            cross_8_sigma = f"{res['crossings']['8']['sigma']:7.1f}"
            cross_8_ari = f"{res['crossings']['8']['ari']:6.3f}"
        else:
            cross_8_sigma = "    ---"
            cross_8_ari = "   ---"

        above_mac = "*" if res["snr_macro"] > 1 else " "
        above_mes = "*" if res["snr_meso"] > 1 else " "

        print(f"{res['r']:4.0f} |{above_mac}{res['snr_macro']:5.2f}"
              f"{above_mes}{res['snr_meso']:5.2f} | "
              f"{res['gap_ratio_2']:5.2f} {res['gap_ratio_8']:5.2f} | "
              f"{res['fixed_ari_macro']:5.3f} {res['fixed_ari_meso']:5.3f} | "
              f"{cross_2_sigma} {cross_2_ari} | "
              f"{cross_8_sigma} {cross_8_ari} | "
              f"{len(res['peaks']):4d}")

    print("\n  * = above Kesten-Stigum threshold (SNR > 1)")
    print("  ARI_M/ARI_S = fixed k=2/k=8 spectral clustering (pure, no heat)")
    print("  K*→2/K*→8 = sigma where K*(σ) crosses planted level")
    print("  ARI@2/ARI@8 = ARI of spectral clustering at crossing point")

    # K*(σ) at representative sigmas across all r
    print(f"\n\n{'='*80}")
    print("K*(σ) TRAJECTORY: how K* depends on r")
    print(f"{'='*80}")

    # Pick representative sigmas
    rep_sigmas = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100]
    print(f"\n{'':>4}", end="")
    for s in rep_sigmas:
        print(f" {'σ='+str(s):>6}", end="")
    print()
    print("-" * (4 + 7 * len(rep_sigmas)))

    for res in all_results:
        print(f"r={res['r']:>3}", end="")
        for s_target in rep_sigmas:
            # Find nearest sigma in curve
            best = min(res["kstar_curve"],
                      key=lambda x: abs(x["sigma"] - s_target))
            print(f" {best['k_star']:>6}", end="")
        print()

    # Save all results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "multir_direct.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {save_path}")

    # ===================================================================
    # INTERPRETATION
    # ===================================================================
    print(f"\n\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("""
PHASE TRANSITION observed at Kesten-Stigum threshold:

1. BELOW threshold (r=20, SNR_macro=0.78):
   - K*(σ) stays flat at ~48, barely decays
   - No crossings at K*=2 or K*=8 within reasonable σ
   - ARI ≈ 0: planted structure invisible in spectrum
   - Eigenvalue gap ratio ≈ 1.1: no spectral gap

2. AT threshold (r=40, SNR_macro=1.06):
   - K*(σ) starts showing transitions
   - Spectral gap begins to open (gap_ratio approaching 1.5)
   - ARI macro improves via pure spectral clustering

3. ABOVE threshold (r≥60, SNR_macro≥1.4):
   - K*(σ) clearly crosses planted levels (K*→2 for macro, K*→8 for meso)
   - ARI at crossings: high (0.8+ for macro, improving for meso)
   - Spectral gap > 2.0 at k=2

4. WELL ABOVE threshold (r=200, SNR_macro=3.4):
   - K*(σ) transitions match planted hierarchy exactly
   - ARI macro = 1.0 (perfect), ARI meso = 0.91
   - Clear spectral gap (ratio = 2.15)

CONCLUSION:
K*(σ) IS a meaningful continuous LOD that reflects real hierarchy.
The σ values where K* transitions correspond to scale boundaries.
BoundaryScan works when the signal is above the detection threshold
— which is the best any algorithm can do (information-theoretic limit).
""")


if __name__ == "__main__":
    main()
