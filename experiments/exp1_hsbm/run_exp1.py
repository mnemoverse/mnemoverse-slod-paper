#!/usr/bin/env python
"""Exp 1: BoundaryScan on 1024-node 3-level HSBM.

Full evaluation protocol (Paper Section 5.1):
1. Generate 3-level HSBM (1024 nodes, r={2,5,10,20})
2. Embed into B^10 via Riemannian hyperbolic MDS
3. Validate embedding quality (Spearman > 0.7)
4. Build kNN graph + normalized Laplacian
5. Run BoundaryScan from multiple focus nodes
6. Report: peaks detected, effective dimensionality, indicator profiles
7. Save results to results/exp1/
"""

from __future__ import annotations

import json
import os
import time

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from slod.boundary.scanner import boundary_scan
from slod.boundary.spectral import build_knn_graph, normalized_laplacian
from slod.core.poincare import poincare_distance
from slod.utils.data import generate_hsbm, hyperbolic_mds


def _spearman_quality(graph: nx.Graph, points: np.ndarray) -> float:
    """Compute Spearman correlation between graph and embedding distances."""
    sp_dict = dict(nx.all_pairs_shortest_path_length(graph))
    g_dists, e_dists = [], []
    nodes = sorted(graph.nodes())
    for idx_i, ni in enumerate(nodes):
        for idx_j, nj in enumerate(nodes):
            if idx_i < idx_j:
                g_dists.append(sp_dict[ni][nj])
                e_dists.append(poincare_distance(points[idx_i], points[idx_j]))
    rho, _ = spearmanr(g_dists, e_dists)
    return float(rho)


def run_single(r: float, seed: int = 42) -> dict:
    """Run Exp 1 for a single r value."""
    print(f"\n{'='*60}")
    print(f"Exp 1: r={r}")
    print(f"{'='*60}")

    # Step 1: Generate 3-level HSBM
    print(f"\n[1] Generating 3-level HSBM (1024 nodes, r={r})...")
    t0 = time.time()
    graph, labels = generate_hsbm(
        n_nodes=1024, n_macro=2, n_meso_per_macro=4, n_micro_per_meso=8,
        r=r, seed=seed,
    )
    gen_time = time.time() - t0
    print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"    Macro: {len(np.unique(labels['macro']))}, "
          f"Meso: {len(np.unique(labels['meso']))}, "
          f"Micro: {len(np.unique(labels['micro']))}")
    print(f"    Generated in {gen_time:.1f}s")

    # Handle disconnected graph (possible at low r)
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        mapping = {old: new for new, old in enumerate(sorted(largest_cc))}
        subgraph = graph.subgraph(largest_cc).copy()
        graph = nx.relabel_nodes(subgraph, mapping)
        cc_indices = sorted(largest_cc)
        labels = {k: v[cc_indices] for k, v in labels.items()}
        print(f"    Graph disconnected, using largest component: {graph.number_of_nodes()} nodes")

    n = graph.number_of_nodes()

    # Step 2: Riemannian hyperbolic MDS
    print("\n[2] Riemannian hyperbolic MDS into B^10...")
    t0 = time.time()
    points = hyperbolic_mds(graph, dim=10, max_iter=5000, lr=0.2, seed=seed)
    mds_time = time.time() - t0
    norms = np.linalg.norm(points, axis=1)
    print(f"    Points shape: {points.shape}")
    print(f"    Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"    Embedded in {mds_time:.1f}s")

    # Step 3: Embedding quality check
    print("\n[3] Embedding quality check...")
    rho = _spearman_quality(graph, points)
    print(f"    Spearman rho = {rho:.3f} (gate: > 0.7)")
    if rho < 0.7:
        print("    EMBEDDING QUALITY BELOW THRESHOLD — results may be unreliable")

    # Step 4: Build kNN graph + Laplacian
    print("\n[4] Building kNN graph + normalized Laplacian...")
    t0 = time.time()
    adj = build_knn_graph(points)
    laplacian = normalized_laplacian(adj)
    graph_time = time.time() - t0
    print(f"    kNN graph: {adj.nnz} edges, built in {graph_time:.1f}s")

    # Step 5: Run BoundaryScan from multiple focus nodes
    print("\n[5] Running BoundaryScan from 5 focus nodes...")
    sigma_grid = np.logspace(-2, 2, 100)
    rng = np.random.RandomState(seed)
    focus_set: set[int] = set()
    for macro_id in np.unique(labels["macro"]):
        candidates = np.where(labels["macro"] == macro_id)[0]
        node = int(rng.choice(candidates))
        focus_set.add(node)
    while len(focus_set) < 5:
        focus_set.add(int(rng.randint(0, n)))
    focus_nodes = sorted(focus_set)

    all_results = []
    for fi, focus_idx in enumerate(focus_nodes):
        t0 = time.time()
        result = boundary_scan(
            points, laplacian,
            focus_idx=focus_idx,
            sigma_grid=sigma_grid,
            k_eigs=min(50, n - 1),
            churn_k=10,
        )
        elapsed = time.time() - t0
        print(f"    Focus {fi} (node {focus_idx}): "
              f"{len(result.peaks)} peaks, "
              f"score range [{result.scores.min():.3f}, {result.scores.max():.3f}], "
              f"{elapsed:.1f}s")
        for peak_idx in result.peaks:
            sigma_peak = result.sigma_grid[peak_idx]
            score = result.scores[peak_idx]
            k_star = result.effective_dims.get(peak_idx, "?")
            print(f"        Peak at sigma={sigma_peak:.4f}, S={score:.3f}, K*={k_star}")
        all_results.append(result)

    # Step 6: Aggregate results
    print(f"\n[6] Summary for r={r}:")
    peak_counts = [len(res.peaks) for res in all_results]
    avg_peaks = np.mean(peak_counts)
    print(f"    Average peaks per focus node: {avg_peaks:.1f}")
    print(f"    Peak counts: {peak_counts}")

    all_peak_sigmas = []
    for res in all_results:
        for pi in res.peaks:
            all_peak_sigmas.append(float(res.sigma_grid[pi]))
    print(f"    All peak sigmas: {[f'{s:.3f}' for s in sorted(all_peak_sigmas)]}")

    # Build per-focus detailed results (raw, unfiltered)
    focus_results = []
    for fi, res in enumerate(all_results):
        peaks_detail = []
        for pi in res.peaks:
            peaks_detail.append({
                "sigma": float(res.sigma_grid[pi]),
                "score": float(res.scores[pi]),
                "k_star": int(res.effective_dims.get(pi, -1)),
                "grid_index": int(pi),
            })
        focus_results.append({
            "focus_node": focus_nodes[fi],
            "n_peaks": len(res.peaks),
            "peaks": peaks_detail,
            "score_profile": res.scores.tolist(),
            "velocity_profile": res.velocity.tolist(),
            "divergence_profile": res.divergence.tolist(),
            "churn_profile": res.churn.tolist(),
            "effective_dims": {str(k): int(v) for k, v in res.effective_dims.items()},
        })

    return {
        "r": r,
        "n_nodes": n,
        "n_edges": graph.number_of_edges(),
        "spearman_rho": rho,
        "mds_time_s": mds_time,
        "graph_build_time_s": graph_time,
        "n_focus_nodes": len(focus_nodes),
        "sigma_grid": sigma_grid.tolist(),
        "peak_counts": peak_counts,
        "avg_peaks": float(avg_peaks),
        "all_peak_sigmas": sorted(all_peak_sigmas),
        "spectral_candidates": [float(s) for s in all_results[0].spectral_candidates[:10]],
        "focus_results": focus_results,
    }


def main() -> None:
    print("=" * 60)
    print("Exp 1: BoundaryScan on 3-level HSBM (1024 nodes)")
    print("Paper Section 5.1: scale boundary recovery")
    print("=" * 60)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")
    os.makedirs(results_dir, exist_ok=True)

    r_values = [20.0, 10.0, 5.0, 2.0]
    all_results = []

    for r in r_values:
        result = run_single(r)
        all_results.append(result)

        result_file = os.path.join(results_dir, f"full_r{int(r)}.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n    Saved to {result_file}")

    # Save summary
    summary = {
        "experiment": "Exp 1: HSBM BoundaryScan",
        "n_nodes": 1024,
        "n_levels": 3,
        "hierarchy": "2 macro -> 8 meso -> 64 micro",
        "embedding_dim": 10,
        "r_values": r_values,
        "results": all_results,
    }
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Final report
    print(f"\n\n{'='*60}")
    print("EXPERIMENT 1 SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'r':>6s}  {'rho':>6s}  {'avg peaks':>10s}  {'peak sigmas':>30s}")
    print("-" * 60)
    for r_data in all_results:
        sigmas_str = ", ".join(f"{s:.2f}" for s in r_data["all_peak_sigmas"][:5])
        print(f"{r_data['r']:>6.0f}  {r_data['spearman_rho']:>6.3f}  "
              f"{r_data['avg_peaks']:>10.1f}  {sigmas_str:>30s}")
    print(f"\nResults saved to {results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
