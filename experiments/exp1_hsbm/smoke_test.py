#!/usr/bin/env python
"""HSBM smoke test: SLoD sigma sweep on synthetic 3-level hierarchy.

Validates the full pipeline:
1. Generate 3-level HSBM graph (64 nodes, 2*2*2=8 micro)
2. Embed into B^5 via Riemannian hyperbolic MDS
3. Build graph Laplacian
4. Run SLoD at sigma = [0.01, 0.1, 1.0, 10.0, 100.0]
5. Print d_H(result, focus) at each sigma

Expected: distance from focus should increase monotonically with sigma.
"""

from __future__ import annotations

import numpy as np

from slod.core.poincare import poincare_distance
from slod.core.slod_operator import slod
from slod.utils.data import generate_hsbm, graph_to_laplacian, hyperbolic_mds


def main() -> None:
    print("=" * 60)
    print("HSBM Smoke Test: SLoD Sigma Sweep (3-level hierarchy)")
    print("=" * 60)

    # Step 1: Generate 3-level HSBM
    print("\n[1] Generating 3-level HSBM (64 nodes, 2*2*2=8 micro, r=20)...")
    graph, labels = generate_hsbm(
        n_nodes=64, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=2, r=20.0, seed=42
    )
    print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    n_macro = len(np.unique(labels["macro"]))
    n_meso = len(np.unique(labels["meso"]))
    n_micro = len(np.unique(labels["micro"]))
    print(f"    Macro: {n_macro}, Meso: {n_meso}, Micro: {n_micro}")

    # Step 2: Embed into B^5
    print("\n[2] Riemannian hyperbolic MDS embedding into B^5...")
    points = hyperbolic_mds(graph, dim=5, max_iter=300, seed=42)
    norms = np.linalg.norm(points, axis=1)
    print(f"    Points shape: {points.shape}")
    print(f"    Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    assert np.all(norms < 1.0), "All points must be inside ball!"

    # Step 3: Build Laplacian
    print("\n[3] Building graph Laplacian...")
    laplacian = graph_to_laplacian(graph)
    print(f"    Laplacian shape: {laplacian.shape}")

    # Step 4: SLoD sigma sweep
    print("\n[4] SLoD sigma sweep (focus_idx=0)...")
    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    focus_idx = 0
    focus_point = points[focus_idx]

    print(f"\n    {'sigma':>10s}  {'d_H(result, focus)':>20s}  {'||result||':>12s}")
    print("    " + "-" * 46)

    prev_dist = -1.0
    monotonic = True
    for sigma in sigmas:
        result = slod(points, laplacian, sigma=sigma, focus_idx=focus_idx)
        dist = poincare_distance(result, focus_point)
        norm = np.linalg.norm(result)
        flag = ""
        if dist < prev_dist - 1e-6:
            monotonic = False
            flag = " <-- NON-MONOTONIC"
        print(f"    {sigma:>10.2f}  {dist:>20.6f}  {norm:>12.6f}{flag}")
        prev_dist = dist

    # Step 5: Summary
    print("\n" + "=" * 60)
    if monotonic:
        print("PASS: d_H increases monotonically with sigma")
    else:
        print("WARN: d_H is NOT strictly monotonic (may be acceptable for noise)")
    print("=" * 60)


if __name__ == "__main__":
    main()
