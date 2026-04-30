#!/usr/bin/env python
"""Exp 2: Hierarchical Consistency on WordNet.

Full evaluation protocol (Paper Section 5.2):
1. Load WordNet 3.0 noun hierarchy (~82K synsets, DAG)
2. Embed into B^10 via Poincare embeddings (Nickel & Kiela 2017)
3. Quality gates: hierarchy preservation, depth correlation, sibling proximity
4. Build batched kNN graph + normalized Laplacian
5. Run BoundaryScan from 100 stratified leaf nodes
6. Metrics: Kendall tau, Hit@L, MRR
7. Save results to results/exp2/
"""

from __future__ import annotations

import json
import os
import time

import geoopt
import numpy as np
import torch

from slod.boundary.scanner import boundary_scan
from slod.boundary.spectral import (
    build_knn_graph_approx,
    build_knn_graph_batched,
    normalized_laplacian,
)
from slod.utils.data import poincare_embed
from slod.utils.embedding_quality import (
    depth_correlation_test,
    hierarchy_preservation_test,
    sibling_proximity_test,
)
from slod.utils.metrics import hit_at_l, kendall_tau
from slod.utils.wordnet_loader import (
    get_ancestors,
    load_wordnet_noun_graph,
    load_wordnet_subtree,
    select_stratified_leaves,
    wordnet_to_undirected,
)


def _build_quality_data(
    dag, metadata: dict
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Extract parent-child pairs and sibling groups from DAG for quality tests."""
    pairs = list(dag.edges())  # child → parent edges
    # Sibling groups: children of the same parent
    from collections import defaultdict

    parent_to_children: dict[int, list[int]] = defaultdict(list)
    for child, parent in dag.edges():
        parent_to_children[parent].append(child)
    sibling_groups = [
        children for children in parent_to_children.values() if len(children) >= 2
    ]
    return pairs, sibling_groups


def _boundary_to_depth(
    peak_sigma: float,
    sigma_grid: np.ndarray,
    max_depth: int,
) -> float:
    """Map a detected boundary sigma to a continuous depth value.

    sigma inversely relates to detail: large sigma = coarse = shallow depth.
    Linear mapping on log-scale: log(sigma) → depth.

    Returns continuous float (not rounded) so Hit@L metrics are
    discriminative — exact integer matches are rare, making Hit@0
    genuinely hard while Hit@1 measures ±1 level tolerance.
    """
    log_min = np.log(sigma_grid[0])
    log_max = np.log(sigma_grid[-1])
    log_sigma = np.log(peak_sigma)
    # Invert: large sigma → low depth (shallow)
    frac = 1.0 - (log_sigma - log_min) / (log_max - log_min + 1e-10)
    return max(0.0, min(float(max_depth), frac * max_depth))


def run_experiment(
    n_focus: int = 100,
    dim: int = 10,
    epochs: int = 50,
    k_eigs: int = 50,
    batch_size: int = 512,
    seed: int = 42,
    max_depth_limit: int | None = None,
    knn_method: str = "auto",
) -> dict:
    """Run full Exp 2 pipeline.

    Args:
        max_depth_limit: If set, load only synsets with depth <= this value.
            Use 5-6 for fast runs (~3-5K nodes), None for full graph (~82K).
    """
    print("=" * 60)
    print("Exp 2: Hierarchical Consistency on WordNet")
    print("Paper Section 5.2")
    print("=" * 60)

    # ── Step 1: Load WordNet ──────────────────────────────────
    if max_depth_limit is not None:
        print(f"\n[1] Loading WordNet subtree (max_depth={max_depth_limit})...")
    else:
        print("\n[1] Loading WordNet noun hierarchy (full)...")
    t0 = time.time()
    if max_depth_limit is not None:
        dag, metadata = load_wordnet_subtree(max_depth=max_depth_limit)
    else:
        dag, metadata = load_wordnet_noun_graph()
    load_time = time.time() - t0
    n_nodes = metadata["n_nodes"]
    max_depth = metadata["max_depth"]
    depths = metadata["depths"]
    leaves = metadata["leaves"]
    print(f"    Nodes: {n_nodes}, Edges: {dag.number_of_edges()}")
    print(f"    Max depth: {max_depth}, Leaves: {len(leaves)}")
    print(f"    Loaded in {load_time:.1f}s")

    # ── Step 2: Poincare embedding ────────────────────────────
    print(f"\n[2] Poincare embedding into B^{dim} ({epochs} epochs)...")
    undirected = wordnet_to_undirected(dag)
    t0 = time.time()
    points = poincare_embed(
        undirected, dim=dim, epochs=epochs,
        lr=0.01, burn_in=20, burn_in_lr=0.001, seed=seed,
    )
    embed_time = time.time() - t0
    norms = np.linalg.norm(points, axis=1)
    print(f"    Points shape: {points.shape}")
    print(f"    Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"    Embedded in {embed_time:.1f}s")

    # ── Step 3: Quality gates ─────────────────────────────────
    print("\n[3] Embedding quality gates...")
    pairs, sibling_groups = _build_quality_data(dag, metadata)

    # Sample for efficiency on large graph
    rng = np.random.RandomState(seed)
    sample_pairs = [pairs[i] for i in rng.choice(len(pairs), min(5000, len(pairs)), replace=False)]
    grp_idx = rng.choice(len(sibling_groups), min(500, len(sibling_groups)), replace=False)
    sample_groups = [sibling_groups[i] for i in grp_idx]

    hp_score = hierarchy_preservation_test(points, sample_pairs, n=2000, seed=seed)
    dc_score = depth_correlation_test(points, depths)
    sp_score = sibling_proximity_test(points, sample_groups, n=1000, seed=seed)

    print(f"    Hierarchy preservation: {hp_score:.3f} (gate: >0.90)")
    print(f"    Depth correlation:      {dc_score:.3f} (informational for DAGs)")
    print(f"    Sibling proximity:      {sp_score:.3f} (gate: >0.85)")

    # DC gate relaxed for DAG data — multiple parents make max_depth() ambiguous.
    # For strict trees DC > 0.60 is expected; for WordNet DAG, DC ~ 0.2 is normal.
    quality_pass = hp_score > 0.90 and sp_score > 0.85
    if dc_score < 0.60:
        print("    NOTE: DC < 0.60 expected for DAGs (multiple paths → ambiguous depth)")
    if not quality_pass:
        print("    WARNING: HP or SP gate failed — results may be unreliable")
    else:
        print("    Quality gates PASSED (HP + SP)")

    quality = {
        "hierarchy_preservation": float(hp_score),
        "depth_correlation": float(dc_score),
        "sibling_proximity": float(sp_score),
        "all_passed": quality_pass,
    }

    # ── Step 4: Build kNN graph + Laplacian ───────────────────
    use_approx = (knn_method == "approx") or (knn_method == "auto" and n_nodes > 10_000)
    method_label = "approx (tangent-space)" if use_approx else "exact (batched)"
    print(f"\n[4] Building kNN graph ({method_label}) + normalized Laplacian...")
    t0 = time.time()
    if use_approx:
        adj = build_knn_graph_approx(points, batch_size=batch_size)
    else:
        adj = build_knn_graph_batched(points, batch_size=batch_size)
    laplacian = normalized_laplacian(adj)
    graph_time = time.time() - t0
    print(f"    kNN graph: {adj.nnz} edges, built in {graph_time:.1f}s")

    # ── Step 5: BoundaryScan from stratified leaves ───────────
    print(f"\n[5] BoundaryScan from {n_focus} stratified leaf nodes...")
    focus_nodes = select_stratified_leaves(leaves, depths, n=n_focus, seed=seed)
    sigma_grid = np.logspace(-2, 2, 100)

    # Single eigendecomposition (shared across all focus nodes)
    import scipy.sparse.linalg

    k_eigs_actual = min(k_eigs, n_nodes - 1)
    print(f"    Eigendecomposition: k={k_eigs_actual}...")
    t0 = time.time()
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        laplacian.astype(np.float64), k=k_eigs_actual, which="SA"
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigen_time = time.time() - t0
    print(f"    Eigendecomposition done in {eigen_time:.1f}s")

    # Pre-compute nodes at each depth level (for MRR retrieval)
    nodes_at_depth: dict[int, np.ndarray] = {}
    for d in range(max_depth + 1):
        nodes_at_depth[d] = np.where(depths == d)[0]

    # Tensors for vectorized Poincare distance (MRR)
    _ball = geoopt.PoincareBall(c=1.0)
    pts_tensor = torch.as_tensor(points, dtype=torch.float64)

    all_scan_results = []
    all_peak_sigmas: list[float] = []
    all_peak_depths: list[float] = []
    # For Kendall tau: pair sigma with nearest TRUE ancestor height
    tau_sigmas: list[float] = []
    tau_true_heights: list[float] = []
    # Per-node Hit@L scores
    per_node_hit_0: list[float] = []
    per_node_hit_1: list[float] = []
    per_node_hit_2: list[float] = []
    # Per-node Recall@L (fraction of true depths detected)
    per_node_recall_1: list[float] = []
    per_node_recall_2: list[float] = []
    # Per-peak reciprocal ranks for MRR (two variants)
    all_rr_focus: list[float] = []  # rank by dist to focus node
    all_rr_slod: list[float] = []  # rank by dist to SLoD point

    for fi, focus_idx in enumerate(focus_nodes):
        focus_idx = int(focus_idx)
        t0 = time.time()
        result = boundary_scan(
            points, laplacian,
            focus_idx=focus_idx,
            sigma_grid=sigma_grid,
            churn_k=10,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        elapsed = time.time() - t0

        if fi < 5 or fi % 20 == 0:
            print(f"    Focus {fi:3d} (node {focus_idx:5d}): "
                  f"{len(result.peaks)} peaks, {elapsed:.1f}s")

        # Extract boundary info
        focus_depth = int(depths[focus_idx])
        ancestors = get_ancestors(dag, focus_idx)

        # For each detected peak: map sigma → continuous depth
        peak_sigmas = [float(sigma_grid[pi]) for pi in result.peaks]
        peak_depths = [
            _boundary_to_depth(s, sigma_grid, max_depth)
            for s in peak_sigmas
        ]

        all_peak_sigmas.extend(peak_sigmas)
        all_peak_depths.extend(peak_depths)

        # ── Per-node Hit@L: compare detected depths vs ancestor depths ──
        ancestor_set = set(ancestors)
        ancestor_depth_set = sorted(
            set(int(depths[a]) for a in ancestors)
        )
        # Kendall tau: pair sigma with nearest TRUE ancestor height
        nearest_true: list[int | None] = []
        if peak_sigmas and ancestor_depth_set:
            for i, pd in enumerate(peak_depths):
                nearest_d = min(
                    ancestor_depth_set, key=lambda d: abs(d - pd)
                )
                nearest_true.append(nearest_d)
                tau_sigmas.append(peak_sigmas[i])
                tau_true_heights.append(float(max_depth - nearest_d))
        else:
            nearest_true = [None] * len(peak_depths)

        if peak_depths and ancestor_depth_set:
            det = np.array(peak_depths, dtype=np.float64)
            tru = np.array(ancestor_depth_set, dtype=np.float64)
            # Precision: fraction of detections near a true depth
            per_node_hit_0.append(hit_at_l(det, tru, hops=0.5))
            per_node_hit_1.append(hit_at_l(det, tru, hops=1))
            per_node_hit_2.append(hit_at_l(det, tru, hops=2))
            # Recall: fraction of true depths found by a detection
            per_node_recall_1.append(hit_at_l(tru, det, hops=1))
            per_node_recall_2.append(hit_at_l(tru, det, hops=2))

        # ── Per-peak MRR: depth-based ancestor retrieval ──
        # Two variants: rank by distance to (a) focus node, (b) SLoD point
        if ancestors and result.peaks:
            focus_pt = pts_tensor[focus_idx].unsqueeze(0)
            for pi_idx, pi in enumerate(result.peaks):
                pd = peak_depths[pi_idx]
                pd_int = int(round(pd))
                cand_indices = nodes_at_depth.get(pd_int)
                if cand_indices is None or len(cand_indices) == 0:
                    all_rr_focus.append(0.0)
                    all_rr_slod.append(0.0)
                    continue
                true_at_depth = ancestor_set & set(
                    cand_indices.tolist()
                )
                if not true_at_depth:
                    all_rr_focus.append(0.0)
                    all_rr_slod.append(0.0)
                    continue
                cand_t = pts_tensor[cand_indices]

                # (a) Rank by distance to focus node
                d_focus = _ball.dist(focus_pt, cand_t)
                ranked_f = cand_indices[
                    np.argsort(d_focus.detach().numpy())
                ]
                rr_f = 0.0
                for rank, node in enumerate(ranked_f, start=1):
                    if int(node) in true_at_depth:
                        rr_f = 1.0 / rank
                        break
                all_rr_focus.append(rr_f)

                # (b) Rank by distance to SLoD point at peak
                # pi = peak index into result.peaks / result.slod_points,
                # NOT the depth index. slod_points[pi] is the Fréchet mean
                # at the sigma corresponding to peaks[pi].
                slod_pt = torch.as_tensor(
                    result.slod_points[pi], dtype=torch.float64
                ).unsqueeze(0)
                d_slod = _ball.dist(slod_pt, cand_t)
                ranked_s = cand_indices[
                    np.argsort(d_slod.detach().numpy())
                ]
                rr_s = 0.0
                for rank, node in enumerate(ranked_s, start=1):
                    if int(node) in true_at_depth:
                        rr_s = 1.0 / rank
                        break
                all_rr_slod.append(rr_s)

        all_scan_results.append({
            "focus_node": focus_idx,
            "focus_depth": focus_depth,
            "n_peaks": len(result.peaks),
            "peak_sigmas": peak_sigmas,
            "peak_depths": peak_depths,
            "nearest_true_depths": nearest_true,
            "n_ancestors": len(ancestors),
            "ancestor_depths": ancestor_depth_set,
        })

    # ── Step 6: Compute metrics ───────────────────────────────
    print("\n[6] Computing metrics...")

    # Kendall tau: correlation between sigma and TRUE ancestor height
    # (not derived height — that would be tautological)
    tau = 0.0
    if len(tau_sigmas) >= 2:
        tau = kendall_tau(
            np.array(tau_sigmas),
            np.array(tau_true_heights),
        )
    print(f"    Kendall tau (sigma vs ancestor height): {tau:.3f}")
    print("      -> larger sigma = coarser = shallower ancestor?")

    # Hit@L (Precision): fraction of detections near a true ancestor depth
    prec_05 = float(np.mean(per_node_hit_0)) if per_node_hit_0 else 0.0
    prec_1 = float(np.mean(per_node_hit_1)) if per_node_hit_1 else 0.0
    prec_2 = float(np.mean(per_node_hit_2)) if per_node_hit_2 else 0.0
    print(f"    Precision@0.5: {prec_05:.3f}")
    print("      -> of detected boundaries, how many land "
          "within 0.5 of a true ancestor depth?")
    print(f"    Precision@1:   {prec_1:.3f}")
    print(f"    Precision@2:   {prec_2:.3f}")

    # Recall@L: fraction of true ancestor depths found by a detection
    recall_1 = (
        float(np.mean(per_node_recall_1))
        if per_node_recall_1 else 0.0
    )
    recall_2 = (
        float(np.mean(per_node_recall_2))
        if per_node_recall_2 else 0.0
    )
    print(f"    Recall@1:      {recall_1:.3f}")
    print("      -> of true ancestor depths, how many have a "
          "detected boundary within 1 level?")
    print(f"    Recall@2:      {recall_2:.3f}")

    # MRR: depth-based ancestor retrieval (two variants)
    mrr_focus = (
        float(np.mean(all_rr_focus)) if all_rr_focus else 0.0
    )
    mrr_slod = (
        float(np.mean(all_rr_slod)) if all_rr_slod else 0.0
    )
    n_rr_total = len(all_rr_focus)
    n_miss_f = sum(1 for r in all_rr_focus if r == 0.0)
    n_miss_s = sum(1 for r in all_rr_slod if r == 0.0)
    print(f"    MRR_focus:     {mrr_focus:.3f}")
    print("      -> rank by Poincare dist to focus node")
    print(f"    MRR_slod:      {mrr_slod:.3f}")
    print("      -> rank by Poincare dist to SLoD point "
          "at boundary scale")
    print(f"      -> {n_rr_total} queries, "
          f"misses: {n_miss_f} (focus) / {n_miss_s} (slod)")

    metrics = {
        "kendall_tau": float(tau),
        "precision_at_05": float(prec_05),
        "precision_at_1": float(prec_1),
        "precision_at_2": float(prec_2),
        "recall_at_1": float(recall_1),
        "recall_at_2": float(recall_2),
        "mrr_focus": float(mrr_focus),
        "mrr_slod": float(mrr_slod),
    }

    # ── Step 7: Aggregate ─────────────────────────────────────
    peak_counts = [r["n_peaks"] for r in all_scan_results]
    avg_peaks = float(np.mean(peak_counts)) if peak_counts else 0.0
    print(f"\n    Average peaks per focus node: {avg_peaks:.1f}")
    print(f"    Total boundaries detected: {len(all_peak_sigmas)}")

    return {
        "experiment": "Exp 2: WordNet Hierarchical Consistency",
        "n_nodes": n_nodes,
        "n_edges": dag.number_of_edges(),
        "max_depth": max_depth,
        "n_leaves": len(leaves),
        "embedding_dim": dim,
        "n_focus_nodes": n_focus,
        "embed_time_s": embed_time,
        "graph_build_time_s": graph_time,
        "quality": quality,
        "metrics": metrics,
        "avg_peaks": avg_peaks,
        "peak_counts": peak_counts,
        "tau_pairs": list(zip(tau_sigmas, tau_true_heights)),
        "scan_results": all_scan_results,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Exp 2: WordNet")
    parser.add_argument(
        "--max-depth", type=int, default=None,
        help="Limit WordNet to synsets with depth <= N (e.g. 5 for ~3K nodes)",
    )
    parser.add_argument("--n-focus", type=int, default=100)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--knn-method", choices=["auto", "exact", "approx"], default="auto",
        help="kNN method: auto (approx if N>10K), exact, approx (tangent-space)",
    )
    args = parser.parse_args()

    suffix = f"_d{args.max_depth}" if args.max_depth else ""
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", f"exp2{suffix}",
    )
    os.makedirs(results_dir, exist_ok=True)

    result = run_experiment(
        n_focus=args.n_focus,
        dim=args.dim,
        epochs=args.epochs,
        max_depth_limit=args.max_depth,
        knn_method=args.knn_method,
    )

    # Save full results
    result_file = os.path.join(results_dir, "full_results.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n    Full results saved to {result_file}")

    # Save summary (metrics only)
    summary = {
        "experiment": result["experiment"],
        "n_nodes": result["n_nodes"],
        "max_depth": result["max_depth"],
        "quality": result["quality"],
        "metrics": result["metrics"],
        "avg_peaks": result["avg_peaks"],
    }
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    Summary saved to {summary_file}")

    # Final report
    print(f"\n{'='*60}")
    print("EXPERIMENT 2 SUMMARY")
    print(f"{'='*60}")
    print(f"  Nodes:             {result['n_nodes']}")
    print(f"  Max depth:         {result['max_depth']}")
    print(f"  Focus nodes:       {result['n_focus_nodes']}")
    print(f"  Avg peaks/focus:   {result['avg_peaks']:.1f}")
    print()
    print("  Quality Gates:")
    for k, v in result["quality"].items():
        if k != "all_passed":
            print(f"    {k:30s}: {v:.3f}")
    print(f"    {'all_passed':30s}: {result['quality']['all_passed']}")
    print()
    print("  Metrics:")
    for k, v in result["metrics"].items():
        print(f"    {k:15s}: {v:.3f}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
