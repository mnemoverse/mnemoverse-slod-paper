#!/usr/bin/env python
"""Scaling benchmark for SLoD BoundaryScan on HSBM.

Design notes (addressing Copilot review on PR #59):
- HSBM hierarchy is held CONSTANT across N: 2 macro x 4 meso x 8 micro = 64
  micro-communities. We scale via nodes_per_micro, so the regime doesn't
  change with N. (Previous version let n_micro_per_meso grow with N,
  which degenerated to "1 node per micro" at large N and triggered a
  dense O(total_micro^2) path in the generator.)
- Memory is measured as peak process RSS via a psutil sampling thread,
  so NumPy/SciPy native allocations are accounted for (tracemalloc only
  tracks Python allocations). A tracemalloc figure is recorded alongside
  for reference but the paper uses RSS.
- Output file tracks raw samples; the plot script sorts by n_nodes.

Output: results/scaling/bench.json (incremental; resumable).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import tracemalloc

import networkx as nx
import numpy as np
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slod.utils.data import generate_hsbm
from slod.boundary.spectral import normalized_laplacian
from slod.boundary.scanner import boundary_scan

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "scaling")
OUT_PATH = os.path.join(OUT_DIR, "bench.json")
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed hierarchy: 2 x 4 x 8 = 64 micro-communities regardless of N.
N_MACRO = 2
N_MESO = 4
N_MICRO_PER_MESO = 8
TOTAL_MICRO = N_MACRO * N_MESO * N_MICRO_PER_MESO  # 64

# Pick sizes as multiples of TOTAL_MICRO for an integer nodes-per-micro.
SIZES = [1024, 2048, 4096, 8192, 16384]  # all multiples of 64


class RSSSampler:
    """Background thread sampling psutil RSS at a fixed interval."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_rss = 0
        self._proc = psutil.Process()
        self._stop = False
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop:
            rss = self._proc.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self.interval_s)

    def __enter__(self) -> RSSSampler:
        self.peak_rss = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)


def main() -> None:
    # Resume if prior results exist (but clear them — we changed the
    # HSBM regime and memory metric, so old data is not comparable).
    results: list[dict] = []
    if os.path.exists(OUT_PATH):
        print(f"Note: overwriting {OUT_PATH} (regime/metric changed)")

    for n_req in SIZES:
        nodes_per_micro = n_req // TOTAL_MICRO
        assert n_req % TOTAL_MICRO == 0, f"N={n_req} must be multiple of {TOTAL_MICRO}"
        assert nodes_per_micro >= 1
        print(f"\n=== N={n_req}  ({N_MACRO}x{N_MESO}x{N_MICRO_PER_MESO} micros, "
              f"{nodes_per_micro} nodes/micro) ===")

        t0 = time.perf_counter()
        graph, _labels = generate_hsbm(
            n_nodes=n_req,
            n_macro=N_MACRO,
            n_meso_per_macro=N_MESO,
            n_micro_per_meso=N_MICRO_PER_MESO,
            r=50.0,
            seed=42,
        )
        if not nx.is_connected(graph):
            lcc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(lcc).copy()
            graph = nx.relabel_nodes(
                graph, {o: i for i, o in enumerate(sorted(graph.nodes))}
            )
        n_actual = graph.number_of_nodes()
        m_actual = graph.number_of_edges()
        t_gen = time.perf_counter() - t0
        print(f"  Graph: N={n_actual}, M={m_actual}, gen={t_gen:.2f}s")

        t0 = time.perf_counter()
        adj = nx.adjacency_matrix(graph).astype(np.float64)
        lap = normalized_laplacian(adj)
        t_lap = time.perf_counter() - t0
        print(f"  Laplacian: nnz={lap.nnz}, t={t_lap:.2f}s")

        rng = np.random.RandomState(42)
        points = rng.randn(n_actual, 10) * 0.1

        # Measure both tracemalloc (Python-only) and RSS (full process).
        tracemalloc.start()
        with RSSSampler(interval_s=0.05) as rss:
            rss_before = psutil.Process().memory_info().rss
            t0 = time.perf_counter()
            _result = boundary_scan(points, lap, focus_idx=0, k_eigs=50)
            t_scan = time.perf_counter() - t0
        _, peak_pyalloc_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_rss_mb = rss.peak_rss / 1024 ** 2
        rss_delta_mb = (rss.peak_rss - rss_before) / 1024 ** 2
        peak_pyalloc_mb = peak_pyalloc_bytes / 1024 ** 2

        print(f"  boundary_scan: t={t_scan:.2f}s, "
              f"peak_rss={peak_rss_mb:.1f}MB "
              f"(delta +{rss_delta_mb:.1f}MB), "
              f"peak_pyalloc={peak_pyalloc_mb:.1f}MB")

        results.append({
            "n_nodes_requested": n_req,
            "nodes_per_micro": nodes_per_micro,
            "n_nodes": n_actual,
            "n_edges": m_actual,
            "k_eigs": 50,
            "t_gen_s": t_gen,
            "t_laplacian_s": t_lap,
            "t_scan_s": t_scan,
            "t_total_s": t_gen + t_lap + t_scan,
            "peak_rss_mb": peak_rss_mb,
            "rss_delta_mb": rss_delta_mb,
            "peak_py_alloc_mb": peak_pyalloc_mb,
            "metric_note": "RSS sampled via psutil 50ms; tracemalloc is Python-only",
        })
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} entries to {OUT_PATH}")


if __name__ == "__main__":
    main()
