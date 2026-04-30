#!/usr/bin/env python
"""Run Exp 1 (boundary_scan with full signals) for additional r values.

Fills the gap: run_exp1.py had r=2,5,10,20; we need r=60,80,100,150,200.
(r=40 already computed.)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_exp1 import run_single

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "exp1")
R_VALUES = [60, 80, 100, 150, 200]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i, r in enumerate(R_VALUES):
        path = os.path.join(RESULTS_DIR, f"full_r{r}.json")
        if os.path.exists(path):
            print(f"[{i+1}/{len(R_VALUES)}] r={r} — already exists, skipping")
            continue

        print(f"\n[{i+1}/{len(R_VALUES)}] Running r={r}...")
        t0 = time.time()
        result = run_single(float(r))
        elapsed = time.time() - t0

        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {path} ({elapsed:.0f}s)")

    # Summary
    print("\n\nAll results:")
    for r in [2, 5, 10, 20, 40, 60, 80, 100, 150, 200]:
        path = os.path.join(RESULTS_DIR, f"full_r{r}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            print(f"  r={r:>3}: rho={d['spearman_rho']:.3f}, "
                  f"peaks={d['avg_peaks']:.1f}, "
                  f"nodes={d['n_nodes']}, edges={d['n_edges']}, "
                  f"MDS={d['mds_time_s']:.0f}s")
        else:
            print(f"  r={r:>3}: MISSING")


if __name__ == "__main__":
    main()
