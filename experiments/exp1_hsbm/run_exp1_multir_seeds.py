"""Exp 1 Multi-r × multi-seed sweep — collects gap_ratio_2 for Figure 2 panel (b)
band rendering.

Runs run_single_r() for each (r, seed) in {20,40,60,80,100,150,200} × seeds.
Aggregates gap_ratio_2 and fixed_ari_{macro,meso,micro} across seeds.

Output: results/exp1/multir_seeds.json with per-r per-seed records + aggregated
median/IQR/min/max.

Usage:
    python experiments/exp1_hsbm/run_exp1_multir_seeds.py [--n-seeds N] [--seed-start S]

Default: 10 seeds starting at 42 (matches §6.3 ablation seed range 42-51, a subset
of the 42-91 used there).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_exp1_multir import run_single_r  # noqa: E402

R_VALUES = [20, 40, 60, 80, 100, 150, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "exp1", "multir_seeds.json"))
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"Sweeping r={R_VALUES} × seeds={seeds}")

    per_seed = []
    t_total = time.time()
    for seed in seeds:
        for r in R_VALUES:
            t0 = time.time()
            res = run_single_r(r=r, seed=seed)
            elapsed = time.time() - t0
            per_seed.append({
                "r": r,
                "seed": seed,
                "gap_ratio_2": res["gap_ratio_2"],
                "gap_ratio_8": res["gap_ratio_8"],
                "fixed_ari_macro": res["fixed_ari_macro"],
                "fixed_ari_meso": res["fixed_ari_meso"],
                "fixed_ari_micro": res["fixed_ari_micro"],
                "n_nodes": res["n_nodes"],
                "elapsed_s": elapsed,
            })
            print(f"  r={r}, seed={seed}: gap={res['gap_ratio_2']:.3f}, "
                  f"macro={res['fixed_ari_macro']:.3f}, meso={res['fixed_ari_meso']:.3f}, "
                  f"micro={res['fixed_ari_micro']:.3f} [{elapsed:.1f}s]")
            # Incremental save (resumable in spirit)
            with open(args.out, "w") as f:
                json.dump({"per_seed": per_seed, "seeds": seeds, "r_values": R_VALUES}, f, indent=2)

    # Aggregate
    print("\nAggregating across seeds...")
    aggregated = []
    for r in R_VALUES:
        rows = [p for p in per_seed if p["r"] == r]
        gaps = np.array([p["gap_ratio_2"] for p in rows])
        macros = np.array([p["fixed_ari_macro"] for p in rows])
        mesos = np.array([p["fixed_ari_meso"] for p in rows])
        micros = np.array([p["fixed_ari_micro"] for p in rows])
        agg = {
            "r": r,
            "n_seeds": len(rows),
            "gap_ratio_2_median": float(np.median(gaps)),
            "gap_ratio_2_q25": float(np.quantile(gaps, 0.25)),
            "gap_ratio_2_q75": float(np.quantile(gaps, 0.75)),
            "gap_ratio_2_min": float(np.min(gaps)),
            "gap_ratio_2_max": float(np.max(gaps)),
            "fixed_ari_macro_median": float(np.median(macros)),
            "fixed_ari_meso_median": float(np.median(mesos)),
            "fixed_ari_micro_median": float(np.median(micros)),
        }
        aggregated.append(agg)
        print(f"  r={r}: gap median={agg['gap_ratio_2_median']:.3f} "
              f"IQR=[{agg['gap_ratio_2_q25']:.3f}, {agg['gap_ratio_2_q75']:.3f}] "
              f"range=[{agg['gap_ratio_2_min']:.3f}, {agg['gap_ratio_2_max']:.3f}]")

    with open(args.out, "w") as f:
        json.dump({
            "per_seed": per_seed,
            "aggregated": aggregated,
            "seeds": seeds,
            "r_values": R_VALUES,
        }, f, indent=2)

    print(f"\nTotal time: {(time.time() - t_total)/60:.1f} min")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
