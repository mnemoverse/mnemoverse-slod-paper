#!/usr/bin/env bash
# Reproduce Tables 3 + 4 (50-seed BCa CIs across 8 ablation configurations).
# 8 configs × 7 r values × 50 seeds = 2800 BoundaryScan runs total.
#
# Runtime: 6-12 hours on a 24-core machine. The pipeline parallelises across
# (r, seed) pairs via parallel_runner.py.
# Requires: pip install -e .
#
# Cached caches (results/exp1/ablation_cache_*.pkl) skip the MDS step on
# repeated runs. The shipped repo has only one such pickle (r=200, seed=42)
# for Tier A demos; Tier C regenerates the rest from scratch on first run.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Tier C: full 50-seed ablation =="
echo "This will take 6-12 hours on a 24-core machine."
echo "Configs: full Poincaré + V-only + D_w-only + C_k-only + direct Laplacian"
echo "         + Euclidean MDS + binary kNN + random points"
echo "r grid:  {20, 40, 60, 80, 100, 150, 200}"
echo "Seeds:   42-91 (50 seeds per cell)"
echo ""
read -p "Continue? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

echo ""
echo "Phase 1/4: full Poincaré + single-indicator variants ..."
python experiments/parallel_runner.py --runner ablation_runner.py

echo ""
echo "Phase 2/4: Euclidean MDS baseline ..."
python experiments/parallel_runner.py --runner ablation_runner_euclidean.py

echo ""
echo "Phase 3/4: random-points + binary kNN controls ..."
python experiments/parallel_runner.py --runner ablation_runner_random_points.py
python experiments/parallel_runner.py --runner ablation_runner_binary_knn.py

echo ""
echo "Phase 4/4: aggregating 50-seed medians + BCa CIs ..."
python experiments/aggregate_ablation.py

echo ""
echo "Done. results/exp1/ablation*.json regenerated."
echo "Verify against paper: python experiments/verify_paper_claims.py"
