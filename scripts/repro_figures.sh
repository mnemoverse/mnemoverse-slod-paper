#!/usr/bin/env bash
# Reproduce paper figures from cached numerical results.
# Figure 1 (D_w anchor) — from cached r=200 Laplacian pickle.
# Figure 3 (phase transition) — from cached multir_seeds.json (10-seed sweep).
# Figure 5 (ablation profiles) — from cached ablation.json.
# Figure 6 (scaling) — separate benchmark (run experiments/bench_scaling.py to refresh).
#
# Runtime: under 30 seconds.
# Requires: pip install -e . and cached results in results/exp1/.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Tier B: reproducing figures from cached results =="

echo "Figure 1 (D_w(σ) vs 1/λ_k empirical anchor) ..."
python scripts/plot_dw_vs_inv_lambda.py

echo ""
echo "Figure 5 (ablation profiles, single-seed=42) ..."
python experiments/plot_ablation_profiles.py 2>/dev/null \
  || echo "  (skipped — plot_ablation_profiles.py needs ablation_cache pickles for seed 42; rerun from Tier C if missing)"

echo ""
echo "Figure 3 (phase transition) requires regenerating per-r outputs."
echo "Run: bash scripts/repro_table1.sh first, then this script."

echo ""
echo "Done."
