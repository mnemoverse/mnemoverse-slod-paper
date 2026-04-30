#!/usr/bin/env bash
# Reproduce Table 1 (HSBM macro/meso/micro ARI, single seed=42) + Figure 2
# (K*(σ) trajectory across seven r values, planted 2x4x8 hierarchy).
#
# Runtime: ~25 minutes single-threaded on an M-series CPU.
# Requires: pip install -e . (this repo)

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Tier B: reproducing Table 1 + Figure 2 =="
echo "Sweeping r in {20, 40, 60, 80, 100, 150, 200} with seed=42 ..."
python experiments/exp1_hsbm/run_exp1_multir.py

echo ""
echo "Generating Figure 2 (K*(σ) trajectory) ..."
python experiments/exp1_hsbm/plot_exp1.py

echo ""
echo "Done. Results in results/exp1/multir_direct.json (Table 1 source data)."
echo "Figure 2 written to paper/figures/H_kstar_trajectory.pdf"
