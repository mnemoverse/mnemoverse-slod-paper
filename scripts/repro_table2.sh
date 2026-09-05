#!/usr/bin/env bash
# Reproduce Table 2 (WordNet noun hierarchy, 82,115 synsets, Kendall τ = 0.79
# between detected boundary scales σ* and true ancestor depth on 100
# stratified leaf foci).
#
# Runtime: about 15 minutes (measured 2026-09-05, x86, 8 threads): Poincaré
# embedding ~1 min, kNN graph ~1 min, Lanczos eigendecomposition ~5 s,
# BoundaryScan on 100 foci ~10 min. BoundaryScan dominates, not Lanczos.
# CPU only: the Exp 2 code path never selects a CUDA device.
#
# Required external data:
#   - NLTK WordNet corpus (downloaded automatically via nltk.download).
#   - Nickel-Kiela Poincaré embedding (β=10, dim=10): the script will
#     attempt to fetch a pre-trained checkpoint, or train one from scratch
#     if not found.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Tier C: WordNet Kendall τ reproduction =="
echo "This takes about 15 minutes on a modern CPU."
echo ""

echo "Step 1: ensure NLTK WordNet corpus is available ..."
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

echo ""
echo "Step 2: load Nickel-Kiela Poincaré embedding for WordNet (β=10, dim=10) ..."
echo "  (See experiments/exp2_wordnet/run_exp2.py for embedding source/checkpoint logic.)"

echo ""
echo "Step 3: BoundaryScan from 100 stratified leaf foci ..."
python experiments/exp2_wordnet/run_exp2.py

echo ""
echo "Step 4: generate Figure 4 (WordNet boundary-scale scatter) ..."
python experiments/exp2_wordnet/plot_exp2.py

echo ""
echo "Done. Expected output: Kendall τ ≈ 0.79 [0.75, 0.83]."
echo "Result file: results/exp2/wordnet_metrics.json"
