# Semantic Level of Detail for Knowledge Graphs

[![arXiv](https://img.shields.io/badge/arXiv-2603.08965-b31b1b.svg)](https://arxiv.org/abs/2603.08965)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Public code companion to **arXiv:2603.08965** — Edward Izgorodin (Mnemoverse.AI),
*Semantic Level of Detail for Knowledge Graphs: Discovering Abstraction Boundaries
via Spectral Heat Diffusion* (extended preprint, 2026).

## Cite

```bibtex
@misc{izgorodin2026slod,
  author       = {Izgorodin, Edward},
  title        = {Semantic Level of Detail for Knowledge Graphs:
                  Discovering Abstraction Boundaries via Spectral Heat Diffusion},
  year         = {2026},
  eprint       = {2603.08965},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2603.08965},
}
```

## TL;DR

- **What.** A continuous-zoom operator over knowledge graphs: heat-kernel
  diffusion on a kNN graph induced by a Poincaré-ball embedding, with
  Fréchet-centroid readout.
- **Why.** Existing community detection (Leiden / modularity) needs a manually
  tuned resolution parameter `γ`; SLoD reads boundaries off spectral gaps.
- **Result.** On HSBM (1024 nodes), macro ARI saturates at 1.00 in the
  high-SNR regime (50-seed median), meso ARI reaches 0.89 [0.86, 0.92] at
  *r*=200; on full WordNet (82K synsets), Kendall τ = 0.79 between
  detected boundary scales and true ancestor depth.
- **Theory.** Hierarchical-coherence guarantee with bounded
  `(1+ε)`-distortion error along the centroid trajectory (Theorem 1, full
  proofs of Lemmas A.1 and A.2 in Appendix A of the paper).

## Reproducing the paper

Three tiers, by time budget:

| Tier | Command | Reproduces | Time | Hardware |
|---|---|---|---|---|
| **A — sanity (~10s)** | `python scripts/verify_appendix_a.py` | Lemma A.2 numerics (`K_{1,100}`: ‖ẇ‖₁ = 20, bound 22) | <1s | any |
| **A** | `python scripts/plot_dw_vs_inv_lambda.py` | Figure 1 (D_w(σ) anchor for Prop 1(i)) from cached r=200 Laplacian | ~5s | any |
| **B — main paper figures (~30 min)** | `bash scripts/repro_table1.sh` | Table 1 (HSBM macro/meso/micro ARI, single seed=42) + Figure 2 (K*(σ) trajectory) | ~25 min | M-series CPU or x86 |
| **B** | `bash scripts/repro_figures.sh` | Figures 3, 5, 6 (phase-transition, ablation profiles, scaling) from cached JSONs | ~10s | any |
| **C — full ablation (hours)** | `bash scripts/repro_tables_3_4.sh` | Tables 3+4 (50-seed BCa CIs, Poincaré + 7 ablation configs × 7 *r* values = 2800 runs) | 6–12h | 24-core preferred |
| **C** | `bash scripts/repro_table2.sh` | WordNet Kendall τ (downloads NLTK WordNet + Nickel-Kiela embedding) | 1–2h | helpful: GPU |

After Tier B + cached JSONs in `results/exp1/`, all six figures and Tables 1, 3, 4 in
the paper PDF can be regenerated. Tier C is for full BCa CIs from scratch.

## Install

```bash
git clone https://github.com/mnemoverse/mnemoverse-slod-paper.git
cd mnemoverse-slod-paper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Smoke test (under 10 seconds):
```bash
python scripts/verify_appendix_a.py
```

Expected output ends with:
```
Bounds at hub focus near sigma = 0 (taking ||dot w||_1 = 20.00):
  Original Lemma A.2: lambda_max          = 2.00  VIOLATED  (20.00 > 2.00)
  Reviewer's 2*lambda_max/Sigma_min       = 4.00  VIOLATED  (20.00 > 4.00)
  Corrected 2*||L||_{1->1}                = 22.00  HOLDS    (20.00 <= 22.00)
```

## Repository layout

```
src/slod/                  Importable package (Apache-2.0)
  core/                    P1: Poincaré geometry + heat kernel + Fréchet centroid (Algorithm 1)
  boundary/                P2/P6: BoundaryScan + indicators (V, D_w, C_k) + Multi-Center
  utils/                   HSBM generator, metrics (ARI, Kendall τ), HP/SP gates, WordNet loader

experiments/               One entrypoint per paper experiment
  exp1_hsbm/               Tables 1, Figures 2, 3 (HSBM K*(σ), phase transition)
  exp2_wordnet/            Table 2, Figure 4 (WordNet Kendall τ)
  ablation_runner*.py      Tables 3, 4, Figure 5 (50-seed ablation, 8 configurations)
  *.py                     Aggregation, parallel orchestration, scaling benchmark

scripts/
  verify_appendix_a.py     Tier A: Lemma A.2 K_{1,n} numerical verification
  plot_dw_vs_inv_lambda.py Tier A: Figure 1 (D_w peak alignment with 1/λ_k)
  repro_table1.sh          Tier B: HSBM single-seed table
  repro_figures.sh         Tier B: figures from cached JSONs
  repro_tables_3_4.sh      Tier C: full 50-seed ablation
  repro_table2.sh          Tier C: WordNet Kendall τ

results/exp1/              Cached numerical artefacts (paper Tables 1, 3, 4 source data)

paper/                     Canonical paper artefacts
  slod_arxiv_v2.pdf        22-page extended preprint
  slod_arxiv_v2.tex        LaTeX source (lualatex + bibtex)
  slod.bib                 Bibliography
  slod_arxiv_v2_arxiv-bundle.tar.gz   The exact tarball uploaded to arXiv
  figures/                 PDFs/PNGs cited in the paper
  tables/                  ablation.tex (auto-generated by aggregate_ablation.py)

tests/                     pytest unit tests for the public package
```

## Hardware and runtime estimates

All paper numbers were measured single-threaded on an Apple M-series CPU
with 16 GB RAM (see §5 of the paper). Lanczos partial eigendecomposition
is the dominant cost for fixed `K_eigs`; wall-clock scales as `N^{0.77}`
across 978–15,771 nodes.

- HSBM 1024 nodes, `K_eigs`=80 — 25–30 s per *r* value
- HSBM 1024 nodes, full ablation (8 configs × 7 *r* × 50 seeds) — 6–12 h on 24 cores
- WordNet 82,115 synsets — Lanczos eigendecomposition ~5 min, BoundaryScan
  on 100 stratified leaf foci ~10 min

## Results table (paper Table 4 highlights)

| Configuration       | meso ARI at *r*=200 | 95% CI       |
|---|---|---|
| Full Poincaré       | **0.89**            | [0.86, 0.92] |
| Direct Laplacian    | 0.71                | [0.67, 0.76] |
| Euclidean MDS       | 0.81                | [0.79, 0.83] |
| Random points       | 0.84                | [0.79, 0.89] |
| Binary kNN          | 0.87                | [0.85, 0.90] |

Cross-cell paired effect (Poincaré − Euclidean, 200 paired
(*r*, seed) pairs at *r* ∈ {80, 100, 150, 200}): 5.7 pp meso ARI gap,
BCa 95% CI [4.8, 6.7] pp, Wilcoxon signed-rank p < 10⁻¹⁵.

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

The methods implemented here are the subject of provisional patent
application **INPI 20262007760519** (filed 2026-03-03, INPI Portugal).
The Apache-2.0 patent grant applies to the released implementation only;
patent rights are retained for other implementations.

## Contact

Edward Izgorodin — `izgorodin@me.com` — [@izgorodin](https://github.com/izgorodin) — Mnemoverse.AI, Funchal, Madeira, Portugal.

For paper questions, open a GitHub issue or email. For commercial
licensing inquiries, please contact the author directly.

## Acknowledgements

This is a discussion paper presented at the 1st Workshop on Graphs Across
AI (GRAAI), IEEE WCCI 2026, Maastricht. Thanks to the two anonymous GRAAI
reviewers and to several rounds of pre-submission adversarial review that
materially improved the manuscript.
