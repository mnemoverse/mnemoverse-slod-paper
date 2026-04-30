# Contributing

This repository is a **public code companion** to the arXiv preprint
**[arXiv:2603.08965](https://arxiv.org/abs/2603.08965)** (Edward Izgorodin,
*Semantic Level of Detail for Knowledge Graphs: Discovering Abstraction
Boundaries via Spectral Heat Diffusion*, Mnemoverse.AI, 2026).

The code in this repository is **frozen at paper-submission state** so that
results in the paper remain reproducible. We do not accept feature pull
requests against the paper's published content. We do welcome:

## Reporting reproducibility issues

If you cannot reproduce a number from the paper using the scripts in this
repo, please open a GitHub Issue with:

- The exact command you ran (e.g. `bash scripts/repro_table1.sh`)
- The output you got vs. the expected output (paper Table / Figure / numerical claim)
- Your environment: OS, Python version, output of `pip list | grep -iE "numpy|scipy|geoopt|torch"`
- For `verify_appendix_a.py`, the full stdout

We aim to respond within a few working days. The paper's numerical
reproducibility is part of the submission contract.

## Reporting bugs in the released code

Open a GitHub Issue. Include a minimal reproduction. We may patch via a
post-publication PR labelled `[errata]` if the bug affects published
numbers; otherwise we'll annotate the issue and fix in a future release.

## Citing this work

If you use this code, please cite the paper:

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

A `CITATION.cff` is also provided at the repository root for tools that
parse it automatically (Zotero, GitHub "Cite this repository", Zenodo).

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). By contributing
an issue or comment, you agree that any incidental code suggestions are
licensed under the same terms.

## Patent notice

The methods implemented here are the subject of provisional patent
application **INPI 20262007760519** (filed 2026-03-03, INPI Portugal). The
Apache-2.0 patent grant covers the released implementation; patent rights
are retained for other implementations and embodiments. This affects
commercial use only — academic and research use is fully covered by the
Apache-2.0 grant.

## Contact

For paper questions or commercial-licensing inquiries, contact the author:
**Edward Izgorodin** — `izgorodin@me.com` — [Mnemoverse.AI](https://mnemoverse.com).
