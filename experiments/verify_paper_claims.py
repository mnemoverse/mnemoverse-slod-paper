#!/usr/bin/env python
"""Cross-check quantitative claims in the paper against ablation_agg.json.

This is a guard against the exact class of mistake Copilot + the
ai-research-scientist agent caught in PR #60: writing a bound like
"within ±5pp at every r" that is literally false at one cell of the
supporting table. Every time §6.3 claims a threshold, the corresponding
check lives here. Run before pushing any commit that touches §6.3 or
the aggregated data:

    python experiments/verify_paper_claims.py

Exit code 0 when all claims pass; non-zero when any is violated.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG = os.path.join(ROOT, "results", "exp1", "ablation_agg.json")


def _by_name(agg: list[dict]) -> dict[str, dict]:
    return {e["name"]: e for e in agg}


def _meds(entry: dict, metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r_str, data in entry["by_r"].items():
        m = data.get(f"ari_{metric}_median")
        if m is not None:
            out[int(r_str)] = float(m)
    return out


def check_random_points_vs_default(by_name: dict, failures: list[str]) -> None:
    """Takeaway (a): random-points tracks default within 5pp at macro everywhere
    and at meso everywhere except r=150."""
    default = by_name.get("full")
    random_ = by_name.get("full (random points)")
    if default is None or random_ is None:
        return
    for metric in ("macro", "meso"):
        d = _meds(default, metric)
        r = _meds(random_, metric)
        for r_val in d.keys() & r.keys():
            delta_pp = abs(d[r_val] - r[r_val]) * 100
            # §6.3 claim: within 5pp at every macro r and at 6/7 meso r
            # (the r=150 meso cell is the documented counter-example).
            if metric == "macro" and delta_pp > 5.0:
                failures.append(
                    f"(a) random-vs-default {metric} at r={r_val}: Δ={delta_pp:.1f}pp > 5pp"
                )
            if metric == "meso" and r_val != 150 and delta_pp > 5.0:
                failures.append(
                    f"(a) random-vs-default {metric} at r={r_val}: Δ={delta_pp:.1f}pp > 5pp "
                    f"(only r=150 is allowed to exceed)"
                )


def check_binary_knn_macro(by_name: dict, failures: list[str]) -> None:
    """Takeaway (c): binary kNN macro within 3pp of default at every r."""
    default = by_name.get("full")
    binary = by_name.get("full (binary kNN)")
    if default is None or binary is None:
        return
    dm = _meds(default, "macro")
    bm = _meds(binary, "macro")
    for r_val in dm.keys() & bm.keys():
        delta_pp = abs(dm[r_val] - bm[r_val]) * 100
        if delta_pp > 3.0:
            failures.append(
                f"(c) binary-vs-default macro at r={r_val}: Δ={delta_pp:.1f}pp > 3pp"
            )


def check_binary_knn_meso_lowmid(by_name: dict, failures: list[str]) -> None:
    """Takeaway (c) refined: binary kNN meso within 3pp at r <= 100
    (higher r is explicitly allowed to diverge)."""
    default = by_name.get("full")
    binary = by_name.get("full (binary kNN)")
    if default is None or binary is None:
        return
    dm = _meds(default, "meso")
    bm = _meds(binary, "meso")
    for r_val in sorted(dm.keys() & bm.keys()):
        if r_val > 100:
            continue
        delta_pp = abs(dm[r_val] - bm[r_val]) * 100
        if delta_pp > 3.0:
            failures.append(
                f"(c) binary-vs-default meso at r={r_val} (r<=100 band): "
                f"Δ={delta_pp:.1f}pp > 3pp"
            )


def check_direct_low_r_fail(by_name: dict, failures: list[str]) -> None:
    """Takeaway (a): direct-Laplacian macro is near zero at r <= 40."""
    direct = by_name.get("direct Laplacian")
    if direct is None:
        return
    for r_val in (20, 40):
        d = direct["by_r"].get(str(r_val), {}).get("ari_macro_median")
        if d is None:
            continue
        if d > 0.1:
            failures.append(
                f"(a) direct Laplacian macro at r={r_val}: {d:.3f} > 0.10 "
                f"(paper claims 'fails outright' ARI ≈ 0 at r ≤ 40)"
            )


def check_euclidean_meso_loss(by_name: dict, failures: list[str]) -> None:
    """Takeaway (b) post-rerun: Euclidean MDS shows small but consistent
    meso-ARI loss at r >= 80, concentrated at meso scale.

    Updated expectations from 50-seed BCa CIs (commit post-2026-04-26):
    -6pp at r=80, -4pp at r=100, -10pp at r=150, -8pp at r=200. Sign is
    consistent (Euclidean lower) at r >= 80; magnitude in the 4-10pp band.
    Tolerance ±3pp per cell.
    """
    default = by_name.get("full")
    eucl = by_name.get("full (euclidean)")
    if default is None or eucl is None:
        return
    dm = _meds(default, "meso")
    em = _meds(eucl, "meso")
    expected_pp = {80: 6, 100: 4, 150: 10, 200: 8}
    deltas: dict[int, float] = {}
    for r_val, expected in expected_pp.items():
        if r_val not in dm or r_val not in em:
            continue
        delta_pp = (dm[r_val] - em[r_val]) * 100
        deltas[r_val] = delta_pp
        if abs(delta_pp - expected) > 3.0:
            failures.append(
                f"(b) Euclidean-vs-default meso at r={r_val}: "
                f"Δ={delta_pp:.1f}pp; paper states {expected}pp (±3pp tol)"
            )
    # Direction (Euclidean lower at r >= 80) must hold for at least 3 of 4 cells.
    sign_ok = sum(1 for d in deltas.values() if d >= 0)
    if sign_ok < 3:
        failures.append(
            f"(b) Euclidean direction inconsistent (paper claims Euclidean "
            f"lower at r>=80): only {sign_ok}/4 cells show non-negative delta"
        )


CHECKS = [
    check_random_points_vs_default,
    check_binary_knn_macro,
    check_binary_knn_meso_lowmid,
    check_direct_low_r_fail,
    check_euclidean_meso_loss,
]


def main() -> int:
    """Run every registered check against ablation_agg.json; exit non-zero on failure."""
    if not os.path.exists(AGG):
        print(f"ERROR: {AGG} not found. Run experiments/aggregate_ablation.py first.",
              file=sys.stderr)
        return 2
    with open(AGG, encoding="utf-8") as f:
        agg = json.load(f)
    by_name = _by_name(agg)

    failures: list[str] = []
    for check in CHECKS:
        check(by_name, failures)

    if failures:
        print("PAPER CLAIM VIOLATIONS:")
        for f in failures:
            print(f"  - {f}")
        print(f"\n{len(failures)} violation(s). Either fix the data or soften the paper text.")
        return 1

    print("All §6.3 quantitative claims pass against the aggregated data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
