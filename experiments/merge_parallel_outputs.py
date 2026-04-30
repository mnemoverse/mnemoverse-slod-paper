"""Merge per-seed JSONs from results/exp1/_parallel/ into canonical paths.

The parallel orchestrator (parallel_runner.py) writes to per-seed paths to
avoid JSON write contention between parallel workers. This script folds those
back into:
    results/exp1/ablation.json              (from main_seed*.json)
    results/exp1/ablation_euclidean.json    (from euclidean_seed*.json)
    results/exp1/ablation_random_points.json (from random_seed*.json)
    results/exp1/ablation_binary_knn.json   (from binary_seed*.json)

Idempotent: dedups on (r, seed, name). Existing canonical JSONs are read,
augmented with per-seed entries, and atomically rewritten.

Usage:
    python experiments/merge_parallel_outputs.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARALLEL_DIR = ROOT / "results" / "exp1" / "_parallel"
RESULTS_DIR = ROOT / "results" / "exp1"

# (short_name, glob_pattern, canonical_path)
SOURCES = [
    ("main",      "main_seed*.json",      RESULTS_DIR / "ablation.json"),
    ("euclidean", "euclidean_seed*.json", RESULTS_DIR / "ablation_euclidean.json"),
    ("random",    "random_seed*.json",    RESULTS_DIR / "ablation_random_points.json"),
    ("binary",    "binary_seed*.json",    RESULTS_DIR / "ablation_binary_knn.json"),
]


def _atomic_json_dump(obj, path: Path) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def merge_one(short: str, pattern: str, canonical: Path) -> tuple[int, int, int]:
    """Returns (existing_kept, per_seed_added, total_after)."""
    existing: list[dict] = []
    if canonical.exists():
        try:
            existing = json.loads(canonical.read_text())
        except Exception as e:
            print(f"  [{short}] WARNING: canonical {canonical.name} unreadable ({e}); starting fresh")
            existing = []

    seen: dict[tuple, dict] = {}
    for r in existing:
        seen[(r.get("r"), r.get("seed"), r["name"])] = r
    existing_kept = len(seen)

    added = 0
    for path in sorted(PARALLEL_DIR.glob(pattern)):
        try:
            recs = json.loads(path.read_text())
        except Exception as e:
            print(f"  [{short}] WARNING: {path.name} unreadable ({e}); skipping")
            continue
        for rec in recs:
            key = (rec.get("r"), rec.get("seed"), rec["name"])
            if key not in seen:
                seen[key] = rec
                added += 1

    merged = list(seen.values())
    if added > 0 or not canonical.exists():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(merged, canonical)
    return existing_kept, added, len(merged)


def main() -> int:
    if not PARALLEL_DIR.exists():
        print(f"No {PARALLEL_DIR} directory found — nothing to merge.")
        return 0

    print(f"Merging from {PARALLEL_DIR}")
    print(f"{'runner':<10} {'kept':>6} {'+added':>7} {'=total':>8} {'canonical':<35}")
    print("-" * 70)
    for short, pattern, canonical in SOURCES:
        kept, added, total = merge_one(short, pattern, canonical)
        print(f"{short:<10} {kept:>6} {added:>+7} {total:>8} {canonical.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
