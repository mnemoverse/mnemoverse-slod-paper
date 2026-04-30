"""Parallel orchestrator for the 4 ablation runners.

Spawns one subprocess per (runner, seed) pair, each writing to its own
per-seed JSON file under results/exp1/_parallel/. After all subprocesses
complete, run experiments/merge_parallel_outputs.py to fold them into
the canonical results/exp1/ablation*.json files.

Why per-seed outputs: each runner currently does `json.dump(results, f)`
(full overwrite) every time it finishes a (r, seed, cfg). Two parallel
workers writing to the same JSON would corrupt it. Per-seed paths
eliminate the contention entirely.

Usage:

    python experiments/parallel_runner.py \\
        --seeds 42-91 \\
        --workers 16 \\
        --machine 5090

    python experiments/parallel_runner.py \\
        --seeds 92-111 \\
        --workers 8 \\
        --machine laptop-m \\
        --skip-runner euclidean

The script is restart-safe: each (runner, seed) checks whether its
per-seed JSON already exists with the expected number of (r, name)
entries before re-running.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

ROOT = Path(__file__).resolve().parent.parent
PARALLEL_DIR = ROOT / "results" / "exp1" / "_parallel"

# (short_name, script_filename, configs_per_seed expected for completion check)
RUNNERS: dict[str, tuple[str, int]] = {
    "main":      ("ablation_runner.py",                7 * 7),  # 7 r × 7 configs
    "euclidean": ("ablation_runner_euclidean.py",      7 * 4),  # 7 r × 4 configs
    "random":    ("ablation_runner_random_points.py",  7 * 2),  # 7 r × 2 configs
    "binary":    ("ablation_runner_binary_knn.py",     7 * 4),  # 7 r × 4 configs
}

# Dependencies: random/binary read the Poincaré cache produced by main.
# Wrapper executes runners in topological waves so the cache is on disk before
# the dependents start. Within a wave, all (runner, seed) jobs run in parallel.
DEPENDS_ON: dict[str, list[str]] = {
    "main":      [],
    "euclidean": [],
    "random":    ["main"],
    "binary":    ["main"],
}


@dataclass
class Job:
    runner: str
    seed: int
    out_path: Path

    @property
    def label(self) -> str:
        return f"{self.runner}/seed={self.seed}"


def parse_seeds(spec: str) -> list[int]:
    """Parse '42-91' or '42,43,44' or '42-46,50,55-58'."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def per_seed_path(runner: str, seed: int) -> Path:
    return PARALLEL_DIR / f"{runner}_seed{seed}.json"


def is_complete(job: Job) -> bool:
    """True iff per-seed JSON has the expected entry count for this runner."""
    if not job.out_path.exists():
        return False
    try:
        records = json.loads(job.out_path.read_text())
    except Exception:
        return False
    expected = RUNNERS[job.runner][1]
    return len(records) >= expected


def run_one(job: Job) -> tuple[Job, int, float, str]:
    """Spawn the runner subprocess for a single (runner, seed) pair.

    Returns (job, exit_code, wallclock_seconds, log_path).
    """
    script = RUNNERS[job.runner][0]
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = PARALLEL_DIR / f"{job.runner}_seed{job.seed}.log"

    env = os.environ.copy()
    env["SLOD_SEEDS"] = str(job.seed)
    env["SLOD_OUT_PATH"] = str(job.out_path)
    # Force single-threaded BLAS so that workers don't oversubscribe cores.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / script)],
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return job, proc.returncode, time.time() - t0, str(log_path)


def heartbeat(state: dict, stop: Event, machine: str, total: int) -> None:
    """Background thread: write progress every 60s to RERUN_PROGRESS_{machine}.log."""
    progress_path = ROOT / "results" / "exp1" / f"RERUN_PROGRESS_{machine}.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while not stop.is_set():
        done = state["done"]
        failed = state["failed"]
        elapsed = time.time() - started
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        remaining = total - done - failed
        eta_s = remaining / rate if rate > 0 else 0.0
        eta_h = eta_s / 3600.0
        line = (
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"machine={machine} done={done}/{total} failed={failed} "
            f"elapsed={elapsed/3600.0:.2f}h rate={rate:.2f}job/s "
            f"ETA={eta_h:.2f}h\n"
        )
        with progress_path.open("a") as f:
            f.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()
        stop.wait(60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="e.g. '42-91' or '42,43,46'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--machine", required=True, help="machine id, e.g. '5090' or 'laptop-m'")
    ap.add_argument(
        "--runners",
        default="main,euclidean,random,binary",
        help="comma-separated runner short names",
    )
    ap.add_argument(
        "--skip-runner",
        default="",
        help="comma-separated runner short names to skip (e.g. 'euclidean' on laptop)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="how many times to retry a failed (runner, seed) before giving up",
    )
    args = ap.parse_args()

    PARALLEL_DIR.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    requested = [r.strip() for r in args.runners.split(",") if r.strip()]
    skipped = {r.strip() for r in args.skip_runner.split(",") if r.strip()}
    runners = [r for r in requested if r in RUNNERS and r not in skipped]
    unknown = [r for r in requested if r not in RUNNERS]
    if unknown:
        print(f"Unknown runner(s): {unknown}; valid: {list(RUNNERS)}", file=sys.stderr)
        return 2
    if not runners:
        print("No runners selected after skip filter.", file=sys.stderr)
        return 2

    # Group runners into dependency waves (topological order).
    waves: list[list[str]] = []
    placed: set[str] = set()
    while len(placed) < len(runners):
        ready = [
            r for r in runners
            if r not in placed and all(d in placed or d not in runners for d in DEPENDS_ON[r])
        ]
        if not ready:
            print(f"ERROR: dependency cycle or unmet dep among {runners}", file=sys.stderr)
            return 2
        waves.append(ready)
        placed.update(ready)

    # Build job lists per wave, skipping already-complete pairs.
    pre_complete = 0
    wave_jobs: list[list[Job]] = []
    total_jobs = 0
    for wave in waves:
        wjobs: list[Job] = []
        for runner in wave:
            for seed in seeds:
                job = Job(runner=runner, seed=seed, out_path=per_seed_path(runner, seed))
                if is_complete(job):
                    pre_complete += 1
                    continue
                wjobs.append(job)
                total_jobs += 1
        wave_jobs.append(wjobs)

    total = total_jobs + pre_complete
    print(f"Plan: machine={args.machine} runners={runners} waves={waves} "
          f"seeds={len(seeds)} workers={args.workers} total_jobs={total} "
          f"already_done={pre_complete} to_run={total_jobs}")

    if total_jobs == 0:
        print("Nothing to do (all complete).")
        return 0

    state = {"done": pre_complete, "failed": 0}
    stop = Event()
    hb = Thread(target=heartbeat, args=(state, stop, args.machine, total), daemon=True)
    hb.start()

    attempts: dict[tuple[str, int], int] = {}
    failures: list[tuple[Job, int, str]] = []

    for wave_idx, jobs in enumerate(wave_jobs):
        if not jobs:
            continue
        wave_runners = sorted({j.runner for j in jobs})
        print(f"\n=== Wave {wave_idx + 1}/{len(wave_jobs)}: {wave_runners} ({len(jobs)} jobs) ===",
              flush=True)
        pending = list(jobs)
        while pending:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(run_one, j): j for j in pending}
                pending = []
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        j, rc, secs, log = fut.result()
                    except Exception as e:
                        rc, secs, log, j = -1, 0.0, "<exception>", job
                        print(f"[ERROR] {job.label} raised {e}", flush=True)
                    key = (j.runner, j.seed)
                    if rc == 0 and is_complete(j):
                        state["done"] += 1
                        print(f"[OK]   {j.label} ({secs:.1f}s)", flush=True)
                    else:
                        attempts[key] = attempts.get(key, 0) + 1
                        if attempts[key] <= args.retries:
                            print(f"[RETRY {attempts[key]}/{args.retries}] "
                                  f"{j.label} rc={rc}", flush=True)
                            pending.append(j)
                        else:
                            state["failed"] += 1
                            failures.append((j, rc, log))
                            print(f"[FAIL] {j.label} rc={rc} log={log}", flush=True)

    stop.set()
    hb.join(timeout=2)

    print(f"\nDone: {state['done']}/{total} ok, {state['failed']} failed.")
    if failures:
        print("Failures:")
        for j, rc, log in failures:
            print(f"  {j.label} rc={rc} log={log}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
