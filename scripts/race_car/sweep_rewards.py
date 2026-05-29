#!/usr/bin/env python3
"""Fan out race-car SAC training across algorithms, reward configs, and seeds.

Each job invokes the matching Python runner (``run_sac_fop.py`` or
``run_sac_zop.py``) in its own subprocess with a distinct ``OUT`` directory and
the chosen reward CLI flags appended. Runs are launched concurrently up to
``--max-parallel`` at a time; a manifest of the full sweep is written to
``<sweep-root>/manifest.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Named reward configurations. Each value is a list of trailing CLI flags
# forwarded to run_sac_{fop,zop}.py (via the shell wrapper's "$@").
# See scripts/race_car/sac_race_car_mixin.py for the reward CLI surface and
# leap_c/examples/race_car/env.py for the per-term semantics.
REWARD_CONFIGS: dict[str, list[str]] = {
    # Recommended default: primarily rewards progress (ds per step), adds a
    # small per-step time penalty, strong off-track penalty, and a large
    # terminal bonus when the lap completes. Matches the "hybrid" preset with
    # c=0.1, bonus=100.0, violation=50.0.
    "balanced": ["--reward-mode", "hybrid"],
    # Pure arc-length progression, no terminal bonus or penalties.
    "progress_only": ["--reward-mode", "progress"],
    # Sparse lap-time reward with +100 on finish.
    "lap_time": ["--reward-mode", "lap_time"],
    # Variants around "balanced" for quick sensitivity sweeps.
    "balanced_strict_violation": [
        "--reward-mode",
        "hybrid",
        "--reward-w-violation",
        "100.0",
    ],
    "balanced_low_time": [
        "--reward-mode",
        "hybrid",
        "--reward-w-time",
        "0.05",
    ],
    "balanced_high_bonus": [
        "--reward-mode",
        "hybrid",
        "--reward-w-bonus",
        "200.0",
    ],
}

# Python runners for each algorithm. We inline the standard flags
# (--with-val, --reuse-code, --seed, --controller, --output-path, --device,
# --dtype) below in build_jobs.
RUNNERS = {
    "fop": SCRIPT_DIR / "run_sac_fop.py",
    "zop": SCRIPT_DIR / "run_sac_zop.py",
}


@dataclass
class Job:
    algo: str
    config: str
    seed: int
    out: Path
    reward_flags: list[str]
    extra_flags: list[str]
    cmd: list[str]
    log_path: Path
    pid: int | None = None
    returncode: int | None = None


def build_jobs(
    algos: list[str],
    configs: list[str],
    seeds: list[int],
    controller: str,
    variant: str,
    device: str,
    dtype: str,
    sweep_root: Path,
    extra_flags: list[str],
) -> list[Job]:
    jobs: list[Job] = []
    for algo, config_name, seed in itertools.product(algos, configs, seeds):
        if algo not in RUNNERS:
            raise ValueError(f"unknown algo {algo!r}")
        if config_name not in REWARD_CONFIGS:
            raise ValueError(
                f"unknown reward config {config_name!r}. Known: {sorted(REWARD_CONFIGS)}"
            )
        reward_flags = REWARD_CONFIGS[config_name]
        out = sweep_root / f"{algo}_{config_name}_seed{seed}"
        runner_flags = [
            "--seed",
            str(seed),
            "--controller",
            controller,
            "--output-path",
            str(out),
            "--with-val",
            "--reuse-code",
            "--device",
            device,
            "--dtype",
            dtype,
        ]
        if algo == "fop":
            runner_flags[4:4] = ["--variant", variant]
        cmd = [
            sys.executable,
            str(RUNNERS[algo]),
            *runner_flags,
            *reward_flags,
            *extra_flags,
        ]
        jobs.append(
            Job(
                algo=algo,
                config=config_name,
                seed=seed,
                out=out,
                reward_flags=list(reward_flags),
                extra_flags=list(extra_flags),
                cmd=cmd,
                log_path=out / "launcher.log",
            )
        )
    return jobs


def launch(job: Job) -> subprocess.Popen:
    job.out.mkdir(parents=True, exist_ok=True)
    log = open(job.log_path, "w", buffering=1)
    proc = subprocess.Popen(
        job.cmd,
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    job.pid = proc.pid
    proc._log_handle = log  # type: ignore[attr-defined]
    return proc


def run_sweep(jobs: list[Job], max_parallel: int) -> None:
    pending = list(jobs)
    running: list[tuple[Job, subprocess.Popen]] = []
    print(f"[sweep] launching {len(pending)} jobs, up to {max_parallel} in parallel")
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            proc = launch(job)
            running.append((job, proc))
            print(
                f"[sweep] start pid={proc.pid} algo={job.algo} "
                f"config={job.config} seed={job.seed} out={job.out}"
            )
        still_running: list[tuple[Job, subprocess.Popen]] = []
        for job, proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((job, proc))
            else:
                job.returncode = rc
                proc._log_handle.close()  # type: ignore[attr-defined]
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(
                    f"[sweep] done  pid={proc.pid} algo={job.algo} "
                    f"config={job.config} seed={job.seed} {status}"
                )
        running = still_running
        if running:
            time.sleep(1.0)


def write_manifest(sweep_root: Path, jobs: list[Job]) -> None:
    manifest = {
        "sweep_root": str(sweep_root),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "jobs": [
            {
                "algo": j.algo,
                "config": j.config,
                "seed": j.seed,
                "out": str(j.out),
                "reward_flags": j.reward_flags,
                "extra_flags": j.extra_flags,
                "cmd": j.cmd,
                "pid": j.pid,
                "returncode": j.returncode,
                "log_path": str(j.log_path),
            }
            for j in jobs
        ],
    }
    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--algos", nargs="+", choices=sorted(RUNNERS), default=["fop", "zop"])
    p.add_argument("--configs", nargs="+", choices=sorted(REWARD_CONFIGS), default=["balanced"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--controller", default="race_car")
    p.add_argument(
        "--variant", default="fop", help="SAC-FOP variant (fop | fopc | foa); ignored for zop."
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--max-parallel", type=int, default=None)
    p.add_argument("--sweep-root", type=Path, default=None)
    p.add_argument(
        "--dry-run", action="store_true", help="Print the jobs and exit without launching."
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra flags forwarded to every run_sac_*.py invocation "
        "(prefix with '--' to separate).",
    )
    args = p.parse_args(argv)
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.sweep_root or (REPO_ROOT / "output" / "sweeps" / timestamp)
    sweep_root = sweep_root.resolve()

    jobs = build_jobs(
        algos=args.algos,
        configs=args.configs,
        seeds=args.seeds,
        controller=args.controller,
        variant=args.variant,
        device=args.device,
        dtype=args.dtype,
        sweep_root=sweep_root,
        extra_flags=args.extra,
    )

    if args.dry_run:
        print(f"[dry-run] sweep_root: {sweep_root}")
        print(f"[dry-run] {len(jobs)} job(s):")
        for j in jobs:
            print(f"  - {j.algo} / {j.config} / seed={j.seed}")
            print(f"    out: {j.out}")
            print(f"    cmd: {' '.join(j.cmd)}")
        return 0

    sweep_root.mkdir(parents=True, exist_ok=True)
    max_parallel = args.max_parallel or min(len(jobs), os.cpu_count() or 1)

    run_sweep(jobs, max_parallel=max_parallel)
    write_manifest(sweep_root, jobs)

    failed = [j for j in jobs if j.returncode != 0]
    print()
    print(f"[sweep] summary: {len(jobs) - len(failed)}/{len(jobs)} ok, {len(failed)} failed")
    for j in jobs:
        tag = "ok" if j.returncode == 0 else f"FAIL rc={j.returncode}"
        print(f"  [{tag}] {j.algo}/{j.config}/seed{j.seed}  ->  {j.out}")
    print(f"[sweep] manifest: {sweep_root / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
