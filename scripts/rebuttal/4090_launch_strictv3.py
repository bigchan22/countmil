#!/usr/bin/env python
"""Fail-closed strict-v3 launcher for the 4x4090 server."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROTOCOL = "strictv3_train_holdout_val_official_test_no_hidden_val"


def _out(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _check_clean(expected_sha: str) -> None:
    head = _out(["git", "rev-parse", "HEAD"])
    if head != expected_sha:
        raise SystemExit(f"HEAD {head} != expected {expected_sha}")
    if _out(["git", "status", "--short"]):
        raise SystemExit("working tree is not clean")
    if _out(["git", "diff", "--cached", "--name-only"]):
        raise SystemExit("staged diff is not empty")


def _job_commands(expected_sha: str) -> dict[int, list[str]]:
    py = ".venv/bin/python"
    base = "PYTHONPATH=src"
    jobs: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
    for method in ["fsconv", "gaussian_amle", "mse"]:
        for seed in [0, 1, 2]:
            jobs[0].append(f"{base} {py} scripts/rebuttal/4090_train_strict_mnist_scalar.py --task digit_sum --method {method} --seed {seed} --device cuda:0")
            jobs[1].append(f"{base} {py} scripts/rebuttal/4090_train_strict_mnist_scalar.py --task signed_random --method {method} --seed {seed} --device cuda:1")
            jobs[2].append(f"{base} {py} scripts/rebuttal/4090_train_strict_mnist_scalar.py --task signed_cancellation --method {method} --seed {seed} --device cuda:2")
    for method in ["ce_kl", "fsconv_count", "official_llp_pvc"]:
        for seed in [0, 1, 2]:
            gpu = seed % 4
            jobs[gpu].append(f"{base} {py} scripts/rebuttal/4090_train_strict_cifar_fixed.py --method {method} --seed {seed} --device cuda:{gpu}")
    for gpu in jobs:
        jobs[gpu].insert(0, f"test $(git rev-parse HEAD) = {expected_sha}")
        jobs[gpu].insert(1, "test -z \"$(git status --short)\"")
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expected-sha", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    expected = args.expected_sha or _out(["git", "rev-parse", "HEAD"])
    _check_clean(expected)
    log_root = Path.home() / "fsconv_strictv3_queues"
    log_root.mkdir(parents=True, exist_ok=True)
    for gpu, commands in _job_commands(expected).items():
        script = log_root / f"gpu{gpu}_queue.sh"
        script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n")
        if args.dry_run:
            print(f"gpu {gpu}: bash {script}")
        else:
            subprocess.run(["tmux", "new-session", "-d", "-s", f"strictv3_gpu{gpu}", f"bash {script}"], check=True)


if __name__ == "__main__":
    main()
