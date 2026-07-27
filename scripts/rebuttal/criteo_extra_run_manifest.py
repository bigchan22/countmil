#!/usr/bin/env python
"""Fail-closed launcher for Criteo extra baseline runs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


METHOD_GPU = {"genbags": 0, "ot_llp": 1, "supervised_oracle": 2}
RESULT_ROOT = "results/rebuttal/4090_criteo_extra_baselines"


def _out(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _check(expected_sha: str) -> None:
    if _out(["git", "rev-parse", "HEAD"]) != expected_sha:
        raise SystemExit("HEAD does not match expected SHA")
    if _out(["git", "status", "--short"]):
        raise SystemExit("working tree is not clean")
    if _out(["git", "diff", "--cached", "--name-only"]):
        raise SystemExit("staged diff is not empty")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    _check(args.expected_sha)
    py = "/home/chanhomin/countmil/.venv/bin/python"
    log_root = Path.home() / "fsconv_criteo_extra_baseline_queues"
    log_root.mkdir(parents=True, exist_ok=True)

    group_script = log_root / "group_prior_cpu.sh"
    group_cmds = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {Path.cwd()}",
        f"test $(git rev-parse HEAD) = {args.expected_sha}",
        "test -z \"$(git status --short)\"",
    ]
    for seed in [0, 1, 2]:
        group_cmds.append(f"PYTHONPATH=src {py} scripts/rebuttal/criteo_group_prior.py --seed {seed} --result-root {RESULT_ROOT}")
    group_script.write_text("\n".join(group_cmds) + "\n")

    if args.dry_run:
        print(f"group_prior: bash {group_script}")
    else:
        subprocess.run(["tmux", "new-session", "-d", "-s", "criteo_group_prior", f"bash {group_script}"], check=True)

    for method, gpu in METHOD_GPU.items():
        commands = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {Path.cwd()}",
            f"test $(git rev-parse HEAD) = {args.expected_sha}",
            "test -z \"$(git status --short)\"",
        ]
        for seed in [0, 1, 2]:
            commands.append(f"CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH=src {py} scripts/rebuttal/criteo_train.py --method {method} --seed {seed} --device cuda:0 --result-root {RESULT_ROOT}")
        script = log_root / f"{method}_gpu{gpu}.sh"
        script.write_text("\n".join(commands) + "\n")
        if args.dry_run:
            print(f"{method}: bash {script}")
        else:
            subprocess.run(["tmux", "new-session", "-d", "-s", f"criteo_{method}", f"bash {script}"], check=True)


if __name__ == "__main__":
    main()
