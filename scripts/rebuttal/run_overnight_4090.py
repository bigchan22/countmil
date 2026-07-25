#!/usr/bin/env python
"""Sequential resumable runner for rebuttal jobs on one RTX 4090."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from countmil.training.run import git_commit


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/rebuttal/overnight_4090_manifest.yaml")
    parser.add_argument("--root", default="results/rebuttal/overnight_4090")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.root)
    status_root = root / "job_status"
    log_root = root / "logs"
    status_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    gpu_index = int(manifest["gpu_index"])
    commit = git_commit()[:8]

    for job in manifest["jobs"]:
        name = _safe_name(job["name"])
        done = status_root / f"{name}.COMPLETED"
        running = status_root / f"{name}.RUNNING"
        failed = status_root / f"{name}.FAILED"
        if done.exists():
            print(f"skip completed {name}")
            continue
        run_id = f"{name}_{commit}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        running.write_text(json.dumps({"job": job, "run_id": run_id, "started": time.time()}, indent=2, sort_keys=True))
        stdout_path = log_root / f"{run_id}.stdout.log"
        stderr_path = log_root / f"{run_id}.stderr.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env.setdefault("PYTHONPATH", "src")
        print(f"run {name}")
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            proc = subprocess.run(job["command"], shell=True, cwd=Path.cwd(), env=env, stdout=out, stderr=err)
        payload = {
            "job": job,
            "run_id": run_id,
            "returncode": proc.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "finished": time.time(),
        }
        if proc.returncode == 0:
            done.write_text(json.dumps(payload, indent=2, sort_keys=True))
            running.unlink(missing_ok=True)
        else:
            failed.write_text(json.dumps(payload, indent=2, sort_keys=True))
            running.unlink(missing_ok=True)
            print(f"failed {name}; continuing")


if __name__ == "__main__":
    main()
