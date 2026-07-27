#!/usr/bin/env python
"""Fail-closed sequential launcher for masked strict-v3 SVHN reruns."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_source_clean() -> bool:
    tracked = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True).strip()
    return not tracked and not staged


def gpu_inventory() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], text=True)
    gpus = []
    for line in out.strip().splitlines():
        idx, name = [part.strip() for part in line.split(",", 1)]
        gpus.append({"index": int(idx), "name": name})
    return gpus


def summary_path(output_root: Path, method: str, seed: int) -> Path:
    return output_root / "svhn_summaries" / f"svhn_sum_{method}_strictv3_masked_n10_train5000_s{seed}.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="configs/rebuttal/4090_svhn_masked_strictv3_manifest.json")
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    head = git_commit()
    if head != args.expected_sha:
        raise SystemExit(f"HEAD {head} does not match expected source SHA {args.expected_sha}")
    if not git_source_clean():
        raise SystemExit("tracked or staged source tree is not clean")
    gpus = gpu_inventory()
    if not gpus:
        raise SystemExit("nvidia-smi did not report a GPU")
    selected = next((g for g in gpus if g["index"] == args.gpu_index), None)
    if selected is None:
        raise SystemExit(f"GPU {args.gpu_index} not found in {gpus}")
    if "4090" not in selected["name"]:
        raise SystemExit(f"selected GPU is not an RTX 4090: {selected}")

    manifest = json.loads((ROOT / args.manifest).read_text())
    output_root = ROOT / manifest["output_root"]
    scheduler_root = output_root / "scheduler"
    status_dir = scheduler_root / "status"
    log_dir = scheduler_root / "logs"
    for path in [status_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)
    (scheduler_root / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (scheduler_root / "git_commit.txt").write_text(head + "\n")
    (scheduler_root / "gpu_inventory.json").write_text(json.dumps(gpus, indent=2, sort_keys=True))

    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path("/home/chanhomin/countmil/.venv/bin/python")
    commands = []
    for job in manifest["jobs"]:
        method = job["method"]
        seed = int(job["seed"])
        out = summary_path(output_root, method, seed)
        if out.exists():
            raise SystemExit(f"refusing to reuse existing summary: {out}")
        cmd = [
            str(py),
            "scripts/rebuttal/a5000_train_svhn_scalar.py",
            "--method",
            method,
            "--seed",
            str(seed),
            "--dataset-root",
            manifest["dataset_root"],
            "--output-root",
            manifest["output_root"],
            "--device",
            "cuda",
            *manifest["common_args"],
        ]
        commands.append((method, seed, cmd))

    if args.dry_run:
        for _, _, cmd in commands:
            print(" ".join(cmd))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    env["TORCH_HOME"] = env.get("TORCH_HOME", str(ROOT / ".torch_cache"))
    for method, seed, cmd in commands:
        if git_commit() != head or not git_source_clean():
            raise SystemExit("tracked or staged source tree changed during queue")
        stamp = utc_stamp()
        key = f"svhn_masked_{method}_s{seed}_{head[:8]}_{stamp}"
        status_path = status_dir / f"{key}.json"
        log_path = log_dir / f"{key}.log"
        status = {
            "status": "RUNNING",
            "method": method,
            "seed": seed,
            "git_commit": head,
            "gpu": selected,
            "hostname": socket.gethostname(),
            "started_at": stamp,
            "command": cmd,
            "log_path": str(log_path),
        }
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        summary = summary_path(output_root, method, seed)
        ok = proc.returncode == 0 and summary.exists()
        if ok:
            payload = json.loads(summary.read_text())
            ok = payload.get("status") == "COMPLETED" and payload.get("metadata", {}).get("git_commit") == head
        status.update(
            {
                "status": "COMPLETED" if ok else "FAILED",
                "returncode": proc.returncode,
                "finished_at": utc_stamp(),
                "summary_path": str(summary),
            }
        )
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        print(f"{status['status']} {method} seed={seed} log={log_path}", flush=True)
        if not ok:
            raise SystemExit(f"job failed: {method} seed {seed}; inspect {log_path}")


if __name__ == "__main__":
    main()
