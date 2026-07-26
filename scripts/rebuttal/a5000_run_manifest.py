#!/usr/bin/env python
"""Resumable single-GPU manifest scheduler for A5000 rebuttal experiments."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
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


def load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} must be JSON-compatible YAML: {exc}") from exc


def gpu_inventory() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name",
        "--format=csv,noheader",
    ]
    out = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in out.strip().splitlines():
        idx, name = [part.strip() for part in line.split(",", 1)]
        gpus.append({"index": int(idx), "name": name})
    return gpus


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def job_key(job: dict[str, Any], short_sha: str) -> str:
    parts = [job["task"], job["method"]]
    if "config" in job:
        parts.append(str(job["config"]))
    if "tau" in job:
        parts.append(f"tau{job['tau']}")
    parts.append(f"s{job['seed']}")
    parts.append(short_sha)
    return slug("_".join(parts))


def summary_path(output_root: Path, job: dict[str, Any]) -> Path | None:
    task = job["task"]
    method = job["method"]
    seed = int(job["seed"])
    if task == "svhn_sum":
        return output_root / "svhn_summaries" / f"svhn_sum_{method}_n10_train5000_s{seed}.json"
    if task == "dependence":
        tau = str(job["tau"]).replace(".", "p")
        return output_root / "dependence_summaries" / f"dependence_tau{tau}_{method}_s{seed}.json"
    return None


def completed(output_root: Path, status_dir: Path, job: dict[str, Any], key: str) -> bool:
    status_path = status_dir / f"{key}.json"
    summary = summary_path(output_root, job)
    if summary is None or not summary.exists():
        return False
    try:
        payload = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "COMPLETED":
        return False
    if not status_path.exists():
        return True
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return False
    return status.get("status") == "COMPLETED"


def run_job(
    job: dict[str, Any],
    gpu: dict[str, Any],
    output_root: Path,
    status_dir: Path,
    lock_dir: Path,
    log_dir: Path,
    short_sha: str,
) -> dict[str, Any]:
    key = job_key(job, short_sha)
    lock_path = lock_dir / f"{key}.lock"
    status_path = status_dir / f"{key}.json"
    started = utc_stamp()
    lock_path.write_text(json.dumps({"job": job, "gpu": gpu, "started_at": started}, indent=2, sort_keys=True))
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        job["script"],
        *job.get("args", []),
        "--device",
        "cuda",
        "--output-root",
        str(output_root),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "src")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    env["TORCH_HOME"] = env.get("TORCH_HOME", str(ROOT / ".torch_cache"))
    log_path = log_dir / f"{key}_{started}.log"
    status = {
        "status": "RUNNING",
        "job": job,
        "job_key": key,
        "git_commit": git_commit(),
        "gpu": gpu,
        "hostname": socket.gethostname(),
        "started_at": started,
        "command": cmd,
        "log_path": str(log_path),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    finished = utc_stamp()
    summary = summary_path(output_root, job)
    ok = proc.returncode == 0 and summary is not None and summary.exists()
    payload: dict[str, Any] = {}
    if summary is not None and summary.exists():
        try:
            payload = json.loads(summary.read_text())
        except json.JSONDecodeError:
            ok = False
    if payload.get("status") != "COMPLETED":
        ok = False
    status.update(
        {
            "status": "COMPLETED" if ok else "FAILED",
            "returncode": proc.returncode,
            "finished_at": finished,
            "summary_path": str(summary) if summary else None,
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return status


def main() -> None:
    args = list(sys.argv[1:])
    dry_run = False
    if "--dry-run" in args:
        dry_run = True
        args.remove("--dry-run")
    manifest_path = Path(args[0]) if args else ROOT / "configs/rebuttal/a5000_manifest.yaml"
    manifest = load_manifest(manifest_path)
    output_root = ROOT / manifest.get("output_root", "results/rebuttal/a5000_svhn_dependence")
    scheduler_root = output_root / "scheduler"
    status_dir = scheduler_root / "status"
    lock_dir = scheduler_root / "locks"
    log_dir = scheduler_root / "logs"
    for path in [status_dir, lock_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    head = git_commit()
    short_sha = head[:8]
    gpus = gpu_inventory()
    if len(gpus) < 4:
        raise SystemExit(f"expected four A5000 GPUs, found {gpus}")
    if not all("A5000" in gpu["name"] for gpu in gpus[:4]):
        raise SystemExit(f"expected A5000 GPU names, found {gpus}")
    (scheduler_root / "gpu_inventory.json").write_text(json.dumps(gpus, indent=2, sort_keys=True))
    (scheduler_root / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (scheduler_root / "git_commit.txt").write_text(head + "\n")

    queue = []
    for job in manifest["jobs"]:
        key = job_key(job, short_sha)
        if completed(output_root, status_dir, job, key):
            print(f"skip completed {key}", flush=True)
            continue
        queue.append(job)
    print(f"queued {len(queue)} jobs on {len(gpus[:4])} GPUs at {head}", flush=True)
    if dry_run:
        for job in queue:
            print(f"would run {job_key(job, short_sha)}", flush=True)
        return

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        active: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        pending = list(queue)
        free = list(gpus[:4])
        while pending or active:
            while pending and free:
                gpu = free.pop(0)
                job = pending.pop(0)
                future = executor.submit(run_job, job, gpu, output_root, status_dir, lock_dir, log_dir, short_sha)
                active[future] = gpu
            done, _ = concurrent.futures.wait(active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                gpu = active.pop(future)
                free.append(gpu)
                result = future.result()
                print(f"{result['status']} {result['job_key']} gpu={gpu['index']} log={result['log_path']}", flush=True)
                if result["status"] != "COMPLETED":
                    failures.append(result)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; inspect {status_dir}")


if __name__ == "__main__":
    main()
