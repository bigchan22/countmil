"""Run metadata helpers for remote experiments."""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def make_run_dir(root: str | Path, experiment: str, method: str, dataset: str, seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname()
    path = Path(root) / experiment / method / dataset / f"{stamp}_{host}_s{seed}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_run_metadata(path: str | Path, config: Mapping[str, Any], seed: int) -> None:
    path = Path(path)
    metadata = {
        "config": dict(config),
        "seed": seed,
        "hostname": socket.gethostname(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
