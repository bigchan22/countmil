#!/usr/bin/env python
"""Verify and document the public RecZoo Criteo_x1 package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from countmil.criteo.data import verify_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/chanhomin/datasets/criteo_x1")
    p.add_argument("--revision", required=True)
    p.add_argument("--download-command", required=True)
    p.add_argument("--out", default="docs/rebuttal/criteo_DATA_ACQUISITION_AUDIT.md")
    p.add_argument("--json-out", default="artifacts/rebuttal/criteo_data_acquisition_audit.json")
    args = p.parse_args()
    audit = verify_download(Path(args.data_root), args.revision, args.download_command)
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(audit, indent=2, sort_keys=True))
    lines = [
        "# Criteo_x1 Data Acquisition Audit",
        "",
        f"- Hugging Face repo: `{audit['huggingface_repo']}`",
        f"- Resolved revision: `{audit['resolved_revision']}`",
        f"- ZIP: `{audit['zip_path']}`",
        f"- ZIP size: `{audit['zip_size']}`",
        f"- ZIP SHA256: `{audit['zip_sha256']}`",
        f"- Download command: `{audit['download_command']}`",
        f"- Caveat: {audit['preprocessing_caveat']}",
        "",
        "## CSV Verification",
        "",
        "| Split | Rows | MD5 | Path |",
        "|---|---:|---|---|",
    ]
    for split, info in audit["csvs"].items():
        lines.append(f"| {split} | {info['rows']} | `{info['md5']}` | `{info['path']}` |")
    lines += ["", "## Disk", "", "```text", audit["disk_usage"].strip(), "```"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

