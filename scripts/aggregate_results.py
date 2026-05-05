#!/usr/bin/env python
"""Aggregate CountMIL JSON summaries or JSONL metrics into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}{k}.", v, out)
    else:
        out[prefix[:-1]] = value


def _read_summary(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    row: dict[str, Any] = {"source": str(path)}
    _flatten("", data, row)
    return [row]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = {"source": str(path), "line": line_no}
        _flatten("", json.loads(line), row)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["results"])
    parser.add_argument("--output", default="results/aggregate.csv")
    parser.add_argument("--kind", choices=["auto", "summary", "jsonl"], default="auto")
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.input:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.json")))
            paths.extend(sorted(p.rglob("*.jsonl")))
        elif p.exists():
            paths.append(p)

    rows: list[dict[str, Any]] = []
    for path in paths:
        if args.kind == "jsonl" or (args.kind == "auto" and path.suffix == ".jsonl"):
            rows.extend(_read_jsonl(path))
        elif args.kind == "summary" or path.suffix == ".json":
            rows.extend(_read_summary(path))

    if not rows:
        raise SystemExit("No rows found.")

    fieldnames = sorted({key for row in rows for key in row})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

