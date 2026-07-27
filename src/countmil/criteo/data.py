"""Strict Criteo_x1 data verification, bag construction, and shard loading."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import torch
from torch.utils.data import Dataset

from countmil.criteo.constants import (
    ALL_FEATURE_COLS,
    BAG_SIZE,
    CATEGORICAL_COLS,
    EXPECTED_COLUMNS,
    EXPECTED_CSV_MD5,
    EXPECTED_ROWS,
    EXPECTED_ZIP_SHA256,
    GROUP_COLS,
    NUMERIC_COLS,
    PROTOCOL_TAG,
    SPLIT_BAGS,
)
from countmil.strict_protocol import canonical_hash


def file_hash(path: Path, algo: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def locate_csvs(root: Path) -> dict[str, Path]:
    found = {}
    for split in ["train", "valid", "test"]:
        matches = sorted(root.rglob(f"{split}.csv"))
        if not matches:
            raise FileNotFoundError(f"could not locate {split}.csv under {root}")
        found[split] = matches[0]
    return found


def audit_csv(path: Path, split: str) -> dict[str, Any]:
    md5 = file_hash(path, "md5")
    if md5 != EXPECTED_CSV_MD5[split]:
        raise RuntimeError(f"{split}.csv md5 mismatch: {md5}")
    scan = pl.scan_csv(path, infer_schema_length=1000)
    schema = scan.collect_schema()
    columns = list(schema.names())
    if columns != EXPECTED_COLUMNS:
        raise RuntimeError(f"{split}.csv columns mismatch: {columns[:5]} ...")
    rows = scan.select(pl.len().alias("n")).collect().item()
    if rows != EXPECTED_ROWS[split]:
        raise RuntimeError(f"{split}.csv row-count mismatch: {rows}")
    label_counts = scan.group_by("label").len().collect().sort("label")
    if set(label_counts["label"].to_list()) != {0, 1}:
        raise RuntimeError(f"{split}.csv labels are not binary")
    num_stats = scan.select(
        [pl.col(c).min().alias(f"{c}_min") for c in NUMERIC_COLS]
        + [pl.col(c).max().alias(f"{c}_max") for c in NUMERIC_COLS]
        + [pl.col(c).null_count().alias(f"{c}_nulls") for c in EXPECTED_COLUMNS]
    ).collect()
    cat_minmax = scan.select(
        [pl.col(c).min().alias(f"{c}_min") for c in CATEGORICAL_COLS]
        + [pl.col(c).max().alias(f"{c}_max") for c in CATEGORICAL_COLS]
    ).collect()
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "md5": md5,
        "rows": int(rows),
        "columns": columns,
        "schema": {name: str(dtype) for name, dtype in zip(schema.names(), schema.dtypes())},
        "label_counts": {str(k): int(v) for k, v in zip(label_counts["label"], label_counts["len"])},
        "numeric_and_null_stats": num_stats.to_dicts()[0],
        "categorical_minmax": cat_minmax.to_dicts()[0],
    }


def verify_download(data_root: Path, revision: str, command: str) -> dict[str, Any]:
    zip_path = data_root / "download" / "Criteo_x1.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    sha = file_hash(zip_path, "sha256")
    if sha != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f"Criteo_x1.zip sha256 mismatch: {sha}")
    extracted = data_root / "extracted"
    try:
        csvs = locate_csvs(extracted)
    except FileNotFoundError:
        extract_zip(zip_path, extracted)
        csvs = locate_csvs(extracted)
    return {
        "huggingface_repo": "reczoo/Criteo_x1",
        "resolved_revision": revision,
        "download_command": command,
        "zip_path": str(zip_path),
        "zip_size": zip_path.stat().st_size,
        "zip_sha256": sha,
        "csvs": {split: audit_csv(path, split) for split, path in csvs.items()},
        "preprocessing_caveat": "RecZoo Criteo_x1 public preprocessing/partitions are used; this is LLP-Bench-style, not an exact original LLP-Bench five-fold reproduction.",
        "disk_usage": os.popen(f"df -h {data_root}").read(),
    }


def _duckdb_columns() -> str:
    return ", ".join(["label", *NUMERIC_COLS, *CATEGORICAL_COLS])


def build_split_shard(
    *,
    csv_path: Path,
    out_dir: Path,
    split: str,
    seed: int,
    requested_bags: int,
    git_sha: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{split}_seed{seed}_selected.parquet"
    shard_path = out_dir / f"{split}_seed{seed}_shard.pt"
    manifest_path = out_dir / f"{split}_seed{seed}_manifest.json"
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    c4, c11 = GROUP_COLS
    sql = f"""
    COPY (
      WITH base AS (
        SELECT row_number() OVER () - 1 AS original_row_id, {_duckdb_columns()}
        FROM read_csv_auto('{csv_path}', header=true)
      ),
      ordered AS (
        SELECT *,
          row_number() OVER (
            PARTITION BY {c4}, {c11}
            ORDER BY md5('{PROTOCOL_TAG}|' || '{split}' || '|' || '{seed}' || '|' || {c4}::VARCHAR || '|' || {c11}::VARCHAR || '|' || original_row_id::VARCHAR)
          ) - 1 AS within_group_pos
        FROM base
      ),
      chunked AS (
        SELECT *, floor(within_group_pos / {BAG_SIZE})::BIGINT AS chunk_index
        FROM ordered
      ),
      full_chunks AS (
        SELECT {c4}, {c11}, chunk_index, count(*) AS n
        FROM chunked
        GROUP BY {c4}, {c11}, chunk_index
        HAVING count(*) = {BAG_SIZE}
      ),
      selected_chunks AS (
        SELECT *,
          row_number() OVER (
            ORDER BY md5('{PROTOCOL_TAG}|' || '{split}' || '|' || '{seed}' || '|' || {c4}::VARCHAR || '|' || {c11}::VARCHAR || '|' || chunk_index::VARCHAR)
          ) - 1 AS bag_id
        FROM full_chunks
        ORDER BY md5('{PROTOCOL_TAG}|' || '{split}' || '|' || '{seed}' || '|' || {c4}::VARCHAR || '|' || {c11}::VARCHAR || '|' || chunk_index::VARCHAR)
        LIMIT {requested_bags}
      ),
      selected_rows AS (
        SELECT sc.bag_id, ch.*,
          row_number() OVER (
            PARTITION BY sc.bag_id
            ORDER BY md5('{PROTOCOL_TAG}|' || '{split}' || '|' || '{seed}' || '|pos|' || ch.original_row_id::VARCHAR)
          ) - 1 AS position
        FROM chunked ch
        JOIN selected_chunks sc
          ON ch.{c4}=sc.{c4} AND ch.{c11}=sc.{c11} AND ch.chunk_index=sc.chunk_index
      )
      SELECT bag_id, position, original_row_id, {_duckdb_columns()}
      FROM selected_rows
      ORDER BY bag_id, position
    ) TO '{parquet_path}' (FORMAT PARQUET)
    """
    con.execute(sql)
    df = pl.read_parquet(parquet_path)
    if df.height % BAG_SIZE:
        raise RuntimeError(f"{split} selected row count is not divisible by bag size")
    num_bags = df.height // BAG_SIZE
    if num_bags == 0:
        raise RuntimeError(f"{split} produced zero full bags")
    if num_bags < requested_bags:
        shortfall = requested_bags - num_bags
    else:
        shortfall = 0
    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long).reshape(num_bags, BAG_SIZE)
    numeric = torch.tensor(df.select(NUMERIC_COLS).to_numpy(), dtype=torch.float32).reshape(num_bags, BAG_SIZE, len(NUMERIC_COLS))
    categorical = torch.tensor(df.select(CATEGORICAL_COLS).to_numpy(), dtype=torch.long).reshape(num_bags, BAG_SIZE, len(CATEGORICAL_COLS))
    row_ids = torch.tensor(df["original_row_id"].to_numpy(), dtype=torch.long).reshape(num_bags, BAG_SIZE)
    counts = labels.sum(dim=1).long()
    click_counts = counts.float()
    unique_groups = df.select(GROUP_COLS).unique().height
    candidate_count = con.execute(
        f"""
        WITH base AS (
          SELECT row_number() OVER () - 1 AS original_row_id, label, {c4}, {c11}
          FROM read_csv_auto('{csv_path}', header=true)
        ),
        ordered AS (
          SELECT *, row_number() OVER (PARTITION BY {c4}, {c11} ORDER BY md5('{PROTOCOL_TAG}|' || '{split}' || '|' || '{seed}' || '|' || {c4}::VARCHAR || '|' || {c11}::VARCHAR || '|' || original_row_id::VARCHAR)) - 1 AS within_group_pos
          FROM base
        ),
        chunked AS (
          SELECT floor(within_group_pos / {BAG_SIZE})::BIGINT AS chunk_index, {c4}, {c11}
          FROM ordered
        )
        SELECT count(*) FROM (
          SELECT {c4}, {c11}, chunk_index
          FROM chunked
          GROUP BY {c4}, {c11}, chunk_index
          HAVING count(*) = {BAG_SIZE}
        )
        """
    ).fetchone()[0]
    manifest_meta = {
        "protocol_tag": PROTOCOL_TAG,
        "split": split,
        "seed": int(seed),
        "group_cols": GROUP_COLS,
        "bag_size": BAG_SIZE,
        "requested_bags": int(requested_bags),
        "selected_bags": int(num_bags),
        "candidate_full_bags": int(candidate_count),
        "shortfall": int(shortfall),
        "unique_grouping_keys_selected": int(unique_groups),
        "retained_instances": int(df.height),
        "residual_instances_discarded_lower_bound": int(EXPECTED_ROWS[split] - int(candidate_count) * BAG_SIZE),
        "click_prevalence": float(labels.float().mean()),
        "click_count_mean": float(click_counts.mean()),
        "click_count_sample_std": float(click_counts.std(unbiased=True)) if num_bags > 1 else 0.0,
        "click_count_quantiles": {str(q): float(torch.quantile(click_counts, q).item()) for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]},
        "zero_click_bag_percentage": float((counts == 0).float().mean().item() * 100.0),
        "max_click_count": int(counts.max().item()),
        "source_csv": str(csv_path),
        "source_csv_md5": file_hash(csv_path, "md5"),
        "git_sha": git_sha,
    }
    manifest_hash = canonical_hash({"row_ids": row_ids, "counts": counts, "metadata": manifest_meta})
    input_hash = canonical_hash({"numeric": numeric, "categorical": categorical, "row_ids": row_ids, "metadata": manifest_meta})
    shard = {
        "numeric": numeric,
        "categorical": categorical,
        "counts": counts,
        "row_ids": row_ids,
        "metadata": manifest_meta | {"manifest_hash": manifest_hash, "input_shard_hash": input_hash},
    }
    if split == "test":
        shard["hidden_labels"] = labels
    else:
        shard["hidden_labels_external_only"] = labels
    torch.save(shard, shard_path)
    manifest_path.write_text(json.dumps(shard["metadata"], indent=2, sort_keys=True))
    return {
        **shard["metadata"],
        "parquet_path": str(parquet_path),
        "shard_path": str(shard_path),
        "shard_size": shard_path.stat().st_size,
        "shard_sha256": file_hash(shard_path, "sha256"),
        "parquet_size": parquet_path.stat().st_size,
        "parquet_sha256": file_hash(parquet_path, "sha256"),
    }


class CriteoBagDataset(Dataset[dict[str, torch.Tensor]]):
    """Bag dataset that can hide instance labels for train/validation."""

    def __init__(self, shard: dict[str, Any], *, expose_hidden_labels: bool) -> None:
        self.shard = shard
        self.expose_hidden_labels = bool(expose_hidden_labels)

    def __len__(self) -> int:
        return int(self.shard["counts"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "numeric": self.shard["numeric"][idx],
            "categorical": self.shard["categorical"][idx],
            "counts": self.shard["counts"][idx],
            "row_ids": self.shard["row_ids"][idx],
        }
        if self.expose_hidden_labels:
            item["hidden_labels"] = self.shard["hidden_labels"][idx]
        return item


def load_shard(path: Path, *, expose_hidden_labels: bool) -> CriteoBagDataset:
    shard = torch.load(path, map_location="cpu")
    if not expose_hidden_labels and "hidden_labels" in shard:
        raise RuntimeError("hidden labels are only allowed for final test evaluation")
    return CriteoBagDataset(shard, expose_hidden_labels=expose_hidden_labels)


def artifact_record(path: Path, *, producing_command: str, run_id: str, git_sha: str, required_to_retrain: bool, required_to_reconstruct_metrics: bool) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_hash(path, "sha256"),
        "producing_command": producing_command,
        "associated_run": run_id,
        "git_sha": git_sha,
        "required_to_retrain": bool(required_to_retrain),
        "required_to_reconstruct_metrics": bool(required_to_reconstruct_metrics),
    }
