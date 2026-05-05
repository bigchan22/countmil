#!/usr/bin/env python
"""Focused multiclass OVR count benchmark: CPU DP vs GPU FFT tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from countmil.aggregators import binary_count_dp, finite_support_convolution_fft_tree


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, device: torch.device, repeats: int, warmup: int) -> tuple[torch.Tensor, float, float]:
    out = None
    for _ in range(warmup):
        out = fn()
        _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    _sync(device)
    seconds = (time.perf_counter() - start) / repeats
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    assert out is not None
    return out.detach(), seconds, peak_mb


def _next_case(batch: int, bag_size: int, num_classes: int, device: torch.device, repeats: int, warmup: int) -> list[dict]:
    logits_cpu = torch.randn(batch, bag_size, num_classes, device="cpu")
    logits_gpu = logits_cpu.to(device)

    def cpu_dp() -> torch.Tensor:
        probs = torch.softmax(logits_cpu, dim=-1)
        ovr = probs.transpose(1, 2).reshape(batch * num_classes, bag_size)
        return binary_count_dp(ovr).probs.reshape(batch, num_classes, bag_size + 1)

    def gpu_fft() -> torch.Tensor:
        probs = torch.softmax(logits_gpu, dim=-1)
        ovr = probs.transpose(1, 2)
        atoms = torch.stack([1.0 - ovr, ovr], dim=-1)
        return finite_support_convolution_fft_tree(atoms, support_min=0).probs

    dp_out, dp_s, _ = _time(cpu_dp, torch.device("cpu"), repeats, warmup)
    fft_out, fft_s, fft_mem = _time(gpu_fft, device, repeats, warmup)
    err = (dp_out.to(fft_out.device) - fft_out).abs().max().item()
    return [
        {
            "method": "cpu_dp",
            "batch": batch,
            "bag_size": bag_size,
            "num_classes": num_classes,
            "device": "cpu",
            "seconds": dp_s,
            "peak_cuda_mem_mb": 0.0,
            "max_abs_err_vs_cpu_dp": 0.0,
        },
        {
            "method": "gpu_fft_tree",
            "batch": batch,
            "bag_size": bag_size,
            "num_classes": num_classes,
            "device": str(device),
            "seconds": fft_s,
            "peak_cuda_mem_mb": fft_mem,
            "max_abs_err_vs_cpu_dp": err,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/bench_multiclass_ovr_fft")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--bag-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = []
    for bag_size in args.bag_sizes:
        rows.extend(_next_case(args.batch, bag_size, args.num_classes, device, args.repeats, args.warmup))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"multiclass_ovr_fft_{device}_s{args.seed}.json"
    csv_path = out_dir / f"multiclass_ovr_fft_{device}_s{args.seed}.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['method']:>12} B={row['batch']:<4} N={row['bag_size']:<4} C={row['num_classes']:<3} "
            f"{row['seconds']:.6f}s mem={row['peak_cuda_mem_mb']:.1f}MB err={row['max_abs_err_vs_cpu_dp']}"
        )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
