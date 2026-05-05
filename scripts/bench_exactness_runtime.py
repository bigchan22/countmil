#!/usr/bin/env python
"""Priority-0 exactness/runtime benchmark entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import torch

from countmil.aggregators import binary_count_dp, finite_support_convolution, grouped_signed_binary_convolution


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_call(
    fn,
    device: torch.device,
    backward: bool,
    repeats: int,
    warmup: int,
) -> tuple[torch.Tensor, float, float]:
    out = None
    for _ in range(warmup):
        out = fn()
        if backward:
            out.square().mean().backward()
        _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
        if backward:
            out.square().mean().backward()
    _sync(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    assert out is not None
    return out.detach(), (time.perf_counter() - start) / repeats, peak_mb


def _binary_case(
    batch: int,
    n_atoms: int,
    device: torch.device,
    backward: bool,
    repeats: int,
    warmup: int,
) -> list[dict]:
    with torch.no_grad():
        compare_logits = torch.randn(batch, n_atoms, device=device)
        compare_probs = torch.sigmoid(compare_logits)
        compare_atoms = torch.stack([1.0 - compare_probs, compare_probs], dim=-1)
        dp_compare = binary_count_dp(compare_probs).probs
        conv_compare = finite_support_convolution(compare_atoms).probs
        max_abs_err = (dp_compare - conv_compare).abs().max().item()

    dp_base = torch.randn(batch, n_atoms, device=device)

    def dp_fn() -> torch.Tensor:
        probs = torch.sigmoid(dp_base.detach().requires_grad_(backward))
        return binary_count_dp(probs).probs

    _, dp_s, dp_mem = _time_call(dp_fn, device, backward, repeats, warmup)

    conv_base = torch.randn(batch, n_atoms, device=device)

    def conv_fn() -> torch.Tensor:
        probs = torch.sigmoid(conv_base.detach().requires_grad_(backward))
        atoms = torch.stack([1.0 - probs, probs], dim=-1)
        return finite_support_convolution(atoms).probs

    _, conv_s, conv_mem = _time_call(conv_fn, device, backward, repeats, warmup)

    return [
        {
            "atom_type": "binary",
            "method": "dp",
            "batch": batch,
            "n_atoms": n_atoms,
            "support_width": n_atoms + 1,
            "device": str(device),
            "backward": backward,
            "repeats": repeats,
            "warmup": warmup,
            "seconds": dp_s,
            "peak_cuda_mem_mb": dp_mem,
            "max_abs_err_vs_dp": 0.0,
        },
        {
            "atom_type": "binary",
            "method": "conv",
            "batch": batch,
            "n_atoms": n_atoms,
            "support_width": n_atoms + 1,
            "device": str(device),
            "backward": backward,
            "repeats": repeats,
            "warmup": warmup,
            "seconds": conv_s,
            "peak_cuda_mem_mb": conv_mem,
            "max_abs_err_vs_dp": max_abs_err,
        },
    ]


def _ordinal_case(
    batch: int,
    n_atoms: int,
    width: int,
    device: torch.device,
    backward: bool,
    repeats: int,
    warmup: int,
) -> list[dict]:
    base = torch.randn(batch, n_atoms, width, device=device)

    def conv_fn() -> torch.Tensor:
        atoms = torch.softmax(base.detach().requires_grad_(backward), dim=-1)
        return finite_support_convolution(atoms).probs

    conv, conv_s, conv_mem = _time_call(conv_fn, device, backward, repeats, warmup)
    return [
        {
            "atom_type": f"ordinal_{width}",
            "method": "conv",
            "batch": batch,
            "n_atoms": n_atoms,
            "support_width": conv.shape[-1],
            "device": str(device),
            "backward": backward,
            "repeats": repeats,
            "warmup": warmup,
            "seconds": conv_s,
            "peak_cuda_mem_mb": conv_mem,
            "max_abs_err_vs_dp": None,
        }
    ]


def _signed_grouped_case(
    batch: int,
    n_atoms: int,
    num_groups: int,
    device: torch.device,
    backward: bool,
    repeats: int,
    warmup: int,
) -> list[dict]:
    base = torch.randn(batch, n_atoms, device=device)
    signs = torch.randint(0, 2, (batch, n_atoms), device=device).mul(2).sub(1)
    group_ids = torch.arange(n_atoms, device=device).remainder(num_groups).expand(batch, n_atoms)

    def conv_fn() -> torch.Tensor:
        probs = torch.sigmoid(base.detach().requires_grad_(backward))
        return grouped_signed_binary_convolution(probs, signs, group_ids, num_groups=num_groups).probs

    conv, conv_s, conv_mem = _time_call(conv_fn, device, backward, repeats, warmup)
    return [
        {
            "atom_type": "signed_binary_grouped",
            "method": "grouped_conv",
            "batch": batch,
            "n_atoms": n_atoms,
            "num_groups": num_groups,
            "support_width": conv.shape[-1],
            "device": str(device),
            "backward": backward,
            "repeats": repeats,
            "warmup": warmup,
            "seconds": conv_s,
            "peak_cuda_mem_mb": conv_mem,
            "max_abs_err_vs_dp": None,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/bench")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--bag-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--ordinal-width", type=int, default=10)
    parser.add_argument("--signed-groups", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    rows = []
    for n_atoms in args.bag_sizes:
        rows.extend(_binary_case(args.batch, n_atoms, device, args.backward, args.repeats, args.warmup))
        rows.extend(_ordinal_case(args.batch, n_atoms, args.ordinal_width, device, args.backward, args.repeats, args.warmup))
        rows.extend(
            _signed_grouped_case(
                args.batch,
                n_atoms,
                args.signed_groups,
                device,
                args.backward,
                args.repeats,
                args.warmup,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"exactness_runtime_{device}_s{args.seed}.json"
    csv_path = out_dir / f"exactness_runtime_{device}_s{args.seed}.csv"
    json_path.write_text(json.dumps(rows, indent=2))

    with csv_path.open("w", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['atom_type']:>10} {row['method']:>4} "
            f"N={row['n_atoms']:<4} {row['seconds']:.5f}s "
            f"mem={row['peak_cuda_mem_mb']:.1f}MB "
            f"err={row['max_abs_err_vs_dp']}"
        )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
