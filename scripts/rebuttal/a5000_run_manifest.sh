#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH="${PYTHONPATH:-src}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

exec .venv/bin/python scripts/rebuttal/a5000_run_manifest.py "${1:-configs/rebuttal/a5000_manifest.yaml}"
