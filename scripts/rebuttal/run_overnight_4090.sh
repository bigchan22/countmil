#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=src .venv/bin/python scripts/rebuttal/run_overnight_4090.py \
  --manifest configs/rebuttal/overnight_4090_manifest.yaml \
  --root results/rebuttal/overnight_4090
