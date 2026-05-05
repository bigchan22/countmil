#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: scripts/launcher/run_local.sh CONFIG [SEED]}"
SEED="${2:-0}"

echo "config=${CONFIG}"
echo "seed=${SEED}"
echo "host=$(hostname)"

