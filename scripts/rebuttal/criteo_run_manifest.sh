#!/usr/bin/env bash
set -euo pipefail
EXPECTED_SHA="${1:-$(git rev-parse HEAD)}"
PYTHONPATH=src /home/chanhomin/countmil/.venv/bin/python scripts/rebuttal/criteo_run_manifest.py --expected-sha "$EXPECTED_SHA"

