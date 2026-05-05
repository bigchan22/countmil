#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 OLD_PID[,OLD_PID...] RESULTS_ROOT CURRENT_NAME[,CURRENT_NAME...] REMAINING_MANIFEST" >&2
  exit 1
fi

OLD_PIDS_CSV="$1"
RESULTS_ROOT="$2"
CURRENT_NAMES_CSV="$3"
REMAINING_MANIFEST="$4"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

IFS=',' read -r -a OLD_PIDS <<< "$OLD_PIDS_CSV"
IFS=',' read -r -a CURRENT_NAMES <<< "$CURRENT_NAMES_CSV"

echo "Waiting for current jobs to finish:"
printf '  %s\n' "${CURRENT_NAMES[@]}"

while true; do
  missing=0
  for name in "${CURRENT_NAMES[@]}"; do
    if ! find "$RESULTS_ROOT/$name" -maxdepth 1 -name '*.json' -type f | grep -q .; then
      missing=$((missing + 1))
    fi
  done
  if [ "$missing" -eq 0 ]; then
    break
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for ${missing} current summaries"
  sleep "$SLEEP_SECONDS"
done

echo "Current jobs finished. Terminating paused old launcher pids: ${OLD_PIDS[*]}"
for pid in "${OLD_PIDS[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 5
for pid in "${OLD_PIDS[@]}"; do
  kill -KILL "$pid" 2>/dev/null || true
done

echo "Starting remaining manifest: ${REMAINING_MANIFEST}"
scripts/launcher/run_neurips_manifest.sh "$REMAINING_MANIFEST"
