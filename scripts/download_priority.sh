#!/usr/bin/env bash
set -u

mkdir -p logs

echo "Starting priority dataset download at $(date)"
echo "Log: logs/download_priority.log"

# Important order:
# CIFAR-100 first because it is your main multi-count benchmark.
# Fashion-MNIST later because its mirror is currently slow.
DATASETS=(
  cifar100
  cifar10
  mnist
  fashion
  svhn
  stl10
)

for d in "${DATASETS[@]}"; do
  echo ""
  echo "=============================="
  echo "DATASET: $d"
  echo "START: $(date)"
  echo "=============================="

  # Timeout prevents one slow mirror from blocking the whole night.
  timeout 45m python scripts/download_one.py "$d"
  status=$?

  echo "END: $(date)"
  echo "STATUS for $d: $status"

  if [ "$status" -eq 124 ]; then
    echo "WARNING: $d timed out after 45 minutes. Skipping for now."
  elif [ "$status" -ne 0 ]; then
    echo "WARNING: $d failed with status $status. Skipping for now."
  else
    echo "SUCCESS: $d"
  fi
done

echo ""
echo "=============================="
echo "DISK USAGE"
echo "=============================="
du -sh data/* || true

echo ""
echo "Finished priority download at $(date)"
