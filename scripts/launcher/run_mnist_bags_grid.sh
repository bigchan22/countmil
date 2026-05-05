#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs/mnist_bags/shukla_grid}"
GPUS="${GPUS:-0 1 2 3}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BAGS="${TEST_BAGS:-1000}"
RUN_ROOT="${RUN_ROOT:-runs}"
RESULTS_DIR="${RESULTS_DIR:-results/mnist_bags}"

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "No configs found in $CONFIG_DIR" >&2
  exit 1
fi

read -r -a GPU_ARRAY <<< "$GPUS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  echo "GPUS is empty" >&2
  exit 1
fi

mkdir -p logs/mnist_bags

idx=0
for cfg in "${CONFIGS[@]}"; do
  gpu="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  name="$(basename "$cfg" .yaml)"
  log="logs/mnist_bags/${name}.log"
  echo "Launching $name on GPU $gpu"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src .venv/bin/python scripts/train_mnist_bags.py \
      --config "$cfg" \
      --device cuda \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --test-bags "$TEST_BAGS" \
      --run-root "$RUN_ROOT" \
      --results-dir "$RESULTS_DIR"
  ) > "$log" 2>&1 &
  idx=$((idx + 1))

  if [ $((idx % ${#GPU_ARRAY[@]})) -eq 0 ]; then
    wait
  fi
done
wait
echo "All MNIST-Bags runs finished."

