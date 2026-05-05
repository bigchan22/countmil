#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:?usage: scripts/launcher/run_neurips_manifest.sh MANIFEST}"
GPUS="${GPUS:-0 1 2 3}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BAGS="${TEST_BAGS:-1000}"
TAG="${TAG:-$(basename "$(dirname "$MANIFEST")")}"
RUN_ROOT="${RUN_ROOT:-runs/neurips_pilot_${TAG}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/neurips_pilot_${TAG}}"
LOG_DIR="${LOG_DIR:-logs/neurips_pilot_${TAG}}"

read -r -a GPU_ARRAY <<< "$GPUS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  echo "GPUS is empty" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "$RESULTS_ROOT" "$LOG_DIR"
cp "$MANIFEST" "$RESULTS_ROOT/manifest.tsv"

echo "manifest=$MANIFEST"
echo "tag=$TAG"
echo "gpus=$GPUS"
echo "epochs=$EPOCHS"
echo "run_root=$RUN_ROOT"
echo "results_root=$RESULTS_ROOT"
echo "log_dir=$LOG_DIR"

idx=0
tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r name script config extra_args; do
  [ -n "$name" ] || continue
  gpu="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  log="$LOG_DIR/${name}.log"
  result_dir="$RESULTS_ROOT/${name}"
  run_dir="$RUN_ROOT/${name}"
  mkdir -p "$result_dir" "$run_dir"
  echo "Launching $name on GPU $gpu"
  (
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python "$script" \
      --config "$config" \
      --device cuda \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --test-bags "$TEST_BAGS" \
      --run-root "$run_dir" \
      --results-dir "$result_dir" \
      $extra_args
  ) > "$log" 2>&1 &
  idx=$((idx + 1))

  if [ $((idx % ${#GPU_ARRAY[@]})) -eq 0 ]; then
    wait
  fi
done
wait

PYTHONPATH=src .venv/bin/python scripts/aggregate_results.py \
  --input "$RESULTS_ROOT" \
  --output "$RESULTS_ROOT/aggregate.csv"
echo "All NeurIPS pilot runs finished."
