#!/bin/bash
# SAMVariants sweep5 queue.
#
# Each config acquires a GPU via the shared gpu-claim protocol
# (/workspace/GPU_QUEUEING.md). Robustness added after a 38-GPU-hour silent
# stall: runs that already finished are skipped, and each config gets bounded
# retries (runs auto-resume from their latest checkpoint, so a retry continues
# rather than restarting).
#
# Never kill "train_gpt.py" by name on this box — other projects share the
# filename. Match on /proc/<pid>/cwd.

cd /workspace/SAMVariants || exit 1
source /venv/main/bin/activate
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  sweep5-am-rel4-s2 sweep5-muon-s2
  sweep5-am-rel4-s3 sweep5-muon-s3
  sweep5-mm-rel4-s2
)
MAX_ATTEMPTS=3

run_one() {
  local name="$1"
  if grep -q "Saving model to" "slurm_logs/${name}.log" 2>/dev/null; then
    echo "[skip] ${name} already complete"
    return 0
  fi
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "[start] ${name} attempt ${attempt}"
    TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_${name}" \
    TRITON_CACHE_DIR="/tmp/triton_cache_${name}" \
    gpu-claim run --owner samvariants --job "${name}" --wait -- \
      python -u /workspace/SAMVariants/train_gpt.py \
      --override_json "sweep_configs/${name}.json" \
      >> "slurm_logs/${name}.log" 2>&1
    if grep -q "Saving model to" "slurm_logs/${name}.log" 2>/dev/null; then
      echo "[done] ${name}"
      return 0
    fi
    echo "[retry] ${name} exited without completing"
    sleep 30
  done
  echo "[fail] ${name} after ${MAX_ATTEMPTS} attempts"
}

pids=()
for name in "${CONFIGS[@]}"; do
  run_one "$name" &
  pids+=($!)
  sleep 5
done
wait "${pids[@]}"
echo "sweep5 queue drained"
