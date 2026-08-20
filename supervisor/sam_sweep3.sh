#!/bin/bash
# SAMVariants sweep3: relative-rho + cross-optimizer arms, via gpu-claim
# (docs: /workspace/GPU_QUEUEING.md). Auto-resume makes restarts cheap.
cd /workspace/SAMVariants || exit 1
source /venv/main/bin/activate
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  sweep3-ma-rel0 sweep3-mom-m-rel4
)

pids=()
for name in "${CONFIGS[@]}"; do
  TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_${name}" \
  TRITON_CACHE_DIR="/tmp/triton_cache_${name}" \
  gpu-claim run --owner samvariants --job "${name}" --wait -- \
    python -u /workspace/SAMVariants/train_gpt.py \
    --override_json "sweep_configs/${name}.json" \
    > "slurm_logs/${name}.log" 2>&1 &
  pids+=($!)
  echo "queued ${name} (waiter pid $!)"
  sleep 5
done

wait "${pids[@]}"
echo "all sweep3 runs complete"
