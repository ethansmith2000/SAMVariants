#!/bin/bash
# SAMVariants sweep2: relative-rho + cross-optimizer arms, via gpu-claim
# (docs: /workspace/GPU_QUEUEING.md). Auto-resume makes restarts cheap.
cd /workspace/SAMVariants || exit 1
source /venv/main/bin/activate
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  sweep2-mm-rel1 sweep2-mm-rel2 sweep2-mm-rel4 sweep2-mm-rel8
  sweep2-am-rel1 sweep2-am-rel4
  sweep2-ma-rel1 sweep2-ma-rel4
  sweep2-adamw
  sweep2-mom-m-rel1
  sweep2-mm-reln0p25 sweep2-mm-reln1
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
echo "all sweep2 runs complete"
