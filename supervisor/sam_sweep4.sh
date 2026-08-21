#!/bin/bash
# SAMVariants sweep4: relative-rho + cross-optimizer arms, via gpu-claim
# (docs: /workspace/GPU_QUEUEING.md). Auto-resume makes restarts cheap.
cd /workspace/SAMVariants || exit 1
source /venv/main/bin/activate
export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  sweep4-am-rel0p25
  sweep4-ma-rel0p25
  sweep4-aa-rel0p5
  sweep4-am-rel0p5
  sweep4-ga-rel0p5
  sweep4-gm-rel0p5
  sweep4-ma-rel0p5
  sweep4-mm-rel0p5
  sweep4-aa-rel1
  sweep4-ga-rel1
  sweep4-aa-rel2
  sweep4-am-rel2
  sweep4-ga-rel2
  sweep4-gm-rel2
  sweep4-ma-rel2
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
echo "all sweep4 runs complete"
