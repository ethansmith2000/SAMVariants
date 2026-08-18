#!/bin/bash
# SAMVariants OWT pilot queue — every run acquires its GPU through the shared
# gpu-claim protocol (docs: /workspace/GPU_QUEUEING.md). Each config launches
# a gpu-claim waiter; runs start whenever a GPU frees up and auto-resume from
# their latest step_* checkpoint, so restarts (supervisor or container) are
# cheap. Exits when all runs have finished.
#
# Never kill "train_gpt.py" by name on this box — other projects use the same
# filename. Target /workspace/SAMVariants paths or the job's pid file.

cd /workspace/SAMVariants || exit 1
source /venv/main/bin/activate

export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  pilot-muon
  pilot-hybrid-rho1
  pilot-hybrid-rho0
  pilot-muon-nesterov
  pilot-hybrid-rho0p3
  pilot-hybrid-rho3
  pilot-hybrid-rhon1
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
echo "all pilot runs complete"
