#!/bin/bash
# SAMVariants OWT pilot: 7 runs via launch_local.sh. Runs auto-resume from
# their latest step_* checkpoint, so restarts (supervisor or container) are
# cheap. Exits 0 when all runs complete; autorestart=unexpected leaves it
# stopped after success but relaunches it if the box/container restarts
# mid-training.

cd /workspace/SAMVariants || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp
export GPUS="0,1,3,4,6"

exec ./launch_local.sh \
  sweep_configs/pilot-muon.json \
  sweep_configs/pilot-hybrid-rho1.json \
  sweep_configs/pilot-hybrid-rho0.json \
  sweep_configs/pilot-muon-nesterov.json \
  sweep_configs/pilot-hybrid-rho0p3.json \
  sweep_configs/pilot-hybrid-rho3.json \
  sweep_configs/pilot-hybrid-rhon1.json
