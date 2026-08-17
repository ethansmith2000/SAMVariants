#!/bin/bash
# =============================================================================
# Local (non-Slurm) sweep launcher: round-robins config JSONs across GPUs.
#
# Usage:
#   ./launch_local.sh cfg1.json cfg2.json ...       # explicit configs
#   ./launch_local.sh sweep_configs/*.json          # glob
#
# One training process per config, assigned to GPUs round-robin; waits when all
# GPUs are busy. Logs to slurm_logs/<config-stem>.log, PIDs recorded so runs
# can be killed with: kill $(cat slurm_logs/<stem>.pid)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
# Restrict to specific GPUs with e.g. GPUS="1,3,7" ./launch_local.sh ...
# (other jobs may occupy some GPUs — check nvidia-smi first)
if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "${GPUS}"
else
  mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NUM_GPUS=${#GPU_LIST[@]}
mkdir -p "${LOG_DIR}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.json> [config.json ...]" >&2
  exit 1
fi

source /venv/main/bin/activate
export TMPDIR="${TMPDIR:-/tmp}"

declare -a gpu_pid
i=0
for cfg in "$@"; do
  slot=$(( i % NUM_GPUS ))
  gpu=${GPU_LIST[$slot]}
  # wait for this slot's previous job to finish
  if [[ -n "${gpu_pid[$slot]:-}" ]]; then
    wait "${gpu_pid[$slot]}" || true
  fi
  stem="$(basename "${cfg}" .json)"
  echo "[gpu ${gpu}] ${stem}"
  CUDA_VISIBLE_DEVICES=${gpu} nohup python "${SCRIPT_DIR}/train_gpt.py" \
    --override_json "${cfg}" > "${LOG_DIR}/${stem}.log" 2>&1 &
  gpu_pid[$gpu]=$!
  echo "${gpu_pid[$gpu]}" > "${LOG_DIR}/${stem}.pid"
  i=$(( i + 1 ))
done

echo "all launched; waiting for completion..."
wait
echo "done"
