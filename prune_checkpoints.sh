#!/bin/bash
# Keep only the newest step_* checkpoint per run directory.
#
# train_gpt.py prunes its own checkpoints (keep_last_n_checkpoints), but
# already-running jobs hold the pre-patch code, so this reaps for them.
# Safe at any time: auto-resume only ever reads the newest checkpoint.
#
#   ./prune_checkpoints.sh [root] [keep]        # one-shot
#   ./prune_checkpoints.sh [root] [keep] loop   # every 5 min until stopped
set -uo pipefail
ROOT="${1:-/workspace/SAMVariants/model-output}"
KEEP="${2:-1}"
MODE="${3:-once}"

prune() {
  local freed=0
  while IFS= read -r rundir; do
    mapfile -t ckpts < <(find "$rundir" -maxdepth 1 -type d -name 'step_*' -printf '%f\n' 2>/dev/null \
      | sed 's/step_//' | sort -n | head -n -"$KEEP")
    for s in "${ckpts[@]:-}"; do
      [ -z "$s" ] && continue
      sz=$(du -sm "$rundir/step_$s" 2>/dev/null | cut -f1)
      rm -rf "$rundir/step_$s" && freed=$((freed + ${sz:-0}))
    done
  done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -type d 2>/dev/null)
  [ "$freed" -gt 0 ] && echo "[$(date +%H:%M:%S)] pruned ${freed}MB"
  return 0
}

if [ "$MODE" = loop ]; then
  while true; do prune; sleep 300; done
else
  prune
fi
