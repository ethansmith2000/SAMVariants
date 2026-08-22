#!/bin/bash
# Kill our stalled trainers so their queue slot retries.
#
# A run whose log has not advanced in $STALL_MIN minutes while the process is
# still alive is deadlocked (this happened when the disk filled mid-write:
# trainer blocked in futex, dataloader workers polling, 38h of GPUs wasted).
# Killing is safe — runs auto-resume from their latest checkpoint.
#
# Only ever targets processes whose cwd is this repo: other projects on this
# box also run a file called train_gpt.py.
set -uo pipefail
REPO=/workspace/SAMVariants
STALL_MIN="${STALL_MIN:-25}"
# Geometries we have decided not to spend compute on. Perturbing along the raw
# momentum direction is MSAM's own choice and is not the cross-optimizer
# question this project is about (Ethan, 2026-08-22); kill those on sight so a
# queued retry cannot grab a GPU.
BLOCK_RE='sweep4-(gm|ga)-'

while true; do
  for p in $(pgrep -f "train_gpt.py --override" 2>/dev/null); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$REPO" ] || continue
    name=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE '[a-z0-9]+-[a-z0-9-]+\.json' | sed 's/\.json//')
    log="$REPO/slurm_logs/${name}.log"
    [ -f "$log" ] || continue
    if [[ "$name" =~ $BLOCK_RE ]]; then
      echo "[$(date +%F_%H:%M)] blocked geometry ${name} — killing pid $p"
      pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
      continue
    fi
    age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
    if [ "$age" -ge "$STALL_MIN" ]; then
      echo "[$(date +%F_%H:%M)] stalled ${name} (log ${age}m old) — killing pid $p"
      pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
    fi
  done
  sleep 300
done
