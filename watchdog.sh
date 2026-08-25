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

# Kills must leave a trail: supervisor routed the previous echoes to /dev/stdout
# and they were unrecoverable when we needed to explain 8 dead runs.
log_event() { echo "[$(date +%F_%H:%M)] $*" | tee -a "$REPO/slurm_logs/watchdog.log"; }

while true; do
  for p in $(pgrep -f "train_gpt.py --override" 2>/dev/null); do
    [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$REPO" ] || continue
    # A gpu-claim WAITER carries the whole training command in its own argv, so
    # the pgrep above matches it while it is merely sitting in the queue holding
    # no GPU. Killing those is what wiped 8 of 10 queued runs on 2026-08-23:
    # a waiter never writes to its log, so its mtime stays at creation and it
    # looks "stalled" the moment STALL_MIN elapses. Only ever target the real
    # trainer child, which has no gpu-claim in its argv.
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "gpu-claim" && continue
    name=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -oE '[a-z0-9]+-[a-z0-9-]+\.json' | sed 's/\.json//')
    log="$REPO/slurm_logs/${name}.log"
    [ -f "$log" ] || continue
    # Second guard: an empty log means the run has not started producing output
    # yet (queued, or still compiling), not that it stalled mid-training.
    [ -s "$log" ] || continue
    if [[ "$name" =~ $BLOCK_RE ]]; then
      log_event "blocked geometry ${name} — killing pid $p"
      pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
      continue
    fi
    age=$(( ( $(date +%s) - $(stat -c %Y "$log") ) / 60 ))
    if [ "$age" -ge "$STALL_MIN" ]; then
      log_event "stalled ${name} (log ${age}m old) — killing pid $p"
      pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null
    fi
  done
  sleep 300
done
