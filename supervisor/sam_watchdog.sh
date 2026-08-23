#!/bin/bash
# Supervised wrapper for the stall watchdog (see watchdog.sh for the rationale).
# STALL_MIN is 45 here, not 25: the long runs validate every 2500 steps (~21min
# at ~7100 steps/hr), so a 25min threshold sits too close to normal quiet gaps.
cd /workspace/SAMVariants || exit 1
exec ./watchdog.sh
