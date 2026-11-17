#!/bin/bash
# The weekly optimal simulation and its figures; see README.md.
#
#   ./run.sh [DOSES] [INITIAL]            run from t = 251
#   RESUME=1 ./run.sh [DOSES] [INITIAL]   continue an interrupted run (same DOSES, INITIAL)
#
# DOSES   the weekly doses (default input/doses_51x16x16_mock.xlsx)
# INITIAL the state at t = 251 (default input/state_t251_mock.csv)
# Results in results/<tag>/, tag = <INITIAL>__<DOSES> (file names without extension).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DOSES=$(realpath "${1:-$HERE/input/doses_51x16x16_mock.xlsx}")
INITIAL=$(realpath "${2:-$HERE/input/state_t251_mock.csv}")
d=$(basename "$DOSES"); i=$(basename "$INITIAL")
TAG="${i%.*}__${d%.*}"
RES="$HERE/results/$TAG"
mkdir -p "$RES"
LOG="$RES/simulate_optimal__$TAG.output"

cd "$HERE/../code/python"
if [ "${RESUME:-0}" = 1 ]; then
    python3 -u seir7_ode.py simulate optimal resume initial="$INITIAL" doses="$DOSES" out="$RES" 2>&1 | tee -a "$LOG"
else
    python3 -u seir7_ode.py simulate optimal initial="$INITIAL" doses="$DOSES" out="$RES" 2>&1 | tee "$LOG"
fi
python3 plot_weekly_optimal.py initial="$INITIAL" doses="$DOSES" out="$RES" "$RES/figures"
echo "results in $RES"
