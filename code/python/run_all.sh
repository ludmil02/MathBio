#!/bin/bash
# Runs every script whose output enters notes/min_spectral_radius_vaccination.tex,
# from code/python/, in dependency order.  Each script's output goes to
# intermediate/<name>.log; intermediate/run_all.log lists exit codes and times.
#
# Two groups run side by side: test_barycenter_three_dose.py (about 90 minutes)
# and the single-dose and Gauss-Seidel scripts.  The Yogi scripts run after
# both, alone, since their times enter Table 4.1.
#
# Usage, from code/python/:
#     ./run_all.sh

cd "$(dirname "$(readlink -f "$0")")" || exit 1
export PYTHONDONTWRITEBYTECODE=1
LOG=intermediate/run_all.log
mkdir -p intermediate
: > "$LOG"

run() {         # run LOGNAME SCRIPT [ARGS...]
    local name=$1; shift
    local t0=$(date +%s)
    python3 -u "$@" > "intermediate/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%H:%M:%S') $name rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
}

echo "$(date '+%F %T') start" | tee -a "$LOG"

run test_barycenter_three_dose test_barycenter_three_dose.py &

run zf_jg                 zf_jg.py
run smoke_zf_jg           smoke_zf_jg.py           # needs wolframscript
run check_block2          check_block2.py
run gp_single_dose        gp_single_dose.py
run reduced_test          reduced_test.py          # needs wolframscript
run plot_chi1_single_dose plot_chi1_single_dose.py
run gs_three_dose         gs_three_dose.py
run check_gs_endpoints    check_gs_endpoints.py
wait

run yogi_three_dose            yogi_three_dose.py
run yogi_three_dose_tmax2e4    yogi_three_dose.py --tmax 20000 --out yogi_three_dose_tmax2e4_results.txt
run yogi_three_dose_eta1e-3    yogi_three_dose.py --eta 1e-3 --tmax 5000 --only barycenter
run yogi_instances_three_dose  yogi_instances_three_dose.py
run plot_yogi_three_dose       plot_yogi_three_dose.py
run make_figures               make_figures.py

echo "$(date '+%F %T') done" | tee -a "$LOG"
