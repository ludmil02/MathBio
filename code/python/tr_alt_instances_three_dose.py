"""Trust region interior point method (at most TR_MAXIT iterations), followed by
the alternating method of Section 4.1 when the trust region method has not
stopped by its own tolerances, on the 36 instances of
test_barycenter_three_dose.py from the 49 starts of gs_three_dose.py.

A start is successful if its rho <= rho_ref (1 + 1e-7), rho_ref the smallest rho
of the 69 interior point runs per instance (test_barycenter_three_dose_results.txt).
Per start: trust region iterations and status, whether the alternating method
ran (and its sweeps), rho, its excess over rho_ref, the stationarity measure r
of the joint problem (t-r), and the time.

Instances run in parallel, one process each, from the easiest to the hardest
(the order of alt_instances_three_dose.py).  One flushed line per start; a
DONE line per instance; per-start results appended to OUT as instances end.

Usage:
    python3 -u tr_alt_instances_three_dose.py [--procs P] [--trmax N]
"""

import argparse
import time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize

import gs_three_dose as T
import alt_three_dose as A
import yogi_three_dose as Y
from test_barycenter_three_dose import set_instance, sigma_pair
from yogi_instances_three_dose import references
OUT = "tr_alt_instances_three_dose_results.txt"
YOGI_RES = "yogi_instances_three_dose_results.txt"


def label(inst):
    budget, r2, r3, sig = inst[:4]
    return "%s b2/b1=%.4g b3/b1=%.4g %s" % (budget, r2, r3, sig)


def difficulty():
    """Iterations of joint projected Yogi from the barycenter per instance, used
    to run the instances from the easiest to the hardest."""
    out = {}
    for line in open(YOGI_RES):
        if line.startswith("#"):
            continue
        f = line.split()
        out[(f[0], float(f[1]), float(f[2]), f[3].replace("_", " "))] = int(f[7])
    return out
OK = 1e-7
SWEEPS, YOGI2_TMAX, R_TOL = 1000, 20000, 1e-10   # SWEEPS only guarantees termination
TR_MAXIT = 500


def run_instance(inst):
    budget, r2, r3, sig, ref = inst
    days = 1.0 if budget == "weekly" else 7.0
    set_instance(days, r2, r3, sig)
    s2, s3 = sigma_pair(sig)
    b = T.b1 / days
    A.SIG = [T.S_.ravel(), s2.ravel(), s3.ravel()]
    A.B = [b, r2 * b, r3 * b]
    opts = dict(T.OPTS)
    opts["maxiter"] = TR_MAXIT
    rows = []
    for name, x0 in T.starts():
        t0 = time.time()
        res = minimize(T.value_grad, x0, jac=True, method="trust-constr",
                       constraints=[T.simplices(3)], bounds=[(0.0, None)] * (3 * T.N), options=opts)
        x = T.project(res.x)
        sw = 0
        if res.status == 0:                       # maxiter reached: finish with the alternating method
            try:
                x, sw, _, _, _, _ = A.alternate(x, SWEEPS, yogi_tol=1e-11,
                                                yogi_tmax=YOGI2_TMAX, r_tol=R_TOL)
            except AssertionError:
                sw = -1
        f, gr = T.value_grad(x)
        rho, r = np.exp(f), Y.stationarity(x, gr)
        el = time.time() - t0
        rows.append((name, res.nit, res.status, sw, rho, (rho - ref) / ref, r, el))
        print("  %-40s %-12s tr_it=%3d st=%d alt_sweeps=%2d rho=%.10f (%+.1e) r=%.0e %5.2fs"
              % ((label(inst), name) + rows[-1][1:]), flush=True)
    return inst, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--trmax", type=int, default=TR_MAXIT)
    a = ap.parse_args()
    TR_MAXIT = a.trmax
    dif = difficulty()
    insts = sorted(references(), key=lambda r: dif[tuple(r[:4])])
    t0 = time.time()
    open(OUT, "w").write("# budget r2 r3 sigma ref start tr_it tr_status alt_sweeps rho rel r seconds\n")
    nok = ntot = nalt = 0
    with Pool(a.procs) as pool:
        for inst, rows in pool.imap_unordered(run_instance, insts):
            budget, r2, r3, sig, ref = inst
            rel = np.array([rw[5] for rw in rows])
            alt = np.array([rw[3] != 0 for rw in rows])
            nok += (rel <= OK).sum(); ntot += len(rows); nalt += alt.sum()
            print("DONE %-40s ref %.10f | at ref: %2d/49 (from barycenter: %s) | best %+.1e worst %+.1e | "
                  "alt used %2d | tr_it max %3d | %5.1fs [%.0fs elapsed]"
                  % (label(inst), ref, (rel <= OK).sum(), "yes" if rel[0] <= OK else "no",
                     rel.min(), rel.max(), alt.sum(), max(rw[1] for rw in rows),
                     sum(rw[7] for rw in rows), time.time() - t0), flush=True)
            with open(OUT, "a") as fh:
                for rw in rows:
                    fh.write("%s %.6g %.6g %s %.12f %s %d %d %d %.12f %.3e %.3e %.2f\n"
                             % ((budget, r2, r3, sig.replace(" ", "_"), ref, rw[0].replace(" ", "_")) + rw[1:]))
    print("ALL %d starts: at ref %d, alternating method used %d; %.0f s"
          % (ntot, nok, nalt, time.time() - t0), flush=True)
    print("written", OUT)
