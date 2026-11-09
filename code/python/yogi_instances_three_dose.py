"""Projected Yogi (Algorithm T5.1) from the barycenter on the 36 instances of Section T4.

The instances are those of test_barycenter_three_dose.py (set_instance is
imported from it), and the reference rho_ref of each instance, the smallest rho
of the joint method over 69 starts, is read from its results file
test_barycenter_three_dose_results.txt.  Reported per instance: rho of projected
Yogi from the barycenter, its relative excess over rho_ref, the iterations, the
stationarity measure r (T5.4) and the KKT check of Algorithm T2.

Writes yogi_instances_three_dose_results.txt.

Usage:
    python3 -u yogi_instances_three_dose.py [--eta E] [--tol T] [--tmax N]
"""

import argparse
import time
import numpy as np

import gs_three_dose as T
import yogi_three_dose as Y
from test_barycenter_three_dose import set_instance

REF = "test_barycenter_three_dose_results.txt"
OUT = "yogi_instances_three_dose_results.txt"
OK = 1e-7             # successful if rho <= rho_ref * (1 + OK), as in Section T4


def references():
    out = []
    for line in open(REF):
        if line.startswith("#"):
            continue
        f = line.split()
        out.append((f[0], float(f[1]), float(f[2]), f[3].replace("_", " "), float(f[4])))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--tmax", type=int, default=200000)
    a = ap.parse_args()
    print("eta=%g tol=%g tmax=%d" % (a.eta, a.tol, a.tmax), flush=True)
    bary = np.full(3 * T.N, 1.0 / T.n)
    rows = []
    t0 = time.time()
    for budget, r2, r3, sig, ref in references():
        set_instance(1.0 if budget == "weekly" else 7.0, r2, r3, sig)
        t = time.time()
        x, it, r = Y.yogi(bary, a.eta, 0.9, 0.999, 1e-8, a.tol, a.tmax)
        rho = np.exp(T.value_grad(x)[0])
        sp, ex = T.kkt(x)
        row = (budget, r2, r3, sig, ref, rho, (rho - ref) / ref, it, r, sp, ex, time.time() - t)
        rows.append(row)
        print("%-6s b2/b1=%-6.4g b3/b1=%-6.4g %-15s ref %.10f | yogi %.10f %+.1e it %6d r %.1e "
              "kkt %.1e %+.1e | %6.1fs" % row, flush=True)
    with open(OUT, "w") as fh:
        fh.write("# eta=%g tol=%g tmax=%d\n" % (a.eta, a.tol, a.tmax))
        fh.write("# budget r2 r3 sigma ref rho_yogi rel_yogi it r spread excess seconds\n")
        for rw in rows:
            fh.write("%s %.6g %.6g %s %.12f %.12f %.3e %d %.3e %.3e %.3e %.1f\n"
                     % ((rw[0], rw[1], rw[2], rw[3].replace(" ", "_")) + rw[4:]))
    rel = np.array([rw[6] for rw in rows])
    print("instances %d, total %.0f s" % (len(rows), time.time() - t0))
    print("yogi from the barycenter within %.0e of the reference: %d of %d (worst %+.2e)"
          % (OK, (rel <= OK).sum(), len(rows), rel.max()))
    print("written", OUT)
