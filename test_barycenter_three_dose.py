"""Does the barycenter start find the smallest rho for three doses?  A test over instances.

Instances: budget scale (weekly b1, or daily b1/7), ratios (b2, b3) / b1, and
sigma_2, sigma_3 equal to P (partially protected) or to P times independent
log-normal factors exp(0.5 Z) with a fixed seed.  The vaccinated counts of the
data are zero and cannot serve as sigma_2.

Per instance the reference is the smallest rho of the joint method
(Algorithm T3) over the 49 starts of Algorithm T4 and NRAND random starts.
Reported: that reference, the joint method and Gauss-Seidel (Algorithm T1) from
the barycenter, their relative excess over the reference, the KKT check of the
Gauss-Seidel endpoint, and the number of distinct values (to 1e-6 relative)
among the joint runs.

Writes test_barycenter_three_dose_results.txt.

Usage:
    python3 -u test_barycenter_three_dose.py
"""

import time
import numpy as np

import gs_three_dose as T

OUT = "test_barycenter_three_dose_results.txt"
NRAND = 20
RATIOS = [(1 / 16, 1 / 16), (1 / 4, 1 / 4), (1, 1), (4, 4), (1 / 2, 1 / 8), (1 / 8, 1 / 2)]
SIGMAS = ["P", "P*noise seed 1", "P*noise seed 2"]
OK = 1e-7             # the barycenter start is OK if rho <= reference * (1 + OK)


def sigma_pair(name):
    if name == "P":
        return T.P_, T.P_
    seed = int(name.split()[-1])
    rng = np.random.default_rng(seed)
    return (T.P_ * np.exp(0.5 * rng.standard_normal(T.P_.shape)),
            T.P_ * np.exp(0.5 * rng.standard_normal(T.P_.shape)))


def set_instance(days, r2, r3, sig):
    s2, s3 = sigma_pair(sig)
    b = T.b1[:, None] / days
    T.SCALES = [(b / T.S_).ravel(), (r2 * b / s2).ravel(), (r3 * b / s3).ravel()]


if __name__ == "__main__":
    rng = np.random.default_rng(20261017)
    rows = []
    t0 = time.time()
    for days in (1.0, 7.0):
        for r2, r3 in RATIOS:
            for sig in SIGMAS:
                set_instance(days, r2, r3, sig)
                t = time.time()
                st = [x for _, x in T.starts()]
                st += [rng.dirichlet(np.ones(T.n), size=3 * T.m).ravel() for _ in range(NRAND)]
                vals = np.array([np.exp(T.value_grad(T.joint(x0)[0])[0]) for x0 in st])
                ref = vals.min()
                rj = vals[0]                                   # st[0] is the barycenter
                xg, sw = T.gauss_seidel(st[0])
                rg = np.exp(T.value_grad(xg)[0])
                sp, ex = T.kkt(xg)
                distinct = len(np.unique(np.round(vals / ref, 6)))
                hit = np.mean(vals <= ref * (1 + OK))
                row = ("weekly" if days == 1 else "daily", r2, r3, sig, ref, rj, (rj - ref) / ref,
                       rg, (rg - ref) / ref, sw, sp, ex, distinct, hit, time.time() - t)
                rows.append(row)
                print("%-6s b2/b1=%-6.4g b3/b1=%-6.4g %-15s ref %.10f | joint bary %+.1e | GS bary %+.1e "
                      "sweeps %3d kkt %.1e %+.1e | distinct %2d, starts at ref %.2f | %5.1fs"
                      % (row[0], r2, r3, sig, ref, row[6], row[8], sw, sp, ex, distinct, hit, row[-1]),
                      flush=True)
    with open(OUT, "w") as fh:
        fh.write("# budget r2 r3 sigma ref rho_joint_bary rel_joint rho_gs_bary rel_gs sweeps "
                 "spread excess distinct frac_at_ref seconds\n")
        for r in rows:
            fh.write("%s %.6g %.6g %s %.12f %.12f %.3e %.12f %.3e %d %.3e %.3e %d %.3f %.1f\n"
                     % ((r[0], r[1], r[2], r[3].replace(" ", "_")) + r[4:]))
    rj = np.array([r[6] for r in rows])
    rg = np.array([r[8] for r in rows])
    print("instances %d, total %.0f s" % (len(rows), time.time() - t0))
    print("joint from the barycenter within %.0e of the reference: %d of %d (worst %+.2e)"
          % (OK, (rj <= OK).sum(), len(rows), rj.max()))
    print("GS    from the barycenter within %.0e of the reference: %d of %d (worst %+.2e)"
          % (OK, (rg <= OK).sum(), len(rows), rg.max()))
    print("written", OUT)
