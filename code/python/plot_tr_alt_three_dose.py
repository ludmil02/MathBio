"""Data of the figures of Section 4.3 of min_spectral_radius_vaccination.tex:
the trust region method (at most TR_MAXIT iterations) followed by the
alternating method of Section 4.1 (chi_2 step by projected Yogi, stopped when
the stationarity measure r of the joint problem is below 1e-10).

Convergence, on the weekly instance with b2 = b1/8, b3 = b1/2, sigma = P,
  from the barycenter and the starts (1,1,4) and (2,4,4):
  fig_tralt_tr_{key}.dat   per trust region iteration: it, relative excess
                           (rho - rho_ref)/rho_ref (at least FLOOR), r;
  fig_tralt_alt_{key}.dat  per sweep of the alternating method: sweep, excess, r.
Allocations, at the endpoint from the barycenter, on the weekly instances
  b2 = b3 = b1/4, sigma = P ("q") and b2 = b1/8, b3 = b1/2, sigma = P ("p"):
  fig_alloc_tralt_{q,p}.dat, one row per age group, chi_k at each location in
  units of 1e-3 per day.

Usage:
    python3 -u plot_tr_alt_three_dose.py
"""

import os
import numpy as np
from scipy.optimize import minimize

import gs_three_dose as T
import alt_three_dose as A
import yogi_three_dose as Y
from test_barycenter_three_dose import set_instance, sigma_pair
from plot_yogi_three_dose import write_alloc, NOTES, FLOOR

TR_MAXIT, R_TOL, SWEEPS = 500, 1e-10, 1000
REF = {"q": 1.661826042010, "p": 1.308827763800}     # rho_ref, test_barycenter_three_dose_results.txt
INST = {"q": (0.25, 0.25, "P"), "p": (0.125, 0.5, "P")}


def use(key):
    r2, r3, sig = INST[key]
    set_instance(1.0, r2, r3, sig)
    s2, s3 = sigma_pair(sig)
    A.SIG = [T.S_.ravel(), s2.ravel(), s3.ravel()]
    A.B = [T.b1, r2 * T.b1, r3 * T.b1]


def rho_r(x):
    f, gr = T.value_grad(x)
    return np.exp(f), Y.stationarity(x, gr)


def run(x0, hist_tr=None, hist_alt=None):
    """Trust region, then (if it reached TR_MAXIT) sweeps chi_1, chi_3, chi_2 until r < R_TOL."""
    opts = dict(T.OPTS)
    opts["maxiter"] = TR_MAXIT

    def cb(xk, state):
        if hist_tr is not None:
            hist_tr.append((state.nit,) + rho_r(T.project(xk)))
        return False

    res = minimize(T.value_grad, x0, jac=True, method="trust-constr", constraints=[T.simplices(3)],
                   bounds=[(0.0, None)] * (3 * T.N), options=opts, callback=cb)
    x = T.project(res.x)
    if res.status != 0:
        return x, res.nit, 0
    xr = x.reshape(3, T.N)
    chi = [T.SCALES[k] * xr[k] for k in range(3)]
    if hist_alt is not None:
        hist_alt.append((0,) + rho_r(x))
    for sw in range(1, SWEEPS + 1):
        chi[0] = A.step_chi1(chi)
        chi[2] = A.step_chi3(chi)
        chi[1], _, _ = A.step_chi2(chi, 1e-11, 20000)
        x = np.concatenate([chi[k] / T.SCALES[k] for k in range(3)])
        rho, r = rho_r(x)
        if hist_alt is not None:
            hist_alt.append((sw, rho, r))
        if r < R_TOL:
            break
    return x, res.nit, sw


def write(fname, rows, ref, cols):
    path = os.path.join(NOTES, fname)
    with open(path, "w") as fh:
        fh.write(" ".join(cols) + "\n")
        for r in rows:
            fh.write("%d %.6e %.6e\n" % (r[0], max((r[1] - ref) / ref, FLOOR), r[2]))
    print("written", path, len(rows), "rows")


if __name__ == "__main__":
    st = dict(T.starts())
    use("p")
    for key, name in (("bary", "barycenter"), ("k1", "k=1 i=1 j=4"), ("k2", "k=2 i=4 j=4")):
        htr, halt = [], []
        x, nit, sw = run(st[name], htr, halt)
        rho, r = rho_r(x)
        print("p %-12s tr_it=%d sweeps=%d rho=%.12f (%+.1e) r=%.1e"
              % (name, nit, sw, rho, (rho - REF["p"]) / REF["p"], r), flush=True)
        write("fig_tralt_tr_%s.dat" % key, htr, REF["p"], ["it", "excess", "r"])
        if halt:
            write("fig_tralt_alt_%s.dat" % key, halt, REF["p"], ["sweep", "excess", "r"])
    for key in ("q", "p"):
        use(key)
        x, nit, sw = run(st["barycenter"])
        rho, r = rho_r(x)
        print("%s barycenter tr_it=%d sweeps=%d rho=%.12f (%+.1e) r=%.1e"
              % (key, nit, sw, rho, (rho - REF[key]) / REF[key], r), flush=True)
        write_alloc("fig_alloc_tralt_%s.dat" % key, x)
