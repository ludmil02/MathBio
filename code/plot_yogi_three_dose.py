"""Data of the figures of Section T5.3 of math_bio_three_doses.tex.

Figure T5.1, convergence on the weekly instance (b2 = b3 = b1/4, sigma = P):
  projected Yogi (Algorithm T5.1) from the barycenter, from the start with
  dose 3 at a vertex (k, i, j) = (3, 1, 4), and from the slowest start
  (2, 4, 1): per iteration the relative excess (rho - rho*)/rho* and r (T5.4);
  block Gauss-Seidel (Algorithm T1) from the barycenter and from (3, 1, 4):
  the relative excess per sweep.  rho* is the smallest rho of these runs.
  Files fig_yogi_hist_{bary,k3,k2}.dat and fig_gs_hist_{bary,k3}.dat.

Figures T5.2 and T5.3, the two weekly instances of Table T5.2 in which Yogi
  ends below rho_ref: chi_k for k = 1, 2, 3 at the endpoint of the joint method
  (Algorithm T3) and of Yogi, both from the barycenter.  Files
  fig_alloc_{s2,p}_{joint,yogi}.dat, one row per age group, columns
  chi_k at each location in units of 1e-3 per day.

Usage:
    python3 -u plot_yogi_three_dose.py
"""

import os
import numpy as np

import gs_three_dose as T
import yogi_three_dose as Y
from test_barycenter_three_dose import set_instance

NOTES = os.path.join(T.G.ROOT, "notes", "suplemental")   # read by make_figures.py
LOC = ["NE", "MW", "S", "W"]
FLOOR = 1e-16                       # relative excesses below this are plotted at it
YOGI = dict(eta=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-10, tmax=200000)
MAXROWS = 1500                      # thin the long histories to about this many rows


def start(name):
    return dict(T.starts())[name]


def write_hist(fname, rows, rho_star, cols):
    fname = os.path.join(NOTES, fname)
    rows = np.array(rows, dtype=float)
    keep = np.unique(np.r_[np.arange(0, len(rows), max(1, len(rows) // MAXROWS)), len(rows) - 1])
    with open(fname, "w") as fh:
        fh.write(" ".join(cols) + "\n")
        for r in rows[keep]:
            exc = max((r[1] - rho_star) / rho_star, FLOOR)
            fh.write("%d %.6e" % (r[0], exc) + "".join(" %.6e" % v for v in r[2:]) + "\n")
    print("written", fname, len(keep), "rows")


def write_alloc(fname, x):
    fname = os.path.join(NOTES, fname)
    chi = [1e3 * T.SCALES[k] * x.reshape(3, T.N)[k] for k in range(3)]   # 1e-3 per day
    with open(fname, "w") as fh:
        fh.write("age " + " ".join("k%d%s" % (k + 1, l) for k in range(3) for l in LOC) + "\n")
        for j in range(T.n):
            fh.write("%d" % j + "".join(" %.6f" % chi[k][i * T.n + j]
                                        for k in range(3) for i in range(T.m)) + "\n")
    print("written", fname)


if __name__ == "__main__":
    # Figure T5.1: the weekly instance of Section T3 (module defaults of gs_three_dose)
    yh, gh = {}, {}
    for key, name in (("bary", "barycenter"), ("k3", "k=3 i=1 j=4"), ("k2", "k=2 i=4 j=1")):
        yh[key] = []
        x, it, r = Y.yogi(start(name), hist=yh[key], **YOGI)
        print("yogi %-12s rho=%.12f it=%d r=%.1e" % (name, yh[key][-1][1], it, r), flush=True)
    for key, name in (("bary", "barycenter"), ("k3", "k=3 i=1 j=4")):
        gh[key] = []
        T.gauss_seidel(start(name), hist=gh[key])
        print("GS   %-12s rho=%.12f sweeps=%d" % (name, gh[key][-1][1], gh[key][-1][0]), flush=True)
    rho_star = min(min(r[1] for r in h) for h in list(yh.values()) + list(gh.values()))
    print("rho* = %.12f" % rho_star)
    for key, h in yh.items():
        write_hist("fig_yogi_hist_%s.dat" % key, h, rho_star, ["it", "excess", "r"])
    for key, h in gh.items():
        write_hist("fig_gs_hist_%s.dat" % key, h, rho_star, ["sweep", "excess"])

    # Figures T5.2 and T5.3: the two instances in which Yogi ends below rho_ref
    bary = np.full(3 * T.N, 1.0 / T.n)
    for key, (r2, r3, sig) in (("s2", (0.25, 0.25, "P*noise seed 2")),
                               ("p", (0.125, 0.5, "P"))):
        set_instance(1.0, r2, r3, sig)
        xj, itj = T.joint(bary)
        xy, ity, _ = Y.yogi(bary, **YOGI)
        rj, ry = np.exp(T.value_grad(xj)[0]), np.exp(T.value_grad(xy)[0])
        print("%s: joint rho=%.10f (it %d), yogi rho=%.10f (it %d)" % (key, rj, itj, ry, ity), flush=True)
        write_alloc("fig_alloc_%s_joint.dat" % key, xj)
        write_alloc("fig_alloc_%s_yogi.dat" % key, xy)
