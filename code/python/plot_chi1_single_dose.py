"""Data of the figure of Section 3.2: the minimizing chi1 for a single dose, and
the chi1 of John's method.

Solves the single dose problem as gp_single_dose.py does (route A, weekly b1)
and writes fig_chi1_single_dose.dat; John's method, l0 of eigenvalues_H_ZF_JG.nb
scaled in each location to spend the same b1, goes to fig_chi1_john.dat.  The
write-up plots both with pgfplots: one row per age group, one column per
location, chi1 in units of 1e-3 per day.

Usage:
    python3 -u plot_chi1_single_dose.py
"""

import os
import numpy as np

import gp_single_dose as G
from zf_jg import ngm_factored, dominant, john_method
from zf_jg_params import default_params
from zf_jg_doses import dose_budgets

OUT = os.path.join(G.ROOT, "notes", "figures", "dat", "fig_chi1_single_dose.dat")   # read by make_figures.py
OUT_JOHN = os.path.join(G.ROOT, "notes", "figures", "dat", "fig_chi1_john.dat")
LOCATIONS = ["Northeast", "Midwest", "South", "West"]


def john():
    par = default_params()
    m, n = par["m"], par["n"]
    s_mat = np.loadtxt(G.S_CSV, delimiter=",")
    b1, _, _ = dose_budgets()
    _, l0 = john_method(ngm_factored(par=par), par)
    chi1 = (b1 / (l0 * s_mat).sum(axis=1))[:, None] * l0
    zero = np.zeros((m, n))
    return chi1, dominant(ngm_factored(chi1, zero, zero, par=par))[0]


def write(fname, chi1):
    with open(fname, "w") as fh:
        fh.write("age " + " ".join(LOCATIONS) + "\n")
        for j in range(chi1.shape[1]):
            fh.write("%d " % j + " ".join("%.6f" % (1e3 * chi1[i, j])
                                          for i in range(chi1.shape[0])) + "\n")
    print("written", fname)


def minimizer():
    par = default_params()
    m, n, eps1, omv = par["m"], par["n"], par["eps1"], par["omega_v"]
    _, a1, _ = G.build_a1(par)
    s_mat = np.loadtxt(G.S_CSV, delimiter=",")
    b1, _, _ = dose_budgets()
    red = G.Reduced(a1, s_mat, b1, eps1, omv)
    ra = G.solve_route_a(red, np.full(m * n, 1.0 / n))
    t = red.t_of(ra.x)
    chi1 = omv * (t.reshape(m, n) - 1.0)
    chi1[chi1 < 1e-12] = 0.0           # the 60+ group, zero to rounding
    return chi1, G.rho_reduced(a1, G.g_of_t(t, eps1))[0]


if __name__ == "__main__":
    chi1, rho = minimizer()
    print("minimum: rho = %.10f" % rho)
    print("chi1 =\n%s" % np.array2string(chi1, precision=6))
    write(OUT, chi1)
    chi_j, rho_j = john()
    print("John's method: rho = %.10f" % rho_j)
    print("chi1 =\n%s" % np.array2string(chi_j, precision=6))
    write(OUT_JOHN, chi_j)
