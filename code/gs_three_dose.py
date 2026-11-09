"""Three doses: the joint local method and block Gauss-Seidel over the dose types.

The test instance of Section S3 of the supplement with the weekly budgets:
sigma_2 = sigma_3 = P (partially protected), b_1 the weekly budget, b_2 = b_3 =
b_1 / 4.  The objective function is (4.1), log rho(A1 diag(g)), in the variables
u-hat of property (i) of Section 2, on the product of 3 m simplices Delta_{n-1}.
In this file u is the chi of the write-up, u_k = SCALES[k] * u-hat_k.

Starts: the barycenter, and the 48 points with one factor (k, i) at a vertex
e_j of its simplex, moved a fraction DELTA towards the barycenter, the other
factors at the barycenter.

For each start:
  joint   trust region interior point method in all 3 N variables;
  GS      sweeps over k = 1, 2, 3, each minimizing over u-hat_k with the other
          two fixed (same method, N variables), until the relative change of rho
          over a sweep is below TOL_SWEEP or MAX_SWEEPS sweeps are done.
At each computed point the KKT conditions are evaluated: per factor (k, i) the
spread of the gradient over the coordinates that are positive, and the smallest
excess of the gradient over the multiplier at the coordinates that vanish.

Writes the log to stdout and the per-start results to gs_three_dose_results.txt.

Usage:
    python3 -u gs_three_dose.py
"""

import os
import time
import numpy as np
import scipy.sparse as sps
from scipy.optimize import minimize, LinearConstraint

import gp_single_dose as G
from zf_jg_params import default_params
from zf_jg_doses import dose_budgets

OUT = "gs_three_dose_results.txt"
P_CSV = os.path.join(G.ROOT, "old_files/solution_nb_Data/p1Matrix.csv")
DELTA = 1e-3          # vertex starts moved this fraction towards the barycenter
TOL_SWEEP = 1e-12     # GS stops when |rho_new - rho_old| / rho_old < TOL_SWEEP
MAX_SWEEPS = 200
FREE = 1e-8           # u-hat above this counts as positive in the KKT check
OPTS = {"gtol": 1e-14, "xtol": 1e-15, "maxiter": 20000, "verbose": 0}

par = default_params()
m, n = par["m"], par["n"]
N = m * n
e1, e2, e3, om, omv = par["eps1"], par["eps2"], par["eps3"], par["omega"], par["omega_v"]
_, a1, _ = G.build_a1(par)
A1 = sps.csr_matrix(a1)
S_ = np.loadtxt(G.S_CSV, delimiter=",")
P_ = np.loadtxt(P_CSV, delimiter=",")
b1, _, _ = dose_budgets()
SCALES = [(b1[:, None] / S_).ravel(), (b1[:, None] / 4 / P_).ravel(),
          (b1[:, None] / 4 / P_).ravel()]          # u_k = SCALES[k] * u-hat_k


def gfun(u1, u2, u3):
    """(2.4), the closed form of g."""
    p1, p2, p3 = u1 / (omv + e2 * u2), e2 * u2 / om, 1.0 + e3 * u3 / om
    return (1.0 + (1.0 - e1) * p1 + p1 * p2) / (1.0 + p1 + p1 * p2 + p1 * p2 * p3)


def value_grad(x):
    """log rho and its gradient in u-hat, shape (3 N,); dg/du by complex step."""
    x = np.asarray(x).reshape(3, N)
    u = [SCALES[k] * x[k] for k in range(3)]
    rho, v, w = G.rho_reduced(a1, gfun(*u))
    drho = (A1.T @ w) * v / float(w @ v)
    gr = np.empty((3, N))
    h = 1e-30
    for k in range(3):
        uc = [a.astype(complex) for a in u]
        uc[k] = uc[k] + 1j * h
        gr[k] = drho * np.imag(gfun(*uc)) / h * SCALES[k] / rho
    return np.log(rho), gr.ravel()


def simplices(nb):
    e = np.zeros((nb * m, nb * N))
    for r in range(nb * m):
        e[r, r * n:(r + 1) * n] = 1.0
    return LinearConstraint(e, np.ones(nb * m), np.ones(nb * m))


def solve(f, x0, nb):
    return minimize(f, x0, jac=True, method="trust-constr",
                    constraints=[simplices(nb)], bounds=[(0.0, None)] * (nb * N),
                    options=OPTS)


def project(x):
    """Clip the small negative entries the interior point method can leave."""
    x = np.clip(np.asarray(x).reshape(3 * m, n), 0.0, None)
    return (x / x.sum(axis=1, keepdims=True)).ravel()


def joint(x0):
    r = solve(value_grad, x0, 3)
    return project(r.x), r.nit


def gauss_seidel(x0, hist=None):
    """hist, if a list, receives (sweep, rho) at the start and after every sweep."""
    x = np.array(x0, dtype=float).reshape(3, N)
    rho_old = np.exp(value_grad(x.ravel())[0])
    if hist is not None:
        hist.append((0, rho_old))
    for sweep in range(1, MAX_SWEEPS + 1):
        for k in range(3):
            def fk(y, k=k):
                z = x.copy()
                z[k] = y
                f, gr = value_grad(z.ravel())
                return f, gr.reshape(3, N)[k]
            y = np.clip(solve(fk, x[k], 1).x, 0.0, None).reshape(m, n)
            x[k] = (y / y.sum(axis=1, keepdims=True)).ravel()   # back onto the simplices
        rho = np.exp(value_grad(x.ravel())[0])
        if hist is not None:
            hist.append((sweep, rho))
        if abs(rho - rho_old) / rho_old < TOL_SWEEP:
            return x.ravel(), sweep
        rho_old = rho
    return x.ravel(), MAX_SWEEPS


def kkt(x):
    """Max relative gradient spread over positive coordinates, and the smallest
    relative excess (g_j - lambda)/|lambda| at vanishing ones (>= 0 at a KKT point)."""
    _, gr = value_grad(x)
    xr, gr = x.reshape(3 * m, n), gr.reshape(3 * m, n)
    spread, excess = 0.0, np.inf
    for r in range(3 * m):
        free = xr[r] > FREE
        lam = gr[r, free].mean()
        spread = max(spread, (gr[r, free].max() - gr[r, free].min()) / abs(lam))
        if (~free).any():
            excess = min(excess, ((gr[r, ~free] - lam) / abs(lam)).min())
    return spread, excess


def starts():
    bary = np.full(3 * N, 1.0 / n)
    out = [("barycenter", bary)]
    for k in range(3):
        for i in range(m):
            for j in range(n):
                x = bary.copy().reshape(3, m, n)
                x[k, i] = (1 - DELTA) * np.eye(n)[j] + DELTA / n
                out.append(("k=%d i=%d j=%d" % (k + 1, i + 1, j + 1), x.ravel()))
    return out


if __name__ == "__main__":
    rows = []
    t_all = time.time()
    for name, x0 in starts():
        t = time.time()
        xj, itj = joint(x0)
        tj = time.time() - t
        t = time.time()
        xg, sw = gauss_seidel(x0)
        tg = time.time() - t
        rj, rg = np.exp(value_grad(xj)[0]), np.exp(value_grad(xg)[0])
        (sj, ej), (sg, eg) = kkt(xj), kkt(xg)
        rows.append((name, rj, itj, tj, sj, ej, rg, sw, tg, sg, eg))
        print("%-12s joint rho=%.10f it=%4d %5.1fs kkt %.1e %+.1e | GS rho=%.10f sweeps=%3d %6.1fs kkt %.1e %+.1e"
              % rows[-1], flush=True)
    print("total %.1f s" % (time.time() - t_all))
    with open(OUT, "w") as fh:
        fh.write("# start rho_joint it_joint t_joint spread_joint excess_joint "
                 "rho_gs sweeps t_gs spread_gs excess_gs\n")
        for r in rows:
            fh.write("%s %.12f %d %.2f %.3e %.3e %.12f %d %.2f %.3e %.3e\n"
                     % ((r[0].replace(" ", "_"),) + r[1:]))
    for label, col in (("joint", 1), ("GS", 6)):
        vals = np.array([r[col] for r in rows])
        distinct = np.unique(np.round(vals, 8))
        print("%-5s: %d starts, min %.10f max %.10f, distinct to 1e-8: %s"
              % (label, len(vals), vals.min(), vals.max(), distinct))
    print("written", OUT)
