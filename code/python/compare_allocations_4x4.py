"""Section 6 of notes/stratified_seir_ngm.tex: four allocations of the same first doses,
4 x 4 data at t = 347, compared by rho(NGM).

  actual          chi1 = D / S, D = C2 of zf_jg_doses.py;
  left ev, doses  chi1 S = b1_i w_p / sum_{l in I_i} w_l  (the alternative of the paper);
  left ev, rates  chi1 = b1_i w_p / sum_{l in I_i} w_l S_l;
  minimizer       the fixed point iteration of Section 4 (Lemma 4.1, (4.11)),
                  checked by the KKT conditions (4.8) and by scipy's trust-constr
                  from random starts.

w_p: the entry in row 2p-1 of the dominant left eigenvector of NGM at chi = 0.
rho(NGM) is computed from the NGM of order 2N (zf_jg.ngm_factored); the fixed
point iteration works with A1 diag(g) of order N (zf_jg.a1_matrix), with
g = (1 - e) + e / t, e = 1 - r (1 - eps1), t = 1 + chi1 / omega_v.

Usage:
    python3 -u compare_allocations_4x4.py | tee intermediate/compare_allocations_4x4.log
"""

import os
import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.optimize import minimize, LinearConstraint, Bounds

import zf_jg as Z
from zf_jg_params import default_params
from zf_jg_doses import C2, dose_budgets

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
S_CSV = os.path.join(HERE, "intermediate", "zf_jg_t347", "s1Matrix.csv")
LOCS = ["Northeast", "Midwest", "South", "West"]
AGES = ["0-19", "20-39", "40-59", "60+"]


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


par = default_params()
m, n = par["m"], par["n"]
N = m * n
eps1, omv = par["eps1"], par["omega_v"]
S = np.loadtxt(S_CSV, delimiter=",")
b1, _, _ = dose_budgets()
zero = np.zeros((m, n))

A1 = Z.a1_matrix(par)                                  # sparse, order N
r = Z.r_vector(par)                                    # q_m / q_s per group
e = 1.0 - r * (1.0 - eps1)                             # (4.3) at theta = zeta = 0
sig_t = S.ravel() * omv                                # sigma-tilde, alpha = 1, omega_p = omega_v
c = b1 + sig_t.reshape(m, n).sum(axis=1)


def rho_ngm(chi1):
    return Z.dominant(Z.ngm_factored(chi1, zero, zero, par=par))[0]


def perron_pair(g):
    mat = (A1 @ sps.diags(g)).tocsr()
    v0 = np.ones(N)
    lam, v = spsl.eigs(mat, k=1, which="LM", tol=1e-14, v0=v0)
    _, w = spsl.eigs(mat.T.tocsr(), k=1, which="LM", tol=1e-14, v0=v0)
    v, w = np.real(v[:, 0]), np.real(w[:, 0])
    return float(np.real(lam[0])), v * np.sign(v.sum()), w * np.sign(w.sum())


def g_of_t(t):
    return (1.0 - e) + e / t


def solve_mu(ea, sg, ci):
    """(4.11): the exact solution of F_i(mu) = c_i."""
    beta = ea / sg
    order = np.argsort(beta)
    beta, s, rr = beta[order], sg[order], np.sqrt(ea * sg)[order]
    q = len(beta)
    C = np.concatenate([[0.0], np.cumsum(s)])
    B = np.concatenate([np.cumsum(rr[::-1])[::-1], [0.0]])
    j = 0
    for jj in range(1, q):
        if C[jj] + B[jj] / np.sqrt(beta[jj - 1]) >= ci:
            j = jj
    return (B[j] / (ci - C[j])) ** 2


def a_vector(t):
    rho, v, w = perron_pair(g_of_t(t))
    return rho, (A1.T @ w) * v / float(w @ v)


def fixed_point(t, tol=1e-13, kmax=1000):
    for k in range(kmax):
        rho, a = a_vector(t)
        tn = np.empty_like(t)
        mu = np.empty(m)
        for i in range(m):
            sl = slice(i * n, (i + 1) * n)
            mu[i] = solve_mu(e[sl] * a[sl], sig_t[sl], c[i])
            tn[sl] = np.maximum(1.0, np.sqrt(e[sl] * a[sl] / (mu[i] * sig_t[sl])))
        d = np.abs(tn - t).max() / np.abs(t).max()
        if k < 5 or k % 10 == 0 or d < tol:
            say("  k = %3d   rho(A1 diag(g)) = %.12f   rel. change of t = %.2e" % (k, rho, d))
        t = tn
        if d < tol:
            break
    return t, mu, k + 1


def kkt_report(t):
    """The spread of e_p a_p / (t_p^2 sigma_p) over the groups with t_p > 1, relative
    to mu_i, and the largest excess of e_p a_p / sigma_p over mu_i where t_p = 1."""
    _, a = a_vector(t)
    ratio = e * a / (t**2 * sig_t)
    out = []
    for i in range(m):
        sl = slice(i * n, (i + 1) * n)
        act = t[sl] > 1.0 + 1e-12
        mu = ratio[sl][act].mean()
        spread = (ratio[sl][act].max() - ratio[sl][act].min()) / mu
        excess = ((e[sl] * a[sl] / sig_t[sl])[~act] / mu - 1.0).max() if (~act).any() else -np.inf
        out.append((spread, excess))
    return out


def chi_from_t(t):
    return (omv * (t - 1.0)).reshape(m, n)


def allocations():
    """The four allocations of Section 6, chi1 of shape (m, n) each, and the
    fixed point data (t, mu, number of iterations)."""
    allocs = {}
    allocs["actual"] = C2 / S
    _, l0 = Z.john_method(Z.ngm_factored(par=par), par)       # w_p, rows 2p-1, (m, n)
    allocs["left ev, doses"] = (b1[:, None] * l0 / l0.sum(axis=1, keepdims=True)) / S
    allocs["left ev, rates"] = b1[:, None] * l0 / (l0 * S).sum(axis=1, keepdims=True)
    say("fixed point iteration from the uniform rates per location")
    t0 = np.repeat(c / sig_t.reshape(m, n).sum(axis=1), n)
    t_fp, mu, iters = fixed_point(t0)
    allocs["minimizer"] = chi_from_t(t_fp)
    return allocs, (t_fp, mu, iters)


def main():
    say("4 x 4 data at t = 347: eps1 = %g, omega_v = %.6g" % (eps1, omv))
    say("b1 = %s" % np.array2string(b1, precision=0))
    say("e_p = 1 - r_p (1 - eps1) by age: %s" % np.array2string(e[:n], precision=4))
    rho0 = rho_ngm(zero)
    say("rho(NGM) at chi = 0: %.12f" % rho0)

    allocs, (t_fp, mu, iters) = allocations()
    say("fixed point: %d iterations, mu = %s" % (iters, " ".join("%.4e" % x for x in mu)))
    for i, (spread, excess) in enumerate(kkt_report(t_fp)):
        say("  KKT %-9s spread over t_p > 1: %.1e   max excess at t_p = 1: %s"
            % (LOCS[i], spread, "none" if excess == -np.inf else "%.3f" % excess))

    # independent check: trust-constr on Psi(t) from random feasible starts
    say("check: trust-constr from 8 random starts")
    Acon = np.zeros((m, N))
    for i in range(m):
        Acon[i, i * n:(i + 1) * n] = sig_t[i * n:(i + 1) * n]
    lc = LinearConstraint(Acon, c, c)

    def f(t):
        rho, a = a_vector(t)
        return rho, -e * a / t**2

    rng = np.random.default_rng(0)
    best = []
    for s in range(8):
        u = rng.random(N)
        t_s = np.empty(N)
        for i in range(m):
            sl = slice(i * n, (i + 1) * n)
            x = u[sl] / (u[sl] * sig_t[sl]).sum() * (c[i] - sig_t[sl].sum())
            t_s[sl] = 1.0 + x
        res = minimize(f, t_s, jac=True, method="trust-constr", constraints=[lc],
                       bounds=Bounds(np.ones(N), np.inf),
                       options=dict(gtol=1e-12, xtol=1e-14, maxiter=5000))
        rr = rho_ngm(chi_from_t(res.x))
        best.append(rr)
        say("  start %d: rho(NGM) = %.12f   max |chi - chi_fp| / max chi_fp = %.1e"
            % (s, rr, np.abs(chi_from_t(res.x) - allocs["minimizer"]).max() / allocs["minimizer"].max()))

    say("rho(NGM) of the four allocations (doses used / b1 per location in brackets)")
    rho_min = rho_ngm(allocs["minimizer"])
    for name, chi in allocs.items():
        used = (chi * S).sum(axis=1) / b1
        rr = rho_ngm(chi)
        say("  %-15s rho = %.10f   reduction from chi = 0: %6.3f%%   above the minimizer: %6.3f%%   [%s]"
            % (name, rr, 100 * (rho0 - rr) / rho0, 100 * (rr - rho_min) / rho_min,
               " ".join("%.12f" % x for x in used)))
    say("chi1 (1e-3 per day), rows %s, columns %s" % (", ".join(LOCS), ", ".join(AGES)))
    for name, chi in allocs.items():
        say("  %s" % name)
        for i in range(m):
            print("      " + " ".join("%9.4f" % (1e3 * x) for x in chi[i]), flush=True)
    say("done")


if __name__ == "__main__":
    main()
