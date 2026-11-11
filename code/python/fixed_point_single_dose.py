"""The fixed point iteration for the single dose problem, Section 3 of
min_spectral_radius_vaccination.tex (subsection "A fixed point iteration").

At t^(k): the Perron vectors v, w of A1 diag(g(t^(k))) and its transpose, by
Arnoldi; a_p = (A1^T w)_p v_p / <w, v>; for each location i, mu_i solves
F_i(mu) = sum_{p in I_i} sigma_p max(1, sqrt(eps1 a_p / (mu sigma_p))) = c_i,
exactly, by sorting beta_p = eps1 a_p / sigma_p; t^(k+1)_p = max(1, sqrt(eps1 a_p / (mu_i sigma_p))).

Data as in gp_single_dose.py.  Checked against rho = 2.9059382878.
Nothing is written to disk.
"""

import numpy as np

from zf_jg_params import default_params
from zf_jg_doses import dose_budgets
from gp_single_dose import build_a1, rho_reduced, g_of_t, S_CSV, RHO_REPORTED


def t_of_mu(mu, a, sig, eps1):
    return np.maximum(1.0, np.sqrt(eps1 * a / (mu * sig)))


def solve_mu(a, sig, ci, eps1):
    """The exact solution of F_i(mu) = c_i, equation (mu-exact) of the paper.

    beta_p = eps1 a_p / sigma_p sorted; on [beta_(j), beta_(j+1)]
    F_i(mu) = C_j + B_j / sqrt(mu), C_j = sum_{l<=j} sigma_(l),
    B_j = sum_{l>j} sqrt(eps1 a_(l) sigma_(l)).  j is the largest index with
    F_i(beta_(j)) >= c_i (F_i(beta_(0)) = +inf), and mu = (B_j/(c_i-C_j))^2.
    """
    order = np.argsort(eps1 * a / sig)
    beta = (eps1 * a / sig)[order]
    s, r = sig[order], np.sqrt(eps1 * a * sig)[order]
    q = len(beta)
    C = np.concatenate([[0.0], np.cumsum(s)])            # C_j, j = 0:q
    B = np.concatenate([np.cumsum(r[::-1])[::-1], [0.0]])  # B_j, j = 0:q
    j = 0
    for jj in range(1, q):
        if C[jj] + B[jj] / np.sqrt(beta[jj - 1]) >= ci:  # F_i(beta_(jj))
            j = jj
    return (B[j] / (ci - C[j])) ** 2


def step(t, a1, sig, c, m, n, eps1):
    rho, v, w = rho_reduced(a1, g_of_t(t, eps1))
    a = (a1.T @ w) * v / float(w @ v)
    tn, mu = np.empty_like(t), np.empty(m)
    for i in range(m):
        sl = slice(i * n, (i + 1) * n)
        mu[i] = solve_mu(a[sl], sig[sl], c[i], eps1)
        tn[sl] = t_of_mu(mu[i], a[sl], sig[sl], eps1)
    return tn, rho, mu


if __name__ == "__main__":
    par = default_params()
    m, n, eps1, omv = par["m"], par["n"], par["eps1"], par["omega_v"]
    _, a1, _ = build_a1(par)
    s_mat = np.loadtxt(S_CSV, delimiter=",")
    b1, _, _ = dose_budgets()
    c = b1 / omv + s_mat.sum(axis=1)
    sig = s_mat.ravel()

    t = np.repeat(c / s_mat.sum(axis=1), n)      # uniform, spends every budget
    for k in range(500):
        tn, rho, mu = step(t, a1, sig, c, m, n, eps1)
        d = np.abs(tn - t).max() / np.abs(t).max()
        print("k=%3d  rho(t^k)=%.12f  |rho-ref|=%.2e  rel. change=%.2e"
              % (k, rho, abs(rho - RHO_REPORTED), d))
        t = tn
        if d < 1e-13:
            break
    rho = rho_reduced(a1, g_of_t(t, eps1))[0]
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print("iterations %d, rho = %.12f, reported %.10f" % (k + 1, rho, RHO_REPORTED))
    print("mu =", " ".join("%.6e" % x for x in mu))
    print("t == 1 at (location, group):", list(zip(*np.where(t.reshape(m, n) == 1.0))))
    print("budget residual:", (s_mat * t.reshape(m, n)).sum(axis=1) / c - 1)
    print("chi1 =\n%s" % (omv * (t.reshape(m, n) - 1.0)))
