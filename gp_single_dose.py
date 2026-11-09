"""The single dose problem of Section 3, with the geometric program of equation (S4.2).

Three routes to the same minimum, from math_bio_positive_matrices_clean.tex
and math_bio_positive_matrices_supplement.tex (numbers with an S):

  A  the interior penalty method of Theorem 3.2 in the normalized variables
     what = u-hat_1 of (3.5): minimize Psi over the product K-hat (3.7) of
     solid simplices,
     with w_(i-1)n+j = b1_i what_ij / (omega_v sigma_1p) and t = 1 + w.  Smooth
     and convex by Theorem 3.1; the gradient is (3.20), one Perron pair of
     A1 diag(g) and one of its transpose, both by Arnoldi.  The raw variables
     t carry constraint coefficients of order 1e7 and the solver stalls on
     them, which is what the normalization is for.

  B  equation (S4.2): the geometric program in (theta, x, t), solved in its
     logarithmic form, where every constraint is a log-sum-exp of affine
     functions.  No eigenvalue enters the objective; the Collatz-Wielandt
     inequality M x <= theta x of (S4.1) carries it.

  C  route A from eight random feasible starting points, which is the test of
     "local implies global" reported in Section S2.

Then the allocation of John's method, l0 scaled in each location to spend the
same b1, for comparison, as in min_rho_single_dose_H_ZF_JG.nb.

The optimizer in both cases is scipy's trust-constr, which with inequality
constraints present runs its trust region interior point method: route A in the
N variables what with a quasi-Newton Hessian, route B in the 1 + 2N variables
(tau, zeta, eta) with the objective Hessian passed as zero and analytic
Jacobians for the log-sum-exp constraints.  Frank-Wolfe and mirror descent are
mentioned in the note as alternatives the simplex structure permits; they are
not implemented here.

Checked against rho = 2.9059382878, the minimum of Section S2 and of
min_rho_single_dose_H_ZF_JG.nb.

Data, the same as in that notebook: A1 from ac_matrix of zf_jg.py scaled by
d_c, then A1 = Ac[0::2, 0::2] per Theorem A.1; S from
old_files/solution_nb_Data/s1Matrix.csv (Sstar of the notebook); b1 from
zf_jg_doses.py, the weekly budget, the row sums of C2.  Nothing is written to
disk.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from scipy.special import logsumexp

from zf_jg_params import default_params
from zf_jg import ac_matrix, ngm_factored, dominant, john_method
from zf_jg_doses import dose_budgets

RHO_REPORTED = 2.9059382878      # Section S2 and min_rho_single_dose_H_ZF_JG.nb
S_CSV = "old_files/solution_nb_Data/s1Matrix.csv"


# ----------------------------------------------------------------- the data

def build_a1(par):
    """A1 of Proposition A.1, with Ac = diag(d0) . ac_matrix, A1 = Ac[0::2, 0::2].

    d_c[(i-1)n+j] = vS[i] ma[i,j] vpsi_S[j] vbeta[j] is the prefactor that
    compute_yz puts into Y and Z; the write-up puts it into Ac instead, so that
    y_k = s*_k is a pure fraction.  Both are diagonal, so the two orderings
    give the same next generation matrix.
    """
    dc = (par["vS"][:, None] * par["ma"]
          * (par["vpsi_S"] * par["vbeta"])[None, :]).ravel()
    ac = (sps.diags(np.repeat(dc, 2)) @ ac_matrix(par)).tocsr()
    return ac, ac.toarray()[0::2, 0::2], dc


def rho_reduced(a1, g):
    """rho(A1 diag(g)) and both Perron vectors, by Arnoldi.

    The start vector is fixed, so that a run reproduces to the last digit.
    """
    mm = sps.csr_matrix(a1) @ sps.diags(g)
    v0 = np.ones(mm.shape[0])
    lam, v = spsl.eigs(mm, k=1, which="LR", tol=0, v0=v0)
    _, w = spsl.eigs(mm.T.tocsr(), k=1, which="LR", tol=0, v0=v0)
    v, w = np.real(v[:, 0]), np.real(w[:, 0])
    if v.sum() < 0:
        v = -v
    if w.sum() < 0:
        w = -w
    return float(np.real(lam[0])), v, w


def g_of_t(t, eps1):
    return (1.0 - eps1) + eps1 / t


# ------------- route A, Theorems 3.1 and 3.2 in the variables what of (3.5)

class Reduced:
    """Psi and its gradient in the normalized variables what."""

    def __init__(self, a1, s_mat, b1, eps1, omv):
        self.a1, self.s_mat, self.eps1 = a1, s_mat, eps1
        self.m, self.n = s_mat.shape
        self.scale = (b1[:, None] / (omv * s_mat)).ravel()   # w = scale * what

    def t_of(self, what):
        return 1.0 + self.scale * np.asarray(what).ravel()

    def value_grad(self, what):
        t = self.t_of(what)
        rho, v, w = rho_reduced(self.a1, g_of_t(t, self.eps1))
        dpsi_dt = -(self.eps1 / (rho * t**2)) * ((self.a1.T @ w) * v) / float(w @ v)
        return np.log(rho), dpsi_dt * self.scale

    def marginal(self, what):
        """The ratio of (3.21) per dose, shape (m, n): equal across free groups."""
        t = self.t_of(what)
        rho, v, w = rho_reduced(self.a1, g_of_t(t, self.eps1))
        num = (self.eps1 / (rho * t**2)) * ((self.a1.T @ w) * v) / float(w @ v)
        return (num / self.s_mat.ravel()).reshape(self.m, self.n), t, rho


def simplex_constraint(m, n):
    """E what <= 1, the m solid simplices of K-hat, (3.7)."""
    e = np.zeros((m, m * n))
    for i in range(m):
        e[i, i * n:(i + 1) * n] = 1.0
    return LinearConstraint(e, -np.inf, np.ones(m))


def solve_route_a(red, what0):
    m, n = red.m, red.n
    return minimize(red.value_grad, np.asarray(what0).ravel(), jac=True,
                    method="trust-constr", constraints=[simplex_constraint(m, n)],
                    bounds=[(0.0, None)] * (m * n),
                    options={"gtol": 1e-14, "xtol": 1e-15, "maxiter": 20000,
                             "verbose": 0})


# -------------------------------------------------- route B, equation (S4.2)
# y = (tau, zeta[0:N], eta[0:N]), theta = e^tau, x = e^zeta, t = e^eta.
# Collatz-Wielandt row k, as log-sum-exp over 2N terms <= 0:
#     log[(A1)_kl (1-eps1)] + zeta_l          - zeta_k - tau
#     log[(A1)_kl  eps1   ] + zeta_l - eta_l  - zeta_k - tau
# dose i:  logsumexp_j ( log(S_ij / c_i) + eta_(i-1)n+j ) <= 0
# box:     eta_k >= 0, that is t_k >= 1

def gp_constraints(a1, s_mat, c_vec, eps1):
    nt = a1.shape[0]
    m, n = s_mat.shape
    la1 = np.log(a1)
    log_a, log_b = la1 + np.log1p(-eps1), la1 + np.log(eps1)
    ls = np.log(s_mat / c_vec[:, None])

    def terms(y, k):
        tau, zeta, eta = y[0], y[1:1 + nt], y[1 + nt:]
        return np.concatenate([log_a[k] + zeta,
                               log_b[k] + zeta - eta]) - zeta[k] - tau

    def cw(y):
        return np.array([logsumexp(terms(y, k)) for k in range(nt)])

    def cw_jac(y):
        jac = np.zeros((nt, 1 + 2 * nt))
        for k in range(nt):
            e = terms(y, k)
            p = np.exp(e - logsumexp(e))               # softmax weights
            p1, p2 = p[:nt], p[nt:]
            jac[k, 0] = -1.0
            jac[k, 1:1 + nt] += p1 + p2
            jac[k, 1 + k] -= 1.0
            jac[k, 1 + nt:] -= p2
        return jac

    def dose(y):
        eta = y[1 + nt:]
        return np.array([logsumexp(ls[i] + eta[i * n:(i + 1) * n])
                         for i in range(m)])

    def dose_jac(y):
        eta = y[1 + nt:]
        jac = np.zeros((m, 1 + 2 * nt))
        for i in range(m):
            e = ls[i] + eta[i * n:(i + 1) * n]
            jac[i, 1 + nt + i * n:1 + nt + (i + 1) * n] = np.exp(e - logsumexp(e))
        return jac

    return [NonlinearConstraint(cw, -np.inf, 0.0, jac=cw_jac),
            NonlinearConstraint(dose, -np.inf, 0.0, jac=dose_jac)]


def solve_route_b(a1, s_mat, c_vec, eps1, t0):
    nt = a1.shape[0]
    rho0, v0, _ = rho_reduced(a1, g_of_t(t0, eps1))
    y0 = np.concatenate([[np.log(rho0) + 1e-3], np.log(np.abs(v0)), np.log(t0)])
    grad = np.zeros(1 + 2 * nt)
    grad[0] = 1.0
    zero_hess = np.zeros((1 + 2 * nt, 1 + 2 * nt))
    return minimize(lambda y: y[0], y0, jac=lambda y: grad,
                    hess=lambda y: zero_hess, method="trust-constr",
                    constraints=gp_constraints(a1, s_mat, c_vec, eps1),
                    bounds=[(None, None)] * (1 + nt) + [(0.0, None)] * nt,
                    options={"gtol": 1e-14, "xtol": 1e-15, "maxiter": 20000,
                             "verbose": 0})


def random_simplex(rng, m, n):
    """A random point of the product of solid simplices."""
    d = rng.dirichlet(np.ones(n + 1), size=m)[:, :n]
    return d.ravel()


# ------------------------------------------------------------------- driver

if __name__ == "__main__":
    par = default_params()
    m, n, eps1, omv = par["m"], par["n"], par["eps1"], par["omega_v"]
    ac, a1, dc = build_a1(par)
    s_mat = np.loadtxt(S_CSV, delimiter=",")
    b1, _, _ = dose_budgets()             # weekly budget, as in the notebook
    c_vec = b1 / omv + s_mat.sum(axis=1)
    nn = m * n
    red = Reduced(a1, s_mat, b1, eps1, omv)

    print("--- the data and the reduction ---")
    print("A1 %s, min entry %.6e, min column sum %.6e"
          % (a1.shape, a1.min(), a1.sum(axis=0).min()))
    acd = ac.toarray()
    print("pairing of Ac: rows %.3e, columns %.3e (relative)"
          % (np.abs(acd[0::2] - acd[1::2]).max() / np.abs(acd).max(),
             np.abs(acd[:, 0::2] - acd[:, 1::2]).max() / np.abs(acd).max()))
    rho_full, _ = dominant(ngm_factored(par=par))
    print("rho(D(0) Ac) 32x32    = %.15f  (notebook 4.423068109190628)" % rho_full)
    print("rho(A1 diag(1)) 16x16 = %.15f  diff %.3e"
          % (rho_reduced(a1, np.ones(nn))[0],
             abs(rho_reduced(a1, np.ones(nn))[0] - rho_full)))

    print("\n--- rho at allocations that exhaust the budget ---")
    chi_unif = (b1 / s_mat.sum(axis=1))[:, None] * np.ones((m, n))
    t = 1.0 + (chi_unif / omv).ravel()
    print("  uniform per location, weekly budget: rho = %.10f, used/b1 = %s"
          % (rho_reduced(a1, g_of_t(t, eps1))[0],
             np.array2string((chi_unif * s_mat).sum(axis=1) / b1, precision=4)))

    what0 = np.full(nn, 1.0 / n)
    print("\n--- route A: Theorem 3.1 in normalized variables ---")
    ra = solve_route_a(red, what0)
    rho_a = rho_reduced(a1, g_of_t(red.t_of(ra.x), eps1))[0]
    print("status %d: %s" % (ra.status, ra.message))
    print("rho_A = %.10f   reported %.10f   diff %.3e"
          % (rho_a, RHO_REPORTED, abs(rho_a - RHO_REPORTED)))

    print("\n--- route B: the geometric program, equation (S4.2) ---")
    rb = solve_route_b(a1, s_mat, c_vec, eps1, red.t_of(what0))
    tb, theta_b = np.exp(rb.x[1 + nn:]), np.exp(rb.x[0])
    rho_b = rho_reduced(a1, g_of_t(tb, eps1))[0]
    print("status %d: %s" % (rb.status, rb.message))
    print("theta of the GP       = %.10f" % theta_b)
    print("rho(A1 diag(g(t_GP))) = %.10f   reported %.10f   diff %.3e"
          % (rho_b, RHO_REPORTED, abs(rho_b - RHO_REPORTED)))
    print("theta vs its own rho  = %.3e  (the Collatz-Wielandt inequality (S4.1) tight at the optimum)"
          % abs(theta_b - rho_b))
    print("rho_A vs rho_B        = %.3e" % abs(rho_a - rho_b))

    print("\n--- route C: eight random feasible starts, Section S2 ---")
    rng = np.random.default_rng(20260921)
    vals, chis = [], []
    for s in range(8):
        r = solve_route_a(red, random_simplex(rng, m, n))
        vals.append(rho_reduced(a1, g_of_t(red.t_of(r.x), eps1))[0])
        chis.append(omv * (red.t_of(r.x) - 1.0))
    vals, chis = np.array(vals), np.array(chis)
    print("  min %.12f  max %.12f  spread %.3e"
          % (vals.min(), vals.max(), vals.max() - vals.min()))
    print("  max |chi1 - chi1_A| over the starts = %.3e  (max chi1 %.3e)"
          % (np.abs(chis - omv * (red.t_of(ra.x) - 1.0)).max(), chis.max()))

    print("\n--- the optimality conditions (3.21) at the route A point ---")
    ratio, ta, rho_at = red.marginal(ra.x)
    what = ra.x.reshape(m, n)
    print("  max relative slack in the dose constraints: %.3e"
          % np.abs((s_mat * omv * (ta.reshape(m, n) - 1.0)).sum(axis=1) / b1 - 1).max())
    for i in range(m):
        free = what[i] > 1e-8          # what is normalized, so this is relative
        mu = ratio[i, free].mean()
        print("    location %d: mu = %.6e, spread over free groups %.2e,"
              " at the bound: %s, their ratio/mu = %s"
              % (i, mu, np.abs(ratio[i, free] / mu - 1).max(),
                 np.where(~free)[0].tolist(),
                 np.array2string(ratio[i, ~free] / mu, precision=4)))

    print("\n--- the minimizer ---")
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    chi1 = omv * (ta.reshape(m, n) - 1.0)
    print("chi1 =\n%s" % chi1)
    print("doses per group =\n%s" % (chi1 * s_mat))
    print("max |t_B - t_A| / max|t_A| = %.3e"
          % (np.abs(tb - ta).max() / np.abs(ta).max()))
    zero = np.zeros((m, n))
    print("rho of the 32x32 NGM at chi1 = %.10f" % dominant(ngm_factored(chi1, zero, zero, par=par))[0])

    print("\n--- John's method: l0 scaled in each location to spend b1 ---")
    _, l0 = john_method(ngm_factored(par=par), par)
    chi_john = (b1 / (l0 * s_mat).sum(axis=1))[:, None] * l0
    rho_john = dominant(ngm_factored(chi_john, zero, zero, par=par))[0]
    print("chi1 =\n%s" % chi_john)
    print("used/b1 = %s" % np.array2string((chi_john * s_mat).sum(axis=1) / b1, precision=10))
    print("rho John = %.10f   minimum %.10f   minimum lower by %.2f per cent"
          % (rho_john, rho_a, 100 * (rho_john - rho_a) / rho_john))
