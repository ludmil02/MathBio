"""Three doses, 51 x 16 data at t = 347, doses b_{k,i} read from an Excel file.

The doses b_{k,i} are read from DOSES_XLSX (or the path given as the second
argument): sheet "week1", one row per location in the order of ngm_seir7.STATES,
columns "state", "b1", "b2", "b3", the doses of each type available at the
location in one WEEK.  They are divided by 7, since chi is a rate per day.
Every b_{k,i} must be positive.

The real weekly doses by location are not available yet.  If the default file
does not exist, it is written with MOCK doses (write_mock_doses), built as in
the paper (rates = doses / eligible) from reference rates chi_ref, uniform over
ages and locations (CHI_REF below, per day):

  b_{k,i} = 7 sum_{p in I_i} sigma_{k,p} chi_ref_{k,p}  (doses per week).

The group sizes are MOCK as well, until the weekly states of the SEIR model are
available:

  sigma_{1,p} = S_p, sigma_{2,p} = V_p, sigma_{3,p} = P_p, the susceptible,
      vaccinated and partially immune people of group p at the disease free
      equilibrium with chi = chi_ref and population N_p (mN).

Problem (1.1) of notes/stratified_seir_ngm.tex with these sigma and b is solved
by alternating over chi1, chi3, chi2 (Section 4):
  chi1: fixed point iteration of Lemma 4.1, g = (1 - e) + e / t;
  chi3: the same, g = g0 / t (e = 1, A1 replaced by A1 diag(g0));
  chi2: in u_p = sigma_{2,p} chi_{2,p} / b_{2,i} (a simplex per location), gradient from
        (4.6)-(4.7), by the spectral projected gradient method of Birgin, Martinez and
        Raydan (argument "spg", the default), scipy's trust-constr ("tc"; an interior
        point method, it moves away from chi_{2,p} = 0 and does worse), or projected
        gradient with Armijo backtracking ("pg").
rho is logged after every step of every sweep, and every increase is counted.
The end points are checked by the KKT conditions (5.1) of the note (kkt_table):
with h_{k,p} = -(d rho / d chi_{k,p}) / sigma_{k,p} and, for each (k, i), mu_{k,i}
the mean of h over the groups that receive doses, (5.1) holds when h_{k,p} = mu_{k,i}
where chi_{k,p} > 0 and h_{k,p} <= mu_{k,i} where chi_{k,p} = 0.
rho is computed from A1 diag(g) (order N = 816) by Arnoldi; the final points are
checked against rho(NGM) of order 2N.

One dose: the same sigma_1, b_1 with chi_2 = chi_3 = 0, solved by the fixed point
iteration alone (the chi1 step), from the uniform allocation.

Allocations compared (all satisfy the constraints):
  uniform   chi_{k,p} = b_{k,i} / sum_{l in I_i} sigma_{k,l}, the same rate for all
            groups of a location (chi = chi_ref for the mock doses);
  left ev   chi_{k,p} sigma_{k,p} = b_{k,i} w_p / sum_{l in I_i} w_l, w the S rows of the
            left eigenvector of NGM at chi_ref (the alternative strategy of the paper);
  minimizer the alternating method, from several starts.

Usage:
    python3 -u three_doses_51x16_mock.py spg [doses.xlsx] | tee intermediate/three_doses_51x16_mock_spg.log
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.optimize import minimize, LinearConstraint, Bounds

import ngm_seir7 as S7

T0 = time.time()
# chi2 step: "spg" spectral projected gradient, "tc" trust-constr, "pg" projected gradient
CHI2 = sys.argv[1] if len(sys.argv) > 1 else "fp"
THETA = 0.5          # damping of the fixed point iteration (Section 5.2); 1 = undamped
# chi1 and chi3 steps: "fp" the fixed point iteration (Section 5.2), "spg" spectral projected
# gradient (as the chi2 step), "newton" Appendix C.3
STEP13 = "fp"
DOSES_XLSX = os.path.join(S7.HERE, "data", "doses_51x16_mock.xlsx")
DOSES = sys.argv[2] if len(sys.argv) > 2 else DOSES_XLSX
CHI_REF = (3.0e-3, 1.0 / 28.0, 1.0e-3)
m, n = S7.m, S7.n
N = m * n
om, omv, e1, e2, e3 = S7.om, S7.omv, S7.e1, S7.e2, S7.e3
Nflat = S7.mN.ravel()
_, _, PREF = S7.yz(np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n)))
dc = PREF.ravel()                                    # d_c
psq = np.tile(S7.psiI * S7.Qs, m)                    # psi_I q_s
r = np.tile(S7.Qm / S7.Qs, m)
C = S7.C
LOC = [np.arange(i * n, (i + 1) * n) for i in range(m)]


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


# ----------------------------------------------------------------------------- model
def pis(chi1, chi2, chi3):
    return chi1 / (omv + e2 * chi2), e2 * chi2 / om, 1.0 + e3 * chi3 / om


def g_of(chi1, chi2, chi3):
    p1, p2, p3 = pis(chi1, chi2, chi3)
    num = 1.0 + r * p1 * (1.0 - e1 + p2)
    den = 1.0 + p1 * (1.0 + p2 + p2 * p3)
    return num / den, num, den


def perron(g):
    """rho, v, w of A1 diag(g), A1 = diag(dc) C diag(psq)."""
    mv = lambda x: dc * (C @ (psq * g * x))
    rmv = lambda x: g * psq * (C.T @ (dc * x))
    op = spsl.LinearOperator((N, N), matvec=mv, rmatvec=rmv, dtype=float)
    v0 = np.ones(N)
    lam, v = spsl.eigs(op, k=1, which="LM", tol=1e-13, v0=v0)
    _, w = spsl.eigs(op.adjoint(), k=1, which="LM", tol=1e-13, v0=v0)
    v, w = np.real(v[:, 0]), np.real(w[:, 0])
    return float(np.real(lam[0])), v * np.sign(v.sum()), w * np.sign(w.sum())


def rho_of(chi):
    return perron(g_of(*chi)[0])[0]


def drho_dg(g):
    rho, v, w = perron(g)
    return rho, (psq * (C.T @ (dc * w))) * v / float(w @ v)      # (A1^T w)_p v_p / w^T v


def dg_dchi2(chi1, chi2, chi3):
    """(4.7) of the note."""
    p1, p2, p3 = pis(chi1, chi2, chi3)
    _, num, den = g_of(chi1, chi2, chi3)
    omp = omv + e2 * chi2
    alpha = 1.0 + p2 + p2 * p3
    dN = r * e2 * p1 * (1.0 / om - (1.0 - e1 + p2) / omp)
    dD = e2 * p1 * ((1.0 + p3) / om - alpha / omp)
    return (den * dN - num * dD) / den**2


def rho_ngm(chi):
    """rho(NGM), order 2N, from ngm_seir7."""
    Y, Z, _ = S7.yz(*(c.reshape(m, n) for c in chi))
    return S7.perron(S7.h_operator(Y, Z, S7.Qs, S7.Qm)[0])[0]


# ----------------------------------------------------------------------------- one block, t form
def solve_mu(h, sg, bi):
    """(4.11): the solution of sum_p sg_p max(1, sqrt(h_p / (mu sg_p))) = c_i with
    c_i = bi + sum_p sg_p.  Written with bi and the tail sums
    c_i - C_j = bi + sum_{l > j} sg_(l), so that groups with very large sg_p (very small
    slope s_p in the chi3 step) do not cause cancellation."""
    beta = h / sg
    o = np.argsort(beta)
    beta, s, rr = beta[o], sg[o], np.sqrt(h * sg)[o]
    q = len(beta)
    tail = np.concatenate([np.cumsum(s[::-1])[::-1], [0.0]])   # tail[j] = sum_{l >= j} s_l
    B = np.concatenate([np.cumsum(rr[::-1])[::-1], [0.0]])
    j = 0
    for jj in range(1, q):
        # F(beta_(jj-1)) >= c_i  <=>  B_jj / sqrt(beta_(jj-1)) - tail_jj >= bi
        if B[jj] / np.sqrt(beta[jj - 1]) - tail[jj] >= bi:
            j = jj
    return (B[j] / (bi + tail[j])) ** 2


def chi2_form(chi):
    """Section 6.3 of the notes: g_p = (kappa + kappa' chi2) / (lambda + lambda' chi2),
    Delta_p = kappa lambda' - kappa' lambda; on the groups with Delta_p > 0,
    g_p = g0_p ((1 - e_p) + e_p / t_p), t_p = 1 + (lambda' / lambda) chi2.
    Returns Delta, g0, e, slope = lambda' / lambda."""
    c1, _, c3 = chi
    ka = omv + r * (1.0 - e1) * c1
    kp = e2 * (1.0 + r * c1 / om)
    la = omv + c1
    lp = e2 * (1.0 + (2.0 + e3 * c3 / om) * c1 / om)
    Delta = ka * lp - kp * la
    return Delta, ka / la, Delta / (ka * lp), lp / la


def fp_step(chi, k, b, sig, tol=1e-12, kmax=300, hist=None, label=""):
    """The step for dose k by the damped fixed point iteration (t_block), falling back to
    spg (block_spg) if an assumption of the notes fails: C not irreducible, or (k = 1) a
    location without a group with Delta_p > 0, or the iteration does not converge in kmax
    iterations.  Returns the new chi, the number of iterations and the method used."""
    reason = None
    if not S7.IRREDUCIBLE:
        reason = "C is not irreducible"
    elif k == 1 and not (chi2_form(chi)[0] > 0).reshape(m, n).any(axis=1).all():
        reason = "a location has no group with Delta_p > 0"
    if reason is None:
        out = {}
        new, it = t_block(chi, k, b, sig, tol=tol, kmax=kmax, hist=hist, out=out)
        if out["converged"]:
            return new, it, "fp"
        reason = "the fixed point iteration did not converge in %d iterations" % kmax
    say("%s dose %d step: %s; spg instead" % (label, k + 1, reason))
    new, it = block_spg(chi, k, b, sig)
    return new, it, "spg"


def t_block(chi, k, b, sig, tol=1e-12, kmax=300, hist=None, theta=None, out=None):
    """Minimize over chi_k (k = 0, 1, 2: first, second, third dose), the others fixed,
    by the damped fixed point iteration of Section 5.2 of the notes:
    t <- t + theta (t(mu) - t), theta = THETA (1/2) unless given; theta = 1 is the
    undamped iteration.  Stops when |t(mu) - t| / |t| < tol.  Returns the new chi and
    the number of iterations.  If hist is a list, rho at each iterate is appended.
    k = 1 (Section 6.3): the groups with Delta_p <= 0 get no second doses, and the
    iteration runs over the groups with Delta_p > 0; every location must have one."""
    theta = THETA if theta is None else theta
    chi = [c.copy() for c in chi]
    c1, c2, c3 = chi
    if k == 1:
        Delta, g0, e, slope = chi2_form(chi)
        slope = np.where(Delta > 0, slope, 0.0)               # the iteration runs where slope > 0
        coef = lambda gd: g0 * e * gd                         # h_p = g0_p e_p (dr/dg)_p
    elif k == 0:
        p2 = e2 * c2 / om
        omp = omv + e2 * c2
        alpha = 1.0 + 2.0 * e2 * c2 / om + e2 * e3 * c2 * c3 / om**2
        slope = alpha / omp                                   # t = 1 + slope chi1
        e = 1.0 - r / alpha * (1.0 - e1 + p2)
        coef = lambda gd: e * gd                              # h_p = e_p (dr/dg)_p
    else:
        p1, p2, _ = pis(c1, c2, np.zeros(N))
        D0 = 1.0 + p1 * (1.0 + 2.0 * p2)
        slope = e3 * p1 * p2 / (om * D0)                      # t = D / D0
        g0 = g_of(c1, c2, np.zeros(N))[0]
        coef = lambda gd: g0 * gd                             # h_p = g0_p (dr/dg)_p
    act = slope > 0
    tval = 1.0 + slope * chi[k]
    for it in range(kmax):
        rcur, gd = drho_dg(g_of(*chi)[0])
        if hist is not None:
            hist.append(rcur)
        h = coef(gd)
        tn = tval.copy()
        for i, I in enumerate(LOC):
            I = I[act[I]]
            if b[k][i] <= 0 or I.size == 0:
                continue
            st = sig[k][I] / slope[I]
            mu = solve_mu(h[I], st, b[k][i])
            tn[I] = np.maximum(1.0, np.sqrt(h[I] / (mu * st)))
        d = np.abs(tn - tval).max() / tval.max()
        tval = tval + theta * (tn - tval)
        chi[k] = np.where(act, (tval - 1.0) / np.where(act, slope, 1.0), 0.0)
        if d < tol:
            break
    if out is not None:
        out["converged"] = bool(d < tol)
    return chi, it + 1


# ----------------------------------------------------------------------------- chi2 block, simplex form
def proj_simplex(y):
    """Euclidean projection onto {u >= 0, sum u = 1}."""
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - 1.0
    j = np.nonzero(u - css / np.arange(1, y.size + 1) > 0)[0][-1]
    return np.maximum(y - css[j] / (j + 1), 0.0)


def proj(y):
    return np.concatenate([proj_simplex(y[I]) for I in LOC])


def chi2_grad_u(chi, b2, sig2):
    rho, gd = drho_dg(g_of(*chi)[0])
    return rho, gd * dg_dchi2(*chi) * np.repeat(b2, n) / sig2


def chi2_block(chi, b, sig, iters=40, s0=1e-2):
    b2, sig2 = np.repeat(b[1], n), sig[1]
    u = sig2 * chi[1] / b2
    to_chi = lambda uu: [chi[0], uu * b2 / sig2, chi[2]]
    rho, gu = chi2_grad_u(chi, b[1], sig2)
    s = s0
    for it in range(iters):
        while True:
            un = proj(u - s * gu)
            rn = rho_of(to_chi(un))
            if rn <= rho + 1e-4 * float(gu @ (un - u)) or s < 1e-14:
                break
            s *= 0.5
        if np.abs(un - u).max() < 1e-13:
            break
        u = un
        rho, gu = chi2_grad_u(to_chi(u), b[1], sig2)
        s *= 2.0
    return to_chi(u), s


def con_matrix(nblocks):
    """Sparse rows: sum_{p in I_i} u_p = 1 for each location and each block."""
    rows = np.repeat(np.arange(nblocks * m), n)
    return sps.csr_matrix((np.ones(nblocks * N), (rows, np.arange(nblocks * N))), shape=(nblocks * m, nblocks * N))


def dg_all(chi):
    """dg_p / dchi_{k,p}, k = 1, 2, 3, by (4.6)-(4.7)."""
    p1, p2, p3 = pis(*chi)
    _, num, den = g_of(*chi)
    omp = omv + e2 * chi[1]
    alpha = 1.0 + p2 + p2 * p3
    e = 1.0 - r / alpha * (1.0 - e1 + p2)
    return [-alpha * e / (omp * den**2), dg_dchi2(*chi), -e3 * p1 * p2 * num / (om * den**2)]


def chi2_block_tc(chi, b, sig, maxiter=300):
    """The chi2 step by scipy's trust-constr in u_p = sigma_{2,p} chi_{2,p} / b_{2,i}."""
    b2 = np.repeat(b[1], n)
    to_chi = lambda uu: [chi[0], uu * b2 / sig[1], chi[2]]

    def f(u):
        ch = to_chi(u)
        rho, gd = drho_dg(g_of(*ch)[0])
        return rho, gd * dg_dchi2(*ch) * b2 / sig[1]

    lc = LinearConstraint(con_matrix(1), np.ones(m), np.ones(m))
    res = minimize(f, sig[1] * chi[1] / b2, jac=True, method="trust-constr", constraints=[lc],
                   bounds=Bounds(np.zeros(N), np.inf),
                   options=dict(gtol=1e-12, xtol=1e-14, maxiter=maxiter))
    u = np.maximum(res.x, 0.0)
    u = proj(u)                                   # exact feasibility
    return to_chi(u), res.nit


def spg(fg, f, u, P, maxiter=200, tol=1e-12, M=10, amin=1e-12, amax=1e12):
    """Spectral projected gradient (Birgin, Martinez, Raydan 2000, SPG2): minimize f over
    the set onto which P projects.  fg(u) -> (f, grad); f(u) -> f.  Barzilai-Borwein
    step, nonmonotone Armijo line search over the last M values.  Returns the best
    point seen, its value, the number of iterations and the last |P(u - g) - u|_inf."""
    fu, g = fg(u)
    hist = [fu]
    best = (fu, u)
    a = 1.0 / max(np.abs(P(u - g) - u).max(), 1e-30)
    for it in range(maxiter):
        pg = np.abs(P(u - g) - u).max()
        if pg < tol:
            break
        d = P(u - a * g) - u
        gd = float(g @ d)
        fmax = max(hist[-M:])
        lam = 1.0
        while True:
            un = u + lam * d
            fn = f(un)
            if fn <= fmax + 1e-4 * lam * gd or lam < 1e-12:
                break
            lam *= 0.5
        fn, gn = fg(un)
        s, y = un - u, gn - g
        sy = float(s @ y)
        a = amax if sy <= 0 else min(amax, max(amin, float(s @ s) / sy))
        u, g, fu = un, gn, fn
        hist.append(fu)
        if fu < best[0]:
            best = (fu, u)
    return best[1], best[0], it + 1, np.abs(P(u - g) - u).max()


def chi2_block_spg(chi, b, sig, maxiter=500):
    """The chi2 step by spg in u_p = sigma_{2,p} chi_{2,p} / b_{2,i}."""
    b2 = np.repeat(b[1], n)
    to_chi = lambda uu: [chi[0], uu * b2 / sig[1], chi[2]]

    def fg(u):
        ch = to_chi(u)
        rho, gd = drho_dg(g_of(*ch)[0])
        return rho, gd * dg_dchi2(*ch) * b2 / sig[1]

    u, _, it, _ = spg(fg, lambda u: rho_of(to_chi(u)), sig[1] * chi[1] / b2, proj, maxiter=maxiter)
    return to_chi(u), it


def block_spg(chi, k, b, sig, maxiter=500):
    """The step for dose k (0, 1, 2 for chi1, chi2, chi3) by spg in
    u_p = sigma_{k,p} chi_{k,p} / b_{k,i}, the other doses fixed."""
    bk = np.repeat(b[k], n)

    def to_chi(u):
        ch = list(chi)
        ch[k] = u * bk / sig[k]
        return ch

    def fg(u):
        ch = to_chi(u)
        rho, gd = drho_dg(g_of(*ch)[0])
        return rho, gd * dg_all(ch)[k] * bk / sig[k]

    u, _, it, _ = spg(fg, lambda u: rho_of(to_chi(u)), sig[k] * chi[k] / bk, proj, maxiter=maxiter)
    return to_chi(u), it


def joint_spg(chi, b, sig, maxiter=3000):
    """All three doses at once by spg in u_{k,p} = sigma_{k,p} chi_{k,p} / b_{k,i}."""
    bb = [np.repeat(b[k], n) for k in range(3)]
    to_chi = lambda U: [U[k * N:(k + 1) * N] * bb[k] / sig[k] for k in range(3)]
    P3 = lambda U: np.concatenate([proj(U[k * N:(k + 1) * N]) for k in range(3)])

    def fg(U):
        ch = to_chi(U)
        rho, gd = drho_dg(g_of(*ch)[0])
        dg = dg_all(ch)
        return rho, np.concatenate([gd * dg[k] * bb[k] / sig[k] for k in range(3)])

    U0 = np.concatenate([sig[k] * chi[k] / bb[k] for k in range(3)])
    U, _, it, pg = spg(fg, lambda U: rho_of(to_chi(U)), U0, P3, maxiter=maxiter)
    return to_chi(U), it, pg


def joint_tc(chi, b, sig, maxiter=3000):
    """All three doses at once by trust-constr in u_{k,p} = sigma_{k,p} chi_{k,p} / b_{k,i}."""
    bb = [np.repeat(b[k], n) for k in range(3)]
    to_chi = lambda U: [U[k * N:(k + 1) * N] * bb[k] / sig[k] for k in range(3)]

    def f(U):
        ch = to_chi(U)
        rho, gd = drho_dg(g_of(*ch)[0])
        dg = dg_all(ch)
        return rho, np.concatenate([gd * dg[k] * bb[k] / sig[k] for k in range(3)])

    lc = LinearConstraint(con_matrix(3), np.ones(3 * m), np.ones(3 * m))
    U0 = np.concatenate([sig[k] * chi[k] / bb[k] for k in range(3)])
    res = minimize(f, U0, jac=True, method="trust-constr", constraints=[lc],
                   bounds=Bounds(np.zeros(3 * N), np.inf),
                   options=dict(gtol=1e-12, xtol=1e-14, maxiter=maxiter))
    U = np.concatenate([proj(np.maximum(res.x[k * N:(k + 1) * N], 0.0)) for k in range(3)])
    return to_chi(U), res.nit, res.status


def kkt_residual(chi, b, sig):
    """max over doses and groups of |u - P(u - grad_u)|, u_p = sigma_{k,p} chi_{k,p} / b_{k,i},
    with grad_u scaled by the sum of |grad_u| over the location.  Zero at a KKT point."""
    _, gd = drho_dg(g_of(*chi)[0])
    p1, p2, p3 = pis(*chi)
    _, num, den = g_of(*chi)
    omp = omv + e2 * chi[1]
    alpha = 1.0 + p2 + p2 * p3
    e = 1.0 - r / alpha * (1.0 - e1 + p2)
    dg = [-alpha * e / (omp * den**2), dg_dchi2(*chi), -e3 * p1 * p2 * num / (om * den**2)]
    out = []
    for k in range(3):
        bb = np.repeat(b[k], n)
        u = sig[k] * chi[k] / bb
        gu = gd * dg[k] * bb / sig[k]
        scale = np.repeat([np.abs(gu[I]).sum() for I in LOC], n)
        gu = gu / scale
        out.append(np.abs(u - proj(u - gu)).max())
    return out


def alternate(chi, b, sig, sweeps=200, tol=1e-11, label=""):
    """The alternating method.  Returns chi, rho and the history of rho: the start,
    then the value after each of the steps chi1, chi3, chi2 of every sweep."""
    rho = rho_of(chi)
    hist = [rho]
    say("%s start: rho = %.12f" % (label, rho))
    for sw in range(sweeps):
        if STEP13 == "newton":                         # Appendix C.3
            chi, _, inf1 = newton(chi, b, sig, doses=(0,), second_order=False, verbose=False)
            it1 = inf1["steps"]
        elif STEP13 == "spg":
            chi, it1 = block_spg(chi, 0, b, sig)
        else:
            chi, it1, _ = fp_step(chi, 0, b, sig, label="%s sweep %d" % (label, sw))
        r1 = rho_of(chi)
        if STEP13 == "newton":
            chi, _, inf3 = newton(chi, b, sig, doses=(2,), second_order=False, verbose=False)
            it3 = inf3["steps"]
        elif STEP13 == "spg":
            chi, it3 = block_spg(chi, 2, b, sig)
        else:
            chi, it3, _ = fp_step(chi, 2, b, sig, label="%s sweep %d" % (label, sw))
        r3 = rho_of(chi)
        if CHI2 == "fp":                               # Section 6.3
            chi, it2, _ = fp_step(chi, 1, b, sig, label="%s sweep %d" % (label, sw))
        elif CHI2 == "spg":
            chi, it2 = chi2_block_spg(chi, b, sig)
        elif CHI2 == "tc":
            chi, it2 = chi2_block_tc(chi, b, sig)
        else:
            chi, _ = chi2_block(chi, b, sig)
            it2 = 40
        r2 = rho_of(chi)
        hist += [r1, r3, r2]
        say("%s sweep %3d: after chi1 %.12f (%d it), chi3 %.12f (%d it), chi2 %.12f (%d it)"
            % (label, sw, r1, it1, r3, it3, r2, it2))
        if abs(rho - r2) < tol * r2:
            break
        rho = r2
    inc = np.diff(hist) / np.array(hist[1:])
    say("%s end after %d sweeps: rho = %.12f; steps that increase rho by more than 1e-12 (relative): %d of %d"
        " (largest relative increase %.1e)"
        % (label, sw + 1, r2, int((inc > 1e-12).sum()), inc.size, max(inc.max(), 0.0)))
    return chi, r2, np.array(hist)


def kkt_table(chi, b, sig, doses=(0, 1, 2), thr=1e-9):
    """The KKT conditions (5.1): for each dose k and location i, h_p = -(drho/dchi_{k,p}) / sigma_{k,p},
    mu = mean of h over the groups with u_p = sigma_{k,p} chi_{k,p} / b_{k,i} > thr.
    Returns, per dose, the largest over i of: spread max |h_p / mu - 1| over those groups,
    ratio max h_p / mu over the other groups (-inf if none), and |sum_p u_p - 1|;
    also the number of groups that receive no doses."""
    _, gd = drho_dg(g_of(*chi)[0])
    dg = dg_all(chi)
    out = {}
    for k in doses:
        h = -gd * dg[k] / sig[k]
        sp, ra, dr, nz = 0.0, -np.inf, 0.0, 0
        for i, I in enumerate(LOC):
            u = sig[k][I] * chi[k][I] / b[k][i]
            pos = u > thr
            mu = h[I][pos].mean()
            sp = max(sp, np.abs(h[I][pos] / mu - 1.0).max())
            if (~pos).any():
                ra = max(ra, (h[I][~pos] / mu).max())
                nz += int((~pos).sum())
            dr = max(dr, abs(u.sum() - 1.0))
        out[k] = (sp, ra, dr, nz)
        say("  KKT dose %d: spread %.1e, ratio where no doses %s (%d groups), dose residual %.1e"
            % (k + 1, sp, "%.3f" % ra if np.isfinite(ra) else "-", nz, dr))
    return out


# ----------------------------------------------------------------------------- Newton, Appendix C
def g_derivs(chi):
    """g, its gradient, shape (3, N), and Hessian, shape (3, 3, N), in (chi1, chi2, chi3),
    group by group.  g = Nt / Dt with (numerator and denominator of (2.5) times omega_p)
      Nt = omv + e2 c2 + r a c1 + (r e2 / om) c1 c2,
      Dt = omv + e2 c2 + c1 + (2 e2 / om) c1 c2 + (e2 e3 / om^2) c1 c2 c3,   a = 1 - eps1."""
    c1, c2, c3 = chi
    a = 1.0 - e1
    q = e2 * e3 / om**2
    Nt = omv + e2 * c2 + r * a * c1 + r * e2 / om * c1 * c2
    Dt = omv + e2 * c2 + c1 + 2.0 * e2 / om * c1 * c2 + q * c1 * c2 * c3
    z = np.zeros(N)
    dN = np.array([r * a + r * e2 / om * c2, e2 + r * e2 / om * c1, z])
    dD = np.array([1.0 + 2.0 * e2 / om * c2 + q * c2 * c3, e2 + 2.0 * e2 / om * c1 + q * c1 * c3, q * c1 * c2])
    HN = np.zeros((3, 3, N))
    HN[0, 1] = HN[1, 0] = r * e2 / om
    HD = np.zeros((3, 3, N))
    HD[0, 1] = HD[1, 0] = 2.0 * e2 / om + q * c3
    HD[0, 2] = HD[2, 0] = q * c2
    HD[1, 2] = HD[2, 1] = q * c1
    g = Nt / Dt
    dg = (dN - g * dD) / Dt
    Hg = (HN - g * HD - dg[:, None] * dD[None, :] - dD[:, None] * dg[None, :]) / Dt
    return g, dg, Hg


A1 = lambda x: dc * (C @ (psq * x))
A1T = lambda y: psq * (C.T @ (dc * y))


class Second:
    """rho, its gradient and Hessian-vector products in x = (chi1, chi2, chi3), shape (3, N),
    by Lemmas C.2-C.3 and (C.3).  Products with S and S^T are solves with the bordered
    matrices B, B' of Lemma C.2 by GMRES; nothing of order N is formed."""

    def __init__(self, chi, gtol=1e-12):
        self.chi = np.array(chi, dtype=float)
        self.g, self.dg, self.Hgx = g_derivs(self.chi)
        rho, v, w = perron(self.g)
        w = w / (w @ v)
        self.rho, self.v, self.w = rho, v, w
        self.a = A1T(w)
        self.grad_g = self.a * v                          # (3.3) with w^T v = 1
        self.grad = self.dg * self.grad_g                 # (3, N)
        self.gtol = gtol
        g = self.g
        self.B = spsl.LinearOperator((N + 1, N + 1), dtype=float,
                                     matvec=lambda z: np.append(rho * z[:N] - A1(g * z[:N]) + v * z[N], w @ z[:N]))
        self.Bt = spsl.LinearOperator((N + 1, N + 1), dtype=float,
                                      matvec=lambda z: np.append(rho * z[:N] - g * A1T(z[:N]) + w * z[N], v @ z[:N]))
        self.solves = 0

    def _solve(self, op, rhs):
        sol, info = spsl.gmres(op, np.append(rhs, 0.0), tol=self.gtol, atol=0.0, restart=200, maxiter=50)
        if info != 0:
            raise RuntimeError("GMRES for the bordered system did not converge (info %d)" % info)
        self.solves += 1
        return sol[:N]

    def S(self, x):
        return self._solve(self.B, x - self.v * (self.w @ x))

    def ST(self, x):
        return self._solve(self.Bt, x - self.w * (self.v @ x))

    def hess_g(self, y):
        """Lemma C.3: D_a S A1 D_v y + D_v A1^T S^T D_a y."""
        return self.a * self.S(A1(self.v * y)) + self.v * A1T(self.ST(self.a * y))

    def hess(self, z):
        """(C.3): J^T H_g J z + sum_p (d rho / d g_p) Hess g_p z, z of shape (3, N)."""
        u = self.hess_g((self.dg * z).sum(axis=0))
        return self.dg * u + self.grad_g * np.einsum("kln,ln->kn", self.Hgx, z)


def _blocks(x):
    return x.reshape(3, m, n)


def proj_free(z, mask, sig):
    """P_F of Appendix C.2: z zero outside the free set, minus its sigma weighted mean,
    location by location (rows of E_F have disjoint supports)."""
    z = np.where(mask, z, 0.0)
    e = np.where(mask, np.array(sig), 0.0)
    eb, zb = _blocks(e), _blocks(z)
    den = (eb * eb).sum(axis=2)                       # 0 for a dose with no free variable (fixed dose)
    coef = np.divide((eb * zb).sum(axis=2), den, out=np.zeros_like(den), where=den > 0)
    return z - (eb * coef[:, :, None]).reshape(3, N)


ONES = np.ones((3, N))


def pcg(hess, mask, rhs, tol, maxit=2000):
    """Projected CG for P H P d = rhs on the range of P_F, in the variables u (the
    constraints are sum_p u = 1 per location, so P_F removes the mean over the free
    groups).  Stops at relative residual tol.  Returns (d, "newton", iterations) or, if a
    direction of negative curvature is met, (direction, "negative curvature", iterations)."""
    d = np.zeros((3, N))
    res = proj_free(rhs, mask, ONES)
    p = res.copy()
    rr = (res * res).sum()
    r0 = np.sqrt(rr)
    for it in range(maxit):
        Hp = proj_free(hess(p), mask, ONES)
        pHp = (p * Hp).sum()
        if pHp <= 0.0:
            return (p if it == 0 else d), "negative curvature", it + 1
        alpha = rr / pHp
        d += alpha * p
        res -= alpha * Hp
        rr_new = (res * res).sum()
        if np.sqrt(rr_new) <= tol * r0:
            return d, "newton", it + 1
        p = res + (rr_new / rr) * p
        rr = rr_new
    return d, "newton (CG limit)", maxit


def multipliers(grad_u, mask):
    """In the variables u: mu = -(mean of grad_u over the free groups of each location),
    shape (3, m), and nu = grad_u + mu, which has the sign of nu of (4.1) (nu_u = (b / sigma) nu)."""
    gb = _blocks(np.where(mask, grad_u, 0.0))
    cnt = _blocks(mask.astype(float)).sum(axis=2)
    mu = -np.divide(gb.sum(axis=2), cnt, out=np.zeros_like(cnt), where=cnt > 0)
    nu = grad_u + np.repeat(mu, n, axis=1)
    return mu, nu


def smallest_reduced_eig(hess, mask, tol=1e-6, maxiter=5000):
    """Smallest eigenvalue and eigenvector of P_F H P_F on the range of P_F, by Lanczos,
    with the complement of the range shifted to the largest eigenvalue.  Returns
    (lambda, vector, converged); if Lanczos does not converge, the best partial result."""
    def pHp(z):
        pz = proj_free(z.reshape(3, N), mask, ONES)
        return proj_free(hess(pz), mask, ONES)
    op = spsl.LinearOperator((3 * N, 3 * N), matvec=lambda z: pHp(z).ravel(), dtype=float)
    lmax = spsl.eigsh(op, k=1, which="LA", tol=1e-3, maxiter=500, return_eigenvectors=False)[0]

    def mv(z):
        z = z.reshape(3, N)
        pz = proj_free(z, mask, ONES)
        return (pHp(pz) + lmax * (z - pz)).ravel()
    op = spsl.LinearOperator((3 * N, 3 * N), matvec=mv, dtype=float)
    try:
        lam, vec = spsl.eigsh(op, k=1, which="SA", tol=tol, maxiter=maxiter)
        conv = True
    except spsl.ArpackNoConvergence as err:
        if len(err.eigenvalues) == 0:
            return np.nan, np.zeros((3, N)), False, lmax
        lam, vec = err.eigenvalues, err.eigenvectors
        conv = False
    return float(lam[0]), proj_free(vec[:, 0].reshape(3, N), mask, ONES), conv, float(lmax)


def newton(chi, b, sig, thr=1e-12, maxit=30, gtol=1e-11, second_order=True, doses=(0, 1, 2),
           verbose=True, label=""):
    """Steps 2-4 of Appendix C.2 from the end point chi of the alternating method, in the
    variables u_{k,p} = sigma_{k,p} chi_{k,p} / b_{k,i} (chi = D u, D = b / sigma; the
    constraints are sum_{p in I_i} u_{k,p} = 1, the gradient is D grad_chi and the Hessian
    D H D).  CG stops at relative residual min(1e-2, |P grad| / |grad|).
    doses: the doses (0, 1, 2 for chi1, chi2, chi3) that are free; the others stay fixed.
    For doses = (0,) or (2,) this is the step of Appendix C.3; for (2,) the groups with
    s_p = 0 (chi1 = 0 or chi2 = 0) are fixed at 0 (Proposition B.1).
    Returns chi, rho, and a dict with the history, the number of Newton steps and the
    second order test."""
    say_ = say if verbose else (lambda *a: None)
    sg = np.array(sig)
    D = np.array([np.repeat(b[k], n) for k in range(3)]) / sg
    u = np.array(chi, dtype=float) / D
    act = np.zeros((3, N), dtype=bool)
    act[list(doses)] = True
    mask = (u > thr) & act
    if tuple(doses) == (2,):
        mask[2] &= (np.asarray(chi[0]) > 0) & (np.asarray(chi[1]) > 0)
    if not _blocks(mask)[list(doses)].any(axis=2).all():
        raise ValueError("a location has no free variable for one of the doses %s" % (doses,))
    u = np.where(mask | ~act, u, 0.0)
    ub = _blocks(u)
    for k in doses:                                   # restore sum u = 1 after zeroing
        ub[k] = ub[k] / ub[k].sum(axis=1)[:, None]
    u = ub.reshape(3, N)
    hist, info = [], {}
    steps = 0
    for outer in range(10):
        for it in range(maxit):
            sec = Second(list(D * u))
            hist.append(sec.rho)
            grad = D * sec.grad
            hess = lambda z: D * sec.hess(D * z)
            gP = proj_free(grad, mask, ONES)
            gn = np.sqrt((gP * gP).sum()) / np.sqrt((np.where(mask, grad, 0.0) ** 2).sum())
            if gn < gtol:
                break
            d, kind, cgit = pcg(hess, mask, -grad, tol=min(1e-2, gn))
            slope = (grad * d).sum()
            if slope > 0:
                d, slope = -d, -slope
            # largest step keeping the free variables non-negative, and the variable that blocks it
            neg = np.flatnonzero((mask & (d < 0)).ravel())
            ratios = -u.ravel()[neg] / d.ravel()[neg]
            tmax = ratios.min() if neg.size else np.inf
            if kind.startswith("newton"):
                tau = min(1.0, tmax)
            else:                                      # negative curvature: start at a step of the size of u
                tau = min(tmax, np.abs(u[mask]).max() / np.abs(d).max())
            for _ in range(40):
                ut = np.where(mask, u + tau * d, u)
                rnew = rho_of(list(D * ut))
                if rnew <= sec.rho + 1e-4 * tau * slope:
                    break
                tau *= 0.5
            if rnew > sec.rho:
                say_("%s newton %d: |P grad| / |grad| = %.1e; no decrease along the %s direction; stop"
                     % (label, it, gn, kind))
                break
            steps += 1
            u = np.maximum(ut, 0.0)
            left = ""
            if tau == tmax:                            # the blocking variable reaches 0 and leaves the free set
                j = neg[np.argmin(ratios)]
                u.ravel()[j] = 0.0
                mask.ravel()[j] = False
                left = ", one variable leaves the free set"
            say_("%s newton %d (%s, %d CG iterations): rho = %.15f, |P grad| / |grad| = %.1e, step %.2e%s"
                 % (label, it, kind, cgit, rnew, gn, tau, left))
        sec = Second(list(D * u))
        grad = D * sec.grad
        hess = lambda z: D * sec.hess(D * z)
        mu, nu = multipliers(grad, mask)
        scale = np.abs(grad[mask]).max()
        cand = ~mask & act
        if tuple(doses) == (2,):                      # groups with s_p = 0 stay out (Proposition B.1)
            cand[2] &= (np.asarray(chi[0]) > 0) & (np.asarray(chi[1]) > 0)
        add = cand & (nu < -1e-8 * scale)
        if add.any():
            say_("%s %d variables with nu < 0 return to the free set" % (label, int(add.sum())))
            mask = mask | add
            continue
        break
    gP = proj_free(grad, mask, ONES)
    gn = np.sqrt((gP * gP).sum()) / np.sqrt((np.where(mask, grad, 0.0) ** 2).sum())
    nuo = nu[~mask & act]
    info.update(rho=sec.rho, hist=np.array(hist), free=int(mask.sum()), grad_rel=gn, steps=steps,
                nu_min=float(nuo.min() / scale) if nuo.size else np.inf)
    say_("%s newton end: rho = %.15f, |P grad| / |grad| = %.1e, free variables %d of %d, "
         "smallest nu / |grad| over the others %.1e"
         % (label, sec.rho, gn, mask.sum(), int(act.sum()), info["nu_min"]))
    if second_order:
        lam, z, conv, lmax = smallest_reduced_eig(hess, mask)
        info.update(lam_min=lam, lam_max=lmax, lanczos_converged=conv, direction=z,
                    strict_local_min=bool(conv and lam > 0 and info["nu_min"] > 0))
        say("%s second order test: eigenvalues of P H P in u from %.3e to %.3e (Lanczos %s): %s"
            % (label, lam, lmax, "converged" if conv else "NOT converged, partial result",
               "strict local minimizer" if info["strict_local_min"] else "not shown"))
    return list(D * u), sec.rho, info


# ----------------------------------------------------------------------------- doses file
MOCK_WEEK_FACTORS = (1.0, 1.25, 1.5)          # week w: the doses of week 1 times this factor


def write_mock_doses(path, sig):
    """Write the mock weekly doses to path: sheet "week1" with
    7 sum_{p in I_i} sigma_{k,p} chi_ref_k, and sheets "week2", "week3" with these
    times MOCK_WEEK_FACTORS."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wk = {"b%d" % (k + 1): np.array([7.0 * sig[k][I].sum() * CHI_REF[k] for I in LOC]) for k in range(3)}
    notes = pd.DataFrame({"notes": [
        "MOCK doses, 51 x 16 data at t = 347; read by optimize_doses_51x16.py and",
        "three_doses_51x16_mock.py (sheet week1 only).",
        "One sheet per week, week1, week2, ...; one row per location (state), in the order",
        "of ngm_seir7.STATES.",
        "b1, b2, b3: first, second, third doses available at the location in that week.",
        "The code divides them by 7 (chi is a rate per day). Every entry must be positive",
        "(b2, b3 are not used in a one dose run).",
        "Mock values, week1: b_k = 7 sum_p sigma_{k,p} chi_ref_k, chi_ref = (3e-3, 1/28, 1e-3)",
        "per day, sigma_1, sigma_2, sigma_3 = S, V, P at the disease free equilibrium with",
        "chi = chi_ref; week2 and week3: week1 times 1.25 and 1.5.",
        "Optional sheets sigma_week1, ...: columns state, age, S, V, P (816 rows), the group",
        "sizes for that week; without them the mock S, V, P above are used.",
        "Replace the columns b1, b2, b3 with the real weekly doses, and add sheets for more weeks."]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for w, f in enumerate(MOCK_WEEK_FACTORS):
            doses = pd.DataFrame({"state": list(S7.STATES), **{c: f * v for c, v in wk.items()}})
            doses.to_excel(xw, sheet_name="week%d" % (w + 1), index=False)
        notes.to_excel(xw, sheet_name="notes", index=False)
    say("wrote mock doses to %s" % path)


def read_doses(path, sheet="week1", doses=(1, 2, 3)):
    """b_{k,i} per day from a sheet of path (weekly doses / 7); only the listed
    dose types must be positive."""
    df = pd.read_excel(path, sheet_name=sheet)
    if list(df["state"]) != list(S7.STATES):
        raise ValueError("%s: the column 'state' must list the 51 locations in the order %s"
                         % (path, " ".join(S7.STATES)))
    b = [df["b%d" % (k + 1)].to_numpy(dtype=float) / 7.0 for k in range(3)]
    for k in doses:
        if not np.all(b[k - 1] > 0):
            raise ValueError("%s, sheet %s: b%d must be positive at every location (zero doses are not handled)"
                             % (path, sheet, k))
    return b


# ----------------------------------------------------------------------------- main
def main():
    ref0 = [np.full(N, c) for c in CHI_REF]
    p1, p2, _ = pis(*ref0)
    _, _, den = g_of(*ref0)
    St = Nflat / den
    sig = [St, p1 * St, p2 * p1 * St]
    if not os.path.exists(DOSES) and DOSES == DOSES_XLSX:
        write_mock_doses(DOSES, sig)
    b = read_doses(DOSES)
    say("doses read from %s" % DOSES)
    # the uniform allocation: the same rate for all groups of a location
    ref = [np.concatenate([np.full(n, b[k][i] / sig[k][I].sum()) for i, I in enumerate(LOC)])
           for k in range(3)]
    say("uniform allocation vs chi_ref: max relative difference %.1e"
        % max(np.abs(ref[k] / CHI_REF[k] - 1.0).max() for k in range(3)))
    Ntot = Nflat.sum()
    say("chi_ref per day: %s" % ", ".join("%.4e" % c for c in CHI_REF))
    say("at chi_ref: S, V, P are %.3f, %.3f, %.3f of the population"
        % tuple(s.sum() / Ntot for s in sig))
    say("doses per day, US total: b1 %.4e, b2 %.4e, b3 %.4e (per 1000 people: %.3f, %.3f, %.3f)"
        % (*(bb.sum() for bb in b), *(1e3 * bb.sum() / Ntot for bb in b)))
    say("r_p in [%.3f, %.3f]" % (r.min(), r.max()))

    # checks of g and of (4.7)
    rng = np.random.default_rng(1)
    chi_r = [x * rng.random(N) * 2 for x in CHI_REF]
    Y, Z, _ = S7.yz(*(c.reshape(m, n) for c in chi_r))
    gchk = (Y + r.reshape(m, n) * Z).ravel() / dc
    say("check g vs ngm_seir7.yz: max diff %.1e" % np.abs(gchk - g_of(*chi_r)[0]).max())
    hh = 1e-20
    cs = (g_of(chi_r[0] + 0j, chi_r[1] + 1j * hh, chi_r[2] + 0j)[0].imag) / hh
    say("check (4.7) vs complex step: max rel diff %.1e"
        % (np.abs(cs - dg_dchi2(*chi_r)).max() / np.abs(cs).max()))

    rho0 = rho_ngm([np.zeros(N)] * 3)
    say("rho(NGM) at chi = 0: %.12f" % rho0)

    # left eigenvector allocation, S rows of the NGM at chi_ref
    Yr, Zr, _ = S7.yz(*(c.reshape(m, n) for c in ref))
    H = S7.h_operator(Yr, Zr, S7.Qs, S7.Qm)[0]
    wS = np.abs(S7.perron(H, transpose=True)[2].reshape(N, 2)[:, 0])
    lev = []
    for k in range(3):
        x = np.empty(N)
        for i, I in enumerate(LOC):
            x[I] = b[k][i] * wS[I] / wS[I].sum() / sig[k][I]
        lev.append(x)

    starts = [("uniform", ref), ("left ev", lev)]
    for s in range(3):
        rs = []
        for k in range(3):
            x = np.empty(N)
            for i, I in enumerate(LOC):
                u = rng.random(n)
                x[I] = b[k][i] * u / u.sum() / sig[k][I]
            rs.append(x)
        starts.append(("random %d" % s, rs))

    say("chi2 step: %s" % {"fp": "damped fixed point iteration (Section 6.3)",
                            "spg": "spectral projected gradient", "tc": "trust-constr",
                            "pg": "projected gradient"}[CHI2])
    # ------------------------------------------------------------------ one dose
    say("ONE DOSE: sigma_1, b_1 as above, chi_2 = chi_3 = 0")
    z0 = np.zeros(N)
    lev1 = [lev[0], z0, z0]
    h1 = []
    c1, it1 = t_block([ref[0], z0, z0], 0, b, sig, tol=1e-13, kmax=1000, hist=h1)
    h1.append(rho_of(c1))
    h1 = np.array(h1)
    inc = np.diff(h1) / h1[1:]
    say("one dose: fixed point iteration from the uniform allocation, %d iterations, rho = %.12f;"
        " iterations that increase rho by more than 1e-12 (relative): %d of %d (largest relative increase %.1e)"
        % (it1, h1[-1], int((inc > 1e-12).sum()), inc.size, max(inc.max(), 0.0)))
    kkt1 = kkt_table(c1, b, sig, doses=(0,))
    r1min = rho_ngm(c1)
    say("one dose: rho(NGM) of the allocations")
    one = {}
    for name, ch in (("chi = 0", [z0] * 3), ("uniform", [ref[0], z0, z0]), ("left ev", lev1),
                     ("minimizer", c1)):
        rr = rho_ngm(ch)
        one[name] = rr
        say("  %-10s rho = %.6f   decrease from chi = 0: %6.2f%%   above the minimizer: %6.3f%%"
            % (name, rr, 100 * (rho0 - rr) / rho0, 100 * (rr - r1min) / r1min))
    say("one dose: groups with no first doses at the minimizer: %d of %d; by age:"
        % (int((sig[0] * c1[0] / np.repeat(b[0], n) <= 1e-9).sum()), N))
    print("    " + " ".join("%s:%d" % (a_, int((sig[0] * c1[0] / np.repeat(b[0], n) <= 1e-9)
                                              .reshape(m, n)[:, j].sum()))
                            for j, a_ in enumerate(S7.AGES)), flush=True)
    say("one dose: first doses by age, US total, per day (uniform / left ev / minimizer)")
    for j, a_ in enumerate(S7.AGES):
        vals = [(sig[0] * ch[0]).reshape(m, n)[:, j].sum() for ch in ([ref[0]], lev1, c1)]
        print("    %-6s %9.0f %9.0f %9.0f" % (a_, *vals), flush=True)

    # ------------------------------------------------------------------ three doses
    say("THREE DOSES")
    results = []
    hists = {}
    for name, ch in starts:
        cm, rm, hh_ = alternate([c.copy() for c in ch], b, sig, label="[%s]" % name)
        kkt_table(cm, b, sig)
        hists[name] = hh_
        results.append((name, cm, rm))
    best = min(results, key=lambda x: x[2])
    say("minimizers from the %d starts: rho in [%.12f, %.12f]"
        % (len(results), min(x[2] for x in results), max(x[2] for x in results)))
    for name, cm, rm in results:
        say("  %-9s rho %.12f   max_k max|chi_k - best| / max best = %.1e"
            % (name, rm, max(np.abs(cm[k] - best[1][k]).max() / best[1][k].max() for k in range(3))))

    chi_min = best[1]
    rmin = rho_ngm(chi_min)
    say("rho(NGM) of order 2N at the best minimizer: %.12f (A1 diag(g): %.12f)" % (rmin, best[2]))
    for k in range(3):
        used = np.array([(sig[k][I] * chi_min[k][I]).sum() for I in LOC]) / b[k]
        say("  dose %d: doses used / b at the minimizer in [%.15f, %.15f]" % (k + 1, used.min(), used.max()))

    say("rho(NGM) of the allocations")
    for name, ch in (("chi = 0", [np.zeros(N)] * 3), ("uniform", ref), ("left ev", lev), ("minimizer", chi_min)):
        rr = rho_ngm(ch)
        say("  %-10s rho = %.6f   decrease from chi = 0: %6.2f%%   above the minimizer: %6.3f%%"
            % (name, rr, 100 * (rho0 - rr) / rho0, 100 * (rr - rmin) / rmin))

    say("doses by age, US total, per day (dose k: uniform / left ev / minimizer)")
    print("    %-6s" % "age" + "".join("   %28s" % ("dose %d" % (k + 1)) for k in range(3)), flush=True)
    for j, a_ in enumerate(S7.AGES):
        row = "    %-6s" % a_
        for k in range(3):
            vals = [(sig[k] * ch[k]).reshape(m, n)[:, j].sum() for ch in (ref, lev, chi_min)]
            row += "   %8.0f %9.0f %9.0f" % tuple(vals)
        print(row, flush=True)
    out = "intermediate/three_doses_51x16_mock_%s.npz" % CHI2
    np.savez(out, chi_min=np.array(chi_min), ref=np.array(ref),
             lev=np.array(lev), sig=np.array(sig), b=np.array(b),
             chi1_one=c1[0], hist_one=h1, **{"hist_" + k.replace(" ", "_"): v for k, v in hists.items()})
    say("saved " + out)
    say("done")


if __name__ == "__main__":
    main()
