"""The alternating method of notes/stratified_seir_ngm.tex for three doses, 51 x 16 model.

Problem (1.1) of the notes: minimize rho(NGM) over the vaccination rates chi_1, chi_2,
chi_3 (first, second, third dose; per day), subject to

    sum_{p in I_i} sigma_{k,p} chi_{k,p} = b_{k,i},   chi_{k,p} >= 0,

for every dose k and location i.  rho(NGM) = rho(A_1 diag(g)) (Theorem 2.1), A_1 of
order N = 816 independent of chi, g_p a function of chi_{1,p}, chi_{2,p}, chi_{3,p}.

The method (Section 6): sweeps over chi_1, chi_3, chi_2, each step minimizing rho over one
dose with the other two fixed, by the damped fixed point iteration of Section 5.2
(t_block, damping THETA = 1/2):
  chi_1 (Section 6.1): g = (1 - e) + e / t;
  chi_3 (Section 6.2): g = g0 / t, over the groups with chi_1 > 0 and chi_2 > 0;
  chi_2 (Section 6.3): g = g0 ((1 - e) + e / t), over the groups with Delta_p > 0.
A step is done by spectral projected gradient instead (block_spg) if C is not
irreducible, if (chi_2) a location has no group with Delta_p > 0, or if the fixed point
iteration does not converge (fp_step).  The sweeps stop when rho changes by less than
1e-11 (relative).  The method is run from several starts (starts); the end point with
the smallest rho is the solution.

Nothing is loaded on import: the model is given by set_model(), from
seir7_ode.Model.ngm_context (the data files of notes/DATA.tex). The doses b and the group
sizes sigma are arguments (see seir7_ode.optimal_rates).
"""

import time

import numpy as np
import scipy.sparse.linalg as spsl

T0 = time.time()
THETA = 0.5          # damping of the fixed point iteration (Section 5.2); 1 = undamped
# wall time (s), iterations and steps of the dose-k steps of alternate (k = 0, 1, 2), added up
# from the last reset_stats()
STATS = {"time": [0.0, 0.0, 0.0], "iters": [0, 0, 0], "steps": [0, 0, 0]}


def reset_stats():
    for v in STATS.values():
        v[:] = [0] * 3


def set_model(m_, n_, C_, dc_, psq_, r_, om_, omv_, e1_, e2_, e3_, irreducible=True):
    """The model of the problem: C (N x N, a matrix or a LinearOperator with C @ x and
    C.T @ x), d_c, psi_I q_s and q_m / q_s per group (length N = m n), and omega, omega_v,
    eps_1, eps_2, eps_3."""
    global m, n, N, C, dc, psq, r, om, omv, e1, e2, e3, LOC, IRREDUCIBLE
    m, n, N = m_, n_, m_ * n_
    C, dc, psq, r = C_, np.asarray(dc_, float), np.asarray(psq_, float), np.asarray(r_, float)
    om, omv, e1, e2, e3 = om_, omv_, e1_, e2_, e3_
    LOC = [np.arange(i * n, (i + 1) * n) for i in range(m)]
    IRREDUCIBLE = irreducible


ELIG = None          # None, or for k = 0, 1, 2 a boolean array (length N): the groups that may receive dose k


def set_eligible(elig):
    """Only the groups with elig[k][p] True may receive doses of type k (None: all groups).
    A location with no eligible group must have b_{k,i} = 0."""
    global ELIG
    ELIG = None if elig is None else [np.asarray(e, bool) for e in elig]


def _elig(k):
    return np.ones(N, bool) if ELIG is None else ELIG[k]


CHI_CAP = None       # None, or the bound chi_{k,p} <= CHI_CAP added to the constraints of (1.1)


def set_cap(cap):
    """Add the constraint chi_{k,p} <= cap for all k, p (None: no bound). Feasible when
    b_{k,i} <= cap sum sigma_{k,p} over the eligible groups of each location."""
    global CHI_CAP
    CHI_CAP = cap


def solve_mu_capped(h, st, bi, tmax, iters=200):
    """mu with sum_p st_p (t_p(mu) - 1) = bi, t_p(mu) = min(tmax_p, max(1, sqrt(h_p / (mu st_p)))):
    the frozen step of Lemma 5.1 with the bound t_p <= tmax_p (the separable convex problem
    with box constraints). F(mu) = sum st (t - 1) - bi is nonincreasing in mu; bisection in
    log mu between a mu with all t_p = tmax_p (F >= 0, feasibility) and one with all t_p = 1."""
    hp = np.maximum(h, 0.0)
    pos = hp > 0
    if not pos.any():
        return np.inf
    ratio = hp[pos] / st[pos]
    lo = np.log(ratio.min() / tmax[pos].max() ** 2) - 1.0         # all t = tmax
    hi = np.log(ratio.max()) + 1.0                                 # all t = 1

    def F(lmu):
        t = np.minimum(tmax, np.maximum(1.0, np.sqrt(hp / (np.exp(lmu) * st))))
        return (st * (t - 1.0)).sum() - bi

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if F(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return np.exp(0.5 * (lo + hi))


def proj_capped_simplex(y, ub, iters=200):
    """Euclidean projection onto {0 <= u <= ub, sum u = 1} (sum ub >= 1), by bisection on the
    shift lam in u = clip(y - lam, 0, ub)."""
    lo, hi = (y - ub).min() - 1.0, y.max()
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        if np.clip(y - lam, 0.0, ub).sum() > 1.0:
            lo = lam
        else:
            hi = lam
        if hi - lo < 1e-15:
            break
    return np.clip(y - 0.5 * (lo + hi), 0.0, ub)


m = n = N = None      # set by set_model


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


# ----------------------------------------------------------------------------- the model
def pis(chi1, chi2, chi3):
    """pi_1, pi_2, pi_3 of (1.2)."""
    return chi1 / (omv + e2 * chi2), e2 * chi2 / om, 1.0 + e3 * chi3 / om


def g_of(chi1, chi2, chi3):
    """g = N / D of (2.4); returns g, N, D."""
    p1, p2, p3 = pis(chi1, chi2, chi3)
    num = 1.0 + r * p1 * (1.0 - e1 + p2)
    den = 1.0 + p1 * (1.0 + p2 + p2 * p3)
    return num / den, num, den


def perron(g):
    """rho, v, w of A1 diag(g), A1 = diag(dc) C diag(psq), by Arnoldi."""
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
    """rho and d rho / d g_p = (A1^T w)_p v_p / w^T v, (3.3)."""
    rho, v, w = perron(g)
    return rho, (psq * (C.T @ (dc * w))) * v / float(w @ v)


def dg_all(chi):
    """d g_p / d chi_{k,p}, k = 1, 2, 3, (3.4)."""
    chi1, chi2, chi3 = chi
    p1, p2, p3 = pis(chi1, chi2, chi3)
    _, num, den = g_of(chi1, chi2, chi3)
    omp = omv + e2 * chi2
    alpha = 1.0 + p2 + p2 * p3
    e = 1.0 - r / alpha * (1.0 - e1 + p2)
    dN = r * e2 * p1 * (1.0 / om - (1.0 - e1 + p2) / omp)
    dD = e2 * p1 * ((1.0 + p3) / om - alpha / omp)
    return [-alpha * e / (omp * den**2), (den * dN - num * dD) / den**2,
            -e3 * p1 * p2 * num / (om * den**2)]


# ----------------------------------------------------------------------------- one step
def solve_mu(h, sg, bi):
    """(5.7): the solution mu of sum_p sg_p max(1, sqrt(h_p / (mu sg_p))) = c_i,
    c_i = bi + sum_p sg_p, written with the tail sums c_i - C_j = bi + sum_{l > j} sg_(l)
    so that groups with very large sg_p (very small slope) cause no cancellation."""
    beta = h / sg
    o = np.argsort(beta)
    beta, s, rr = beta[o], sg[o], np.sqrt(h * sg)[o]
    q = len(beta)
    tail = np.concatenate([np.cumsum(s[::-1])[::-1], [0.0]])   # tail[j] = sum_{l >= j} s_l
    B = np.concatenate([np.cumsum(rr[::-1])[::-1], [0.0]])
    j = 0
    for jj in range(1, q):
        if B[jj] / np.sqrt(beta[jj - 1]) - tail[jj] >= bi:      # F(beta_(jj-1)) >= c_i
            j = jj
    return (B[j] / (bi + tail[j])) ** 2


def chi2_form(chi):
    """Section 6.3: g_p = (kappa + kappa' chi2) / (lambda + lambda' chi2),
    Delta_p = kappa lambda' - kappa' lambda; for Delta_p > 0,
    g_p = g0_p ((1 - e_p) + e_p / t_p), t_p = 1 + (lambda' / lambda) chi2.
    Returns Delta, g0, e, slope = lambda' / lambda."""
    c1, _, c3 = chi
    ka = omv + r * (1.0 - e1) * c1
    kp = e2 * (1.0 + r * c1 / om)
    la = omv + c1
    lp = e2 * (1.0 + (2.0 + e3 * c3 / om) * c1 / om)
    Delta = ka * lp - kp * la
    return Delta, ka / la, Delta / (ka * lp), lp / la


def t_block(chi, k, b, sig, tol=1e-12, kmax=300, theta=None, out=None):
    """Minimize rho over chi_k (k = 0, 1, 2 for the first, second, third dose), the other
    doses fixed, by the damped fixed point iteration t <- t + theta (t(mu) - t) of Section
    5.2 (Lemma 5.1, (5.7)), in the one dose form of Sections 6.1-6.3 (g_p affine in 1 / t_p,
    t_p = 1 + slope_p chi_{k,p}; the groups with slope_p = 0 get no doses of type k).
    Stops when |t(mu) - t| / |t| < tol.  Returns the new chi and the number of iterations;
    out["converged"] tells whether the tolerance was reached."""
    theta = THETA if theta is None else theta
    chi = [c.copy() for c in chi]
    c1, c2, c3 = chi
    if k == 0:                                                # Section 6.1
        p2 = e2 * c2 / om
        omp = omv + e2 * c2
        alpha = 1.0 + 2.0 * e2 * c2 / om + e2 * e3 * c2 * c3 / om**2
        slope = alpha / omp
        e = 1.0 - r / alpha * (1.0 - e1 + p2)
        coef = lambda gd: e * gd                              # h_p = e_p (d rho / d g)_p
    elif k == 1:                                              # Section 6.3
        Delta, g0, e, slope = chi2_form(chi)
        slope = np.where(Delta > 0, slope, 0.0)
        coef = lambda gd: g0 * e * gd
    else:                                                     # Section 6.2
        p1, p2, _ = pis(c1, c2, np.zeros(N))
        D0 = 1.0 + p1 * (1.0 + 2.0 * p2)
        slope = e3 * p1 * p2 / (om * D0)
        g0 = g_of(c1, c2, np.zeros(N))[0]
        coef = lambda gd: g0 * gd
    act = (slope > 0) & _elig(k)
    tval = 1.0 + slope * chi[k]
    tmax = 1.0 + slope * CHI_CAP if CHI_CAP is not None else None
    d = np.inf
    for it in range(kmax):
        _, gd = drho_dg(g_of(*chi)[0])
        h = coef(gd)
        tn = tval.copy()
        for i, I in enumerate(LOC):
            I = I[act[I]]
            if b[k][i] <= 0 or I.size == 0:
                continue
            st = sig[k][I] / slope[I]
            if tmax is None:
                mu = solve_mu(h[I], st, b[k][i])
                tn[I] = np.maximum(1.0, np.sqrt(h[I] / (mu * st)))
            else:
                mu = solve_mu_capped(h[I], st, b[k][i], tmax[I])
                tn[I] = np.minimum(tmax[I], np.maximum(1.0, np.sqrt(np.maximum(h[I], 0.0) / (mu * st))))
        d = np.abs(tn - tval).max() / tval.max()
        tval = tval + theta * (tn - tval)
        chi[k] = np.where(act, (tval - 1.0) / np.where(act, slope, 1.0), 0.0)
        if d < tol:
            break
    if out is not None:
        out["converged"] = bool(d < tol)
    return chi, it + 1


# ----------------------------------------------------------------------------- fallback
def proj_simplex(y):
    """Euclidean projection onto {u >= 0, sum u = 1}."""
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - 1.0
    j = np.nonzero(u - css / np.arange(1, y.size + 1) > 0)[0][-1]
    return np.maximum(y - css[j] / (j + 1), 0.0)


def proj(y, k=None, ub=None):
    """Projection onto the product of the simplices of the locations; with k, the simplex of
    location i is that of its groups eligible for dose k (zero on the others); with ub, the
    simplex is capped, u_p <= ub_p."""
    if k is None or (ELIG is None and ub is None):
        return np.concatenate([proj_simplex(y[I]) for I in LOC])
    out = np.zeros_like(y)
    el = _elig(k)
    for I in LOC:
        E = I[el[I]]
        if E.size:
            out[E] = proj_simplex(y[E]) if ub is None else proj_capped_simplex(y[E], ub[E])
    return out


def _ub(k, b, sig):
    """u_p = sigma_{k,p} chi_{k,p} / b_{k,i} <= sigma_{k,p} CHI_CAP / b_{k,i}, or None."""
    if CHI_CAP is None:
        return None
    bk = np.repeat(b[k], n)
    return np.where(bk > 0, sig[k] * CHI_CAP / np.where(bk > 0, bk, 1.0), 1.0)   # b = 0: u <= 1, no bound


def spg(fg, f, u, P, maxiter=500, tol=1e-12, M=10, amin=1e-12, amax=1e12):
    """Spectral projected gradient (Birgin, Martinez, Raydan 2000): minimize f over the set
    onto which P projects; fg(u) -> (f, grad).  Barzilai-Borwein step, nonmonotone Armijo
    line search over the last M values.  Returns the best point seen and the iterations."""
    fu, g = fg(u)
    hist = [fu]
    best = (fu, u)
    a = 1.0 / max(np.abs(P(u - g) - u).max(), 1e-30)
    for it in range(maxiter):
        if np.abs(P(u - g) - u).max() < tol:
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
    return best[1], it + 1


def block_spg(chi, k, b, sig, maxiter=500):
    """The step for dose k by spg in u_p = sigma_{k,p} chi_{k,p} / b_{k,i} (a simplex per
    location), the other doses fixed."""
    bk = np.repeat(b[k], n)
    bsafe = np.where(bk > 0, bk, 1.0)

    def to_chi(u):
        ch = list(chi)
        ch[k] = u * bk / sig[k]
        return ch

    def fg(u):
        ch = to_chi(u)
        rho, gd = drho_dg(g_of(*ch)[0])
        return rho, gd * dg_all(ch)[k] * bk / sig[k]

    ub = _ub(k, b, sig)
    u0 = proj(sig[k] * chi[k] / bsafe, k, ub)
    u, it = spg(fg, lambda u: rho_of(to_chi(u)), u0, lambda y: proj(y, k, ub), maxiter=maxiter)
    return to_chi(u), it


def fp_step(chi, k, b, sig, label=""):
    """The step for dose k: t_block, or block_spg if C is not irreducible, if (k = 1) a
    location has no group with Delta_p > 0, or if t_block does not converge.
    Returns the new chi, the number of iterations and the method used."""
    reason = None
    if not IRREDUCIBLE:
        reason = "C is not irreducible"
    elif k == 1 and not (chi2_form(chi)[0] > 0).reshape(m, n).any(axis=1).all():
        reason = "a location has no group with Delta_p > 0"
    if reason is None:
        out = {}
        new, it = t_block(chi, k, b, sig, out=out)
        if out["converged"]:
            return new, it, "fp"
        reason = "the fixed point iteration did not converge"
    say("%s dose %d step: %s; spg instead" % (label, k + 1, reason))
    new, it = block_spg(chi, k, b, sig)
    return new, it, "spg"


def _timed_step(chi, k, b, sig, label):
    """fp_step, with its wall time and iterations added to STATS[...][k]."""
    tic = time.time()
    out = fp_step(chi, k, b, sig, label)
    STATS["time"][k] += time.time() - tic
    STATS["iters"][k] += out[1]
    STATS["steps"][k] += 1
    return out


# ----------------------------------------------------------------------------- the method
def alternate(chi, b, sig, sweeps=200, tol=1e-11, label=""):
    """The alternating method from chi.  Returns chi, rho, the number of sweeps and the
    history of rho (the start, then after each step chi1, chi3, chi2 of every sweep)."""
    rho = rho_of(chi)
    hist = [rho]
    say("%s start: rho = %.12f" % (label, rho))
    for sw in range(sweeps):
        lab = "%s sweep %d" % (label, sw)
        chi, it1, _ = _timed_step(chi, 0, b, sig, lab)
        r1 = rho_of(chi)
        chi, it3, _ = _timed_step(chi, 2, b, sig, lab)
        r3 = rho_of(chi)
        chi, it2, _ = _timed_step(chi, 1, b, sig, lab)
        r2 = rho_of(chi)
        hist += [r1, r3, r2]
        say("%s sweep %3d: after chi1 %.12f (%d it), chi3 %.12f (%d it), chi2 %.12f (%d it)"
            % (label, sw, r1, it1, r3, it3, r2, it2))
        if abs(rho - r2) < tol * r2:
            break
        rho = r2
    say("%s end after %d sweeps: rho = %.12f" % (label, sw + 1, r2))
    return chi, r2, sw + 1, np.array(hist)


def starts(b, sig, nrand, seed):
    """The starts, all satisfying the constraints: "uniform" (the same rate for all groups
    of a location), "left ev" (doses in proportion to the susceptible rows of the dominant
    left eigenvector of NGM at the uniform allocation) and nrand random allocations
    (doses split at random among the groups of each location)."""
    def _uniform(k, i, I):
        e = _elig(k)[I]
        s = sig[k][I][e].sum()
        return np.where(e, b[k][i] / s if s > 0 else 0.0, 0.0)

    uni = [np.concatenate([_uniform(k, i, I) for i, I in enumerate(LOC)]) for k in range(3)]
    wS = np.abs(perron(g_of(*uni)[0])[2])   # left Perron vector of A_1 diag(g) at the uniform allocation
    def _share(k, i, I, wt):          # doses b_{k,i} in proportion to wt over the eligible groups
        wt = np.where(_elig(k)[I], wt, 0.0)
        return b[k][i] * wt / wt.sum() / sig[k][I] if wt.sum() > 0 else np.zeros(I.size)

    lev = [np.concatenate([_share(k, i, I, wS[I]) for i, I in enumerate(LOC)]) for k in range(3)]
    out = [("uniform", uni), ("left ev", lev)]
    rng = np.random.default_rng(seed)
    for s in range(nrand):
        alloc = [np.concatenate([_share(k, i, I, rng.random(n)) for i, I in enumerate(LOC)])
                 for k in range(3)]
        out.append(("random %d" % (s + 1), alloc))
    if CHI_CAP is not None:           # project every start onto the capped simplices
        capped = []
        for name, chi in out:
            new = []
            for k in range(3):
                bk = np.repeat(b[k], n)
                bs = np.where(bk > 0, bk, 1.0)
                u = proj(sig[k] * chi[k] / bs, k, _ub(k, b, sig))
                new.append(u * bk / sig[k])
            capped.append((name, new))
        out = capped
    return out


def kkt_table(chi, b, sig, thr=1e-9):
    """The KKT conditions (4.1): for each dose k and location i, h_p = -(d rho / d chi_{k,p})
    / sigma_{k,p} and mu the mean of h over the groups with u_p = sigma_{k,p} chi_{k,p} / b_{k,i}
    > thr.  Returns, per dose, the largest over i of: the spread max |h_p / mu - 1| over those
    groups (0 at a KKT point), the ratio max h_p / mu over the other groups (at most 1),
    |sum_p u_p - 1|, and the number of groups with no doses."""
    _, gd = drho_dg(g_of(*chi)[0])
    dg = dg_all(chi)
    out = {}
    for k in range(3):
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
