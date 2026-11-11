"""Three doses: the alternating method of Section 4.1 of
min_spectral_radius_vaccination.tex.

Each sweep updates chi_1, chi_3, chi_2 in turn, the other two fixed:
  chi_1  the fixed point iteration of Section 3 with theta = chi_2,
         zeta = chi_3: e_p, omega_p, gamma_p of (t-coeffs), sigma~_p, c_i of
         (t-program), g_p = (1 - e_p) + e_p / t_p;
  chi_3  the same iteration with e_p = 1 and A1 replaced by A1 diag(g0),
         t_p = 1 + Q1_p chi_3p / Q0_p, on the p with chi_1p chi_2p > 0
         (the other p get no third doses: g_p does not depend on chi_3p);
  chi_2  projected Yogi on the chi_2 block (Algorithm t-alg-yogi).
The instance, the objective function, its gradient, the 49 starts and the KKT
check of the joint problem are those of gs_three_dose.py.

Reports per start: rho after each sweep, whether the chi_1 and chi_3 steps
ever increased rho, the final rho against the joint projected Yogi value
1.661826042009, and the KKT conditions of the joint problem at the end.

Usage:
    python3 -u alt_three_dose.py [--only NAME] [--sweeps K] [--out FILE]
"""

import argparse
import time
import numpy as np

import gs_three_dose as T
import gp_single_dose as G
from fixed_point_single_dose import solve_mu, t_of_mu
from yogi_three_dose import proj

m, n, N = T.m, T.n, T.N
e1, e2, e3, om, omv = T.e1, T.e2, T.e3, T.om, T.omv
a1 = T.a1
S_, P_, b1 = T.S_, T.P_, T.b1
SIG = [S_.ravel(), P_.ravel(), P_.ravel()]
B = [b1, b1 / 4, b1 / 4]
RHO_JOINT = 1.661826042009
FP_TOL, FP_MAX = 1e-13, 2000


def rho_of(chi):
    return G.rho_reduced(a1, T.gfun(*chi))[0]


def fixed_point(kappa, eta_, h, sig, c, active, t0):
    """Fixed point iteration of Section 3 for g_p = kappa_p + eta_p h_p / t_p
    on the active p (h_p = 1 for chi_1; h_p = g0_p for chi_3, i.e. A1 diag(g0)),
    with g_p = kappa_p held fixed elsewhere.  Returns t."""
    t = t0.copy()
    for _ in range(FP_MAX):
        g = np.where(active, kappa + eta_ * h / t, kappa)
        rho, v, w = G.rho_reduced(a1, g)
        a = h * (a1.T @ w) * v / float(w @ v)     # (A1 diag(h))^T w * v / <w, v>
        tn = t.copy()
        for i in range(m):
            sl = np.zeros(N, bool)
            sl[i * n:(i + 1) * n] = True
            J = sl & active
            mu = solve_mu(a[J], sig[J], c[i], eta_[J])
            tn[J] = t_of_mu(mu, a[J], sig[J], eta_[J])
        d = np.abs(tn - t).max() / np.abs(t).max()
        t = tn
        if d < FP_TOL:
            break
    return t


def step_chi1(chi):
    th, ze = chi[1], chi[2]
    omp = omv + e2 * th
    gam = 1 + 2 * e2 * th / om + e2 * e3 * th * ze / om ** 2
    e = (e1 + e2 * th / om + e2 * e3 * th * ze / om ** 2) / gam
    sig = SIG[0] * omp / (omv * gam)
    c = B[0] / omv + sig.reshape(m, n).sum(axis=1)
    t0 = 1 + gam / omp * chi[0]
    # check of (t-def): g = (1 - e) + e / t
    assert np.allclose(T.gfun(*chi), (1 - e) + e / t0, rtol=1e-12, atol=0)
    act = np.ones(N, bool)
    t = fixed_point(1 - e, e, np.ones(N), sig, c, act, t0)
    return omp / gam * (t - 1)


def step_chi3(chi):
    x1, x2 = chi[0], chi[1]
    Q0 = omv + x1 + e2 * x2 + 2 * e2 / om * x1 * x2
    Q1 = e2 * e3 / om ** 2 * x1 * x2
    act = Q1 > 0
    for i in range(m):
        assert act[i * n:(i + 1) * n].any(), "location %d: no p with chi1 chi2 > 0" % i
    g0 = T.gfun(x1, x2, np.zeros(N))
    Q1s = np.where(act, Q1, 1.0)
    sig = np.where(act, SIG[2] * Q0 / (omv * Q1s), 0.0)
    c = B[2] / omv + sig.reshape(m, n).sum(axis=1)
    t0 = np.where(act, 1 + Q1s * chi[2] / Q0, 1.0)
    if (~act).any():   # move the third doses of inactive p to the active p of the location
        x3 = np.where(act, chi[2], 0.0).reshape(m, n)
        x3 *= (B[2] / (SIG[2].reshape(m, n) * x3).sum(axis=1))[:, None]
        t0 = np.where(act, 1 + Q1s * x3.ravel() / Q0, 1.0)
    # check of (alt-third): g = g0 / t
    x3c = np.where(act, (Q0 / Q1s) * (t0 - 1), 0.0)
    assert np.allclose(T.gfun(x1, x2, x3c)[act], (g0 / t0)[act], rtol=1e-12, atol=0)
    t = fixed_point(np.where(act, 0.0, g0), np.ones(N), g0, sig, c, act, t0)
    return np.where(act, (Q0 / Q1s) * (t - 1), 0.0)


def step_chi2(chi, tol, tmax, eta=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    """Projected Yogi on u-hat_2 with chi_1, chi_3 fixed."""
    x = np.vstack([chi[k] / T.SCALES[k] for k in range(3)])

    def vg(y):
        z = x.copy()
        z[1] = y
        f, gr = T.value_grad(z.ravel())
        return f, gr.reshape(3, N)[1]

    y = x[1].copy()
    f, q = vg(y)
    mu, s = np.zeros_like(q), q * q
    for it in range(1, tmax + 1):
        mu = beta1 * mu + (1 - beta1) * q
        s = s - (1 - beta2) * np.sign(s - q * q) * q * q
        d = np.sqrt(s) + eps
        z = y - eta * mu / (1 - beta1 ** it) / d
        y = proj(z.reshape(m, n), d.reshape(m, n)).ravel()
        f, q = vg(y)
        r = np.abs(y.reshape(m, n) - proj((y - q).reshape(m, n), np.ones((m, n)))).max()
        if r < tol:
            break
    return T.SCALES[1] * y, it, r


def step_chi2_tr(chi):
    """The chi_2 step by the trust region interior point method on u-hat_2,
    chi_1 and chi_3 fixed (same options as the joint method)."""
    x = np.vstack([chi[k] / T.SCALES[k] for k in range(3)])

    def vg(y):
        z = x.copy()
        z[1] = y
        f, gr = T.value_grad(z.ravel())
        return f, gr.reshape(3, N)[1]

    res = T.solve(vg, x[1], 1)
    y = np.clip(res.x, 0.0, None).reshape(m, n)
    y = (y / y.sum(axis=1, keepdims=True)).ravel()
    return T.SCALES[1] * y, res.nit, 0.0


CHI2_METHOD = "yogi"     # or "tr"


def alternate(x0, sweeps, tol_sweep=1e-12, yogi_tol=1e-10, yogi_tmax=20000, r_tol=None):
    """Sweeps chi_1, chi_3, chi_2.  Stops when the relative change of rho over a
    sweep is below tol_sweep, or, if r_tol is given, instead when the stationarity
    measure r of the joint problem (t-r) is below r_tol; at most `sweeps` sweeps."""
    xr = np.asarray(x0).reshape(3, N)
    chi = [T.SCALES[k] * xr[k] for k in range(3)]
    rho = rho_of(chi)
    hist = [rho]
    up13, up2, yit = 0.0, 0.0, 0
    for sw in range(1, sweeps + 1):
        rho_old = rho
        chi[0] = step_chi1(chi)
        r1 = rho_of(chi)
        chi[2] = step_chi3(chi)
        r3 = rho_of(chi)
        if CHI2_METHOD == "tr":
            chi[1], it, _ = step_chi2_tr(chi)
        else:
            chi[1], it, _ = step_chi2(chi, yogi_tol, yogi_tmax)
        yit += it
        r2 = rho_of(chi)
        up13 = max(up13, (r1 - rho_old) / rho_old, (r3 - r1) / r1)
        up2 = max(up2, (r2 - r3) / r3)
        rho = r2
        hist.append(rho)
        if r_tol is not None:
            xs = np.concatenate([chi[k] / T.SCALES[k] for k in range(3)])
            from yogi_three_dose import stationarity
            if stationarity(xs, T.value_grad(xs)[1]) < r_tol:
                break
        elif abs(rho - rho_old) / rho_old < tol_sweep:
            break
    x = np.concatenate([chi[k] / T.SCALES[k] for k in range(3)])
    return x, sw, hist, up13, up2, yit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--polish", action="store_true",
                    help="after convergence, run joint projected Yogi from the endpoint")
    ap.add_argument("--pre", type=int, default=0,
                    help="run this many joint projected Yogi iterations before alternating")
    a = ap.parse_args()
    rows, t_all = [], time.time()
    for name, x0 in T.starts():
        if a.only is not None and name != a.only:
            continue
        t0 = time.time()
        if a.pre > 0:
            from yogi_three_dose import yogi
            x0, _, _ = yogi(x0, 1e-2, 0.9, 0.999, 1e-8, 1e-10, a.pre)
        x, sw, hist, up13, up2, yit = alternate(x0, a.sweeps)
        el = time.time() - t0
        rho = hist[-1]
        if a.polish:
            from yogi_three_dose import yogi
            t1 = time.time()
            xj, itj, rj = yogi(x, 1e-2, 0.9, 0.999, 1e-8, 1e-10, 200000)
            elj = time.time() - t1
            rhoj = np.exp(T.value_grad(xj)[0])
            sp, ex = T.kkt(xj)
            rows.append((name, rhoj, (rhoj - RHO_JOINT) / RHO_JOINT, sw, el, rho, itj, elj, rj, sp, ex))
            print("%-12s rho=%.12f rel-joint=%+.1e | alt sweeps=%3d %5.1fs rho_alt=%.10f | "
                  "joint Yogi it=%6d %6.1fs r=%.1e kkt %.1e %+.1e" % rows[-1], flush=True)
            continue
        sp, ex = T.kkt(x)
        rows.append((name, rho, (rho - RHO_JOINT) / RHO_JOINT, sw, yit, el, up13, up2, sp, ex))
        print("%-12s rho=%.12f rel-joint=%+.1e sweeps=%3d yogi-it=%6d %6.1fs "
              "max up(1,3)=%+.1e max up(2)=%+.1e kkt %.1e %+.1e" % rows[-1], flush=True)
        if a.only is not None:
            print("rho per sweep:", " ".join("%.12f" % r for r in hist))
    print("total %.1f s" % (time.time() - t_all))
    vals = np.array([r[1] for r in rows])
    print("starts %d, min %.12f max %.12f, distinct to 1e-8: %s"
          % (len(vals), vals.min(), vals.max(), np.unique(np.round(vals, 8))))
    if a.out is not None:
        with open(a.out, "w") as fh:
            if a.polish:
                fh.write("# start rho rel_joint sweeps t_alt rho_alt joint_it t_joint r spread excess\n")
                fmt = "%s %.12f %.3e %d %.2f %.12f %d %.2f %.3e %.3e %.3e\n"
            else:
                fh.write("# start rho rel_joint sweeps yogi_it seconds up13 up2 spread excess\n")
                fmt = "%s %.12f %.3e %d %d %.2f %.3e %.3e %.3e %.3e\n"
            for r in rows:
                fh.write(fmt % ((r[0].replace(" ", "_"),) + r[1:]))
        print("written", a.out)
