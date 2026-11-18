"""Matrix-free implicit Runge--Kutta integrator with automatic step size.

The L-stable, stiffly accurate SDIRK method of order 4 with an embedded method of order 3
(Hairer and Wanner, Solving ODEs II, Section IV.6, Table 6.5, gamma = 1/4). Each stage
    z_i = x + h sum_{j<i} a_ij f(t + c_j h, z_j) + h gamma f(t + c_i h, z_i)
is solved by simplified Newton iterations with the matrix I - h gamma J, J = f_x(t, x) at the
start of the step, applied as v -> v - h gamma jvp(t, x, v) and inverted by GMRES (no matrix is
formed). The error estimate is h sum_i (b_i - bhat_i) f_i in the norm
    ||e|| = sqrt(mean((e / (atol + rtol max(|x|, |x_new|)))^2)),
and the step is h_new = h min(5, max(0.2, 0.9 ||e||^(-1/4))).

Usage:
    python3 -u sdirk.py      order conditions of the coefficients, and a stiff test problem
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

GAMMA = 0.25
A = np.array([[1 / 4, 0, 0, 0, 0],
              [1 / 2, 1 / 4, 0, 0, 0],
              [17 / 50, -1 / 25, 1 / 4, 0, 0],
              [371 / 1360, -137 / 2720, 15 / 544, 1 / 4, 0],
              [25 / 24, -49 / 48, 125 / 16, -85 / 12, 1 / 4]])
C = A.sum(axis=1)
B = A[-1].copy()                                           # stiffly accurate: x_new = z_5
BHAT = np.array([59 / 48, -17 / 96, 225 / 32, -85 / 12, 0.0])


def _norm(e, scale):
    return np.sqrt(np.mean((e / scale) ** 2))


def solve(f, jvp, t0, t1, x0, rtol=1e-6, atol=1e-8, t_eval=None, h0=None, newton_tol=0.03,
          newton_max=8, gmres_tol=1e-4):
    """Integrate x' = f(t, x) from t0 to t1. jvp(t, x, v) = f_x(t, x) v. The steps are cut so as
    to end exactly at the points of t_eval (default [t1]). Returns (ts, ys, stats), ys[k] the
    state at ts[k]; stats: nfev, njev (Jacobian-vector products), nsteps, nreject, steps (h),
    h_next (the step proposed after the last one, to pass as h0 when restarting at t1)."""
    t_eval = [t1] if t_eval is None else sorted(t_eval)
    x = np.asarray(x0, float).copy()
    t, n = float(t0), x.size
    st = {"nfev": 0, "njev": 0, "nsteps": 0, "nreject": 0, "steps": []}
    fx = f(t, x)
    st["nfev"] += 1
    if h0 is None:                 # Hairer, Norsett, Wanner, Solving ODEs I, Section II.4, p. 169
        sc = atol + rtol * np.abs(x)
        d0, d1 = _norm(x, sc), _norm(fx, sc)
        h1 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
        h1 = min(h1, t1 - t0)
        d2 = _norm(f(t + h1, x + h1 * fx) - fx, sc) / h1
        st["nfev"] += 1
        h2 = (max(1e-6, h1 * 1e-3) if max(d1, d2) <= 1e-15
              else (0.01 / max(d1, d2)) ** (1 / 5))          # order 4: exponent 1/(4+1)
        h0 = min(100 * h1, h2)
    h = min(h0, t1 - t0)
    ts, ys, k_out = [], [], 0
    while k_out < len(t_eval) and t_eval[k_out] <= t:      # output points at t0
        ts.append(t)
        ys.append(x.copy())
        k_out += 1
    while t < t1 - 1e-12 * max(1.0, abs(t1)):
        target = t_eval[k_out] if k_out < len(t_eval) else t1
        hit = h >= target - t
        hs = target - t if hit else h
        tx, xx = t, x
        Mop = LinearOperator((n, n), dtype=float,
                             matvec=lambda v: v - hs * GAMMA * jvp(tx, xx, v))
        scale0 = atol + rtol * np.abs(x)
        K, ok = np.empty((5, n)), True
        for i in range(5):
            rhs = x + hs * (A[i, :i] @ K[:i]) if i else x.copy()
            z = rhs + hs * GAMMA * (K[i - 1] if i else fx)
            ti = t + C[i] * hs
            conv = False
            for it in range(newton_max):
                fz = f(ti, z)
                st["nfev"] += 1
                r = z - rhs - hs * GAMMA * fz
                nmv = [0]

                def cb(_):
                    nmv[0] += 1
                dz, info = gmres(Mop, -r, tol=gmres_tol, atol=0.0, restart=40, maxiter=10,
                                 callback=cb, callback_type="pr_norm")
                st["njev"] += nmv[0] + 1
                z = z + dz
                if _norm(dz, scale0) < newton_tol:
                    conv = True
                    break
            if not conv:
                ok = False
                break
            K[i] = (z - rhs) / (hs * GAMMA)                    # f(ti, z) from the stage equation
        if not ok:                                         # Newton failed: halve the step
            h = 0.5 * hs
            st["nreject"] += 1
            continue
        xn = x + hs * (B @ K)
        err = _norm(hs * ((B - BHAT) @ K), atol + rtol * np.maximum(np.abs(x), np.abs(xn)))
        fac = min(5.0, max(0.2, 0.9 * err ** -0.25)) if err > 0 else 5.0
        if err > 1.0:
            h = hs * fac
            st["nreject"] += 1
            continue
        t = target if hit else t + hs
        x = xn
        fx = f(t, x)
        st["nfev"] += 1
        st["nsteps"] += 1
        st["steps"].append(hs)
        h = hs * fac if not hit else max(h, hs * fac)
        if hit:
            ts.append(t)
            ys.append(x.copy())
            k_out += 1
    st["h_next"] = h                                       # the proposed next step, for a restart
    return ts, ys, st


def _check_order():
    """The order conditions up to order 4 for b and up to order 3 for bhat."""
    e = np.ones(5)
    conds = {"b.e": (e, 1), "b.c": (C, 1 / 2), "b.c^2": (C ** 2, 1 / 3), "b.Ac": (A @ C, 1 / 6),
             "b.c^3": (C ** 3, 1 / 4), "b.(c*Ac)": (C * (A @ C), 1 / 8),
             "b.Ac^2": (A @ C ** 2, 1 / 12), "b.AAc": (A @ (A @ C), 1 / 24)}
    for nm, (v, val) in conds.items():
        print(f"  {nm:10s} b: {B @ v - val:+.1e}   bhat: {BHAT @ v - val:+.1e}", flush=True)


if __name__ == "__main__":
    print("order conditions (residuals; b up to order 4, bhat up to order 3):", flush=True)
    _check_order()
    # stiff linear test: x' = L (x - g(t)) + g'(t), g = (sin t, cos t, ...), exact x = g
    rng = np.random.default_rng(0)
    nn = 200
    lam = -np.logspace(0, 3, nn)
    Q, _ = np.linalg.qr(rng.standard_normal((nn, nn)))
    Lm = Q @ np.diag(lam) @ Q.T
    w = np.linspace(1, 2, nn)
    g = lambda t: np.sin(w * t)
    gp = lambda t: w * np.cos(w * t)
    f = lambda t, x: Lm @ (x - g(t)) + gp(t)
    jv = lambda t, x, v: Lm @ v
    print("stiff test (eigenvalues -1 to -1e3), t in [0, 10], exact x = sin(w t):", flush=True)
    for tol in (1e-4, 1e-6, 1e-8):
        ts, ys, st = solve(f, jv, 0.0, 10.0, g(0.0), rtol=tol, atol=tol)
        print(f"  rtol=atol={tol:.0e}: error {np.abs(ys[-1] - g(10.0)).max():.2e}, "
              f"{st['nsteps']} steps, {st['nreject']} rejected, {st['nfev']} f, {st['njev']} jvp", flush=True)
